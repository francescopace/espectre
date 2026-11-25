/*
 * ESPectre - Raw CSI Session Controller
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "raw_csi_session_controller.h"

#include <cctype>
#include <utility>

#include <esp_timer.h>
#if defined(ESP_PLATFORM)
#include <esp_random.h>
#endif

namespace espectre {
namespace {

RawCsiChipType raw_chip_type(const std::string &chip) {
  std::string normalized;
  for (char character : chip) {
    if (std::isalnum(static_cast<unsigned char>(character))) {
      normalized.push_back(
          static_cast<char>(std::tolower(static_cast<unsigned char>(character))));
    }
  }
  if (normalized == "esp32c3" || normalized == "c3") return RawCsiChipType::C3;
  if (normalized == "esp32c5" || normalized == "c5") return RawCsiChipType::C5;
  if (normalized == "esp32c6" || normalized == "c6") return RawCsiChipType::C6;
  if (normalized == "esp32s2" || normalized == "s2") return RawCsiChipType::S2;
  if (normalized == "esp32s3" || normalized == "s3") return RawCsiChipType::S3;
  if (normalized == "esp32") return RawCsiChipType::ESP32;
  return RawCsiChipType::UNKNOWN;
}

void fill_session_id(uint8_t *session_id, uint64_t seed) {
#if defined(ESP_PLATFORM)
  esp_fill_random(session_id, ESPECTRE_RAW_CSI_SESSION_ID_BYTES);
#else
  seed ^= static_cast<uint64_t>(esp_timer_get_time());
  for (size_t index = 0U; index < ESPECTRE_RAW_CSI_SESSION_ID_BYTES; ++index) {
    seed ^= seed << 13U;
    seed ^= seed >> 7U;
    seed ^= seed << 17U;
    session_id[index] = static_cast<uint8_t>(seed);
  }
#endif
}

bool offer_packet(void *context, const RawCsiPacketView &packet) {
  return static_cast<IDirectHttpService *>(context)->offer_raw_packet(packet);
}

}  // namespace

RawCsiSessionController::Binding::~Binding() { detach(); }

void RawCsiSessionController::Binding::Session::stop_capture(RawCsiStopReason reason) {
  if (auto *owner = std::exchange(runtime, nullptr)) owner->stop_raw_collection(reason);
}

void RawCsiSessionController::Binding::stop(IDirectHttpService &transport, RawCsiStopReason reason) {
  const auto current = session;
  if (!current || current->runtime == nullptr) return;
  transport.stop_raw_session(reason);
  // An asynchronous transport delivers its final reason from the frontend loop.
  current->stop_capture(reason);
}

void RawCsiSessionController::Binding::detach() {
  auto *transport = std::exchange(service, nullptr);
  if (transport == nullptr) return;
  transport->set_raw_session_requested_callback({});
  stop(*transport, RawCsiStopReason::SHUTDOWN);
}

void RawCsiSessionController::configure(IDirectHttpService *service,
                                        RuntimeFrontendController *runtime,
                                        uint64_t device_id,
                                        std::string chip,
                                        StoppedCallback stopped_callback,
                                        StartedCallback started_callback) {
  detach();
  if (service == nullptr || runtime == nullptr) return;
  if (!observer_lifetime_) observer_lifetime_ = std::make_shared<char>();
  binding_ = std::make_shared<Binding>();
  binding_->observer_lifetime = observer_lifetime_;
  binding_->service = service;
  binding_->runtime = runtime;
  binding_->device_id = device_id;
  binding_->chip = std::move(chip);
  if (stopped_callback) binding_->stopped_callback = std::make_shared<StoppedCallback>(std::move(stopped_callback));
  binding_->started_callback = std::move(started_callback);
  // A queued service callback can outlive shutdown and even the next configure().
  service->set_raw_session_requested_callback([weak = std::weak_ptr<Binding>(binding_)](std::string *message) {
    auto binding = weak.lock();
    return binding && binding->begin(message);
  });
}

void RawCsiSessionController::set_wifi_update_pending_callback(std::function<bool()> callback) {
  if (binding_) binding_->wifi_update_pending = std::move(callback);
}

bool RawCsiSessionController::Binding::begin(std::string *message) {
  if (service == nullptr || runtime == nullptr || device_id == 0U ||
      !runtime->capabilities().supports_raw_csi) {
    if (message != nullptr) *message = "raw CSI collection is unavailable";
    return false;
  }
  if ((session && session->runtime) || runtime->operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    if (message != nullptr) *message = "a CSI collection is already active";
    return false;
  }
  if (runtime->wifi_update_pending() || (wifi_update_pending && wifi_update_pending())) {
    if (message != nullptr) *message = "a Wi-Fi update is in progress";
    return false;
  }
  // The runtime refuses collection while sensing is paused; say so instead of a capture failure.
  if (!runtime->services_armed()) {
    if (message != nullptr) *message = "sensing is paused";
    return false;
  }
  RawCsiSessionConfig config;
  config.device_id = device_id;
  config.chip = raw_chip_type(chip);
  fill_session_id(config.session_id, device_id);
  auto current = std::make_shared<Session>(Session{runtime, stopped_callback});
  if (!service->start_raw_session(
          config, [current, lifetime = observer_lifetime](RawCsiStopReason reason) {
            const auto notify = std::move(current->stopped_callback);
            current->stop_capture(reason);
            if (!lifetime.expired() && notify) (*notify)(reason);
          })) {
    if (message != nullptr) *message = "the CSI collector is busy";
    return false;
  }
  // The transport is ours now. A listener may detach it from inside runtime startup.
  session = current;
  const bool started = runtime->start_raw_collection(offer_packet, service);
  if (service == nullptr || !started || runtime->operation_state() != RuntimeOperationState::RAW_COLLECTION) {
    current->runtime = nullptr;
    if (service != nullptr) {
      service->stop_raw_session(RawCsiStopReason::INTERNAL_ERROR);
    } else if (started) {
      runtime->stop_raw_collection(RawCsiStopReason::SHUTDOWN);
    }
    if (message != nullptr) *message = "CSI capture could not be started";
    return false;
  }
  if (started_callback) started_callback();
  if (message != nullptr) *message = "CSI collection started";
  return true;
}

bool RawCsiSessionController::begin(std::string *message) {
  const auto binding = binding_;
  if (binding) return binding->begin(message);
  if (message != nullptr) *message = "raw CSI collection is unavailable";
  return false;
}

bool RawCsiSessionController::active() const { return binding_ && binding_->session && binding_->session->runtime; }

void RawCsiSessionController::ensure_runtime_consistency() {
  if (active() && binding_->runtime->operation_state() != RuntimeOperationState::RAW_COLLECTION) {
    binding_->service->stop_raw_session(RawCsiStopReason::INTERNAL_ERROR);
  }
}

void RawCsiSessionController::shutdown(RawCsiStopReason reason) {
  const auto binding = binding_;
  if (binding && binding->service) binding->stop(*binding->service, reason);
}

void RawCsiSessionController::detach() {
  auto binding = std::move(binding_);
  if (binding) binding->detach();
}

}  // namespace espectre
