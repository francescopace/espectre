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

std::string session_hex(const uint8_t *session_id) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string value(ESPECTRE_RAW_CSI_SESSION_ID_BYTES * 2U, '0');
  for (size_t index = 0U; index < ESPECTRE_RAW_CSI_SESSION_ID_BYTES; ++index) {
    value[index * 2U] = kHex[session_id[index] >> 4U];
    value[index * 2U + 1U] = kHex[session_id[index] & 0x0FU];
  }
  return value;
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

}  // namespace

void RawCsiSessionController::configure(IDirectHttpService *service,
                                        RuntimeFrontendController *runtime,
                                        uint64_t device_id,
                                        std::string chip,
                                        StoppedCallback stopped_callback) {
  service_ = service;
  runtime_ = runtime;
  device_id_ = device_id;
  chip_ = std::move(chip);
  stopped_callback_ = std::move(stopped_callback);
}

bool RawCsiSessionController::handle_command(const EspectreCommand &command,
                                             const FrontendCommandContext &context,
                                             std::string *code,
                                             std::string *message,
                                             std::string *data_json) {
  if (service_ == nullptr || runtime_ == nullptr || device_id_ == 0U ||
      !runtime_->capabilities().supports_raw_csi) {
    if (code != nullptr) *code = "unsupported";
    if (message != nullptr) *message = "raw CSI collection is unavailable";
    return false;
  }
  if (command.command == "stop_raw_stream") {
    if (authorization_.empty() || context.authorization != authorization_) {
      if (code != nullptr) *code = "not_raw_session_owner";
      if (message != nullptr) *message = "the raw CSI bearer does not own this session";
      return false;
    }
    const bool stopped = service_->stop_raw_session(RawCsiStopReason::REQUESTED);
    if (code != nullptr) *code = stopped ? "ok" : "unavailable";
    if (message != nullptr) {
      *message = stopped ? "raw CSI collection stopped" : "raw CSI collection could not be stopped";
    }
    return stopped;
  }
  if (!authorization_.empty() ||
      runtime_->operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    if (code != nullptr) *code = "busy_raw_collection";
    if (message != nullptr) *message = "a raw CSI session is already active";
    return false;
  }

  RawCsiSessionConfig session;
  session.device_id = device_id_;
  session.chip = raw_chip_type(chip_);
  fill_session_id(session.session_id, device_id_ ^ context.connection_token);
  if (!service_->start_raw_session(
          session, [this](RawCsiStopReason reason) { handle_stopped_(reason); })) {
    if (code != nullptr) *code = "busy_raw_collection";
    if (message != nullptr) *message = "the raw CSI collector is busy";
    return false;
  }
  if (!runtime_->start_raw_collection(&offer_packet_, this)) {
    (void) service_->stop_raw_session(RawCsiStopReason::INTERNAL_ERROR);
    if (code != nullptr) *code = "unavailable";
    if (message != nullptr) *message = "raw CSI capture could not be started";
    return false;
  }

  authorization_ = session_hex(session.session_id);
  if (code != nullptr) *code = "ok";
  if (message != nullptr) *message = "raw CSI collection started";
  if (data_json != nullptr) {
    *data_json = "{\"session_id\":\"" + authorization_ +
                 "\",\"endpoint\":\"" + ESPECTRE_RAW_CSI_ENDPOINT +
                 "\",\"transport\":\"http\",\"protocol_version\":2,"
                 "\"record_version\":8,\"frame_prefix_bytes\":60}";
  }
  return true;
}

void RawCsiSessionController::ensure_runtime_consistency() {
  if (!authorization_.empty() && runtime_ != nullptr && service_ != nullptr &&
      runtime_->operation_state() != RuntimeOperationState::RAW_COLLECTION) {
    (void) service_->stop_raw_session(RawCsiStopReason::INTERNAL_ERROR);
  }
}

void RawCsiSessionController::shutdown(RawCsiStopReason reason) {
  if (!authorization_.empty() && service_ != nullptr) {
    (void) service_->stop_raw_session(reason);
  }
  authorization_.clear();
}

bool RawCsiSessionController::offer_packet_(void *context,
                                            const RawCsiPacketView &packet) {
  auto *controller = static_cast<RawCsiSessionController *>(context);
  return controller != nullptr && controller->service_ != nullptr &&
         controller->service_->offer_raw_packet(packet);
}

void RawCsiSessionController::handle_stopped_(RawCsiStopReason reason) {
  authorization_.clear();
  if (runtime_ != nullptr) (void) runtime_->stop_raw_collection(reason);
  if (stopped_callback_) stopped_callback_(reason);
}

}  // namespace espectre
