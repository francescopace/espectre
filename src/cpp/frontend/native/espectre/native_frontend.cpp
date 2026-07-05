/*
 * ESPectre - Native Frontend Adapter
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "native_frontend.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "ble_protocol.h"
#include "espectre_log.h"
#include "esp_timer.h"
#include "frontend_control_helpers.h"
#include "frontend_mqtt_helpers.h"
#include "frontend_sysinfo_helpers.h"
#include "runtime_time.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "runtime_listener_utils.h"
#include "sdkconfig.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "espectre.native";
constexpr uint8_t kTelemetryMotionStateIdle = 0;
constexpr uint8_t kTelemetryMotionStateMotion = 1;

float current_free_memory_kb() {
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  return static_cast<float>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT)) / 1024.0f;
#else
  return 0.0f;
#endif
}

}  // namespace

NativeFrontend::NativeFrontend(IBleBindings *bindings) : bindings_(bindings) {}

NativeFrontend::NativeFrontend(IBleBindings *bindings, IMqttTransport *mqtt_transport, IOtaService *ota_service)
    : bindings_(bindings), mqtt_transport_(mqtt_transport), ota_service_(ota_service) {}

void NativeFrontend::set_runtime_config(const RuntimeConfig &config) { runtime_.set_config(config); }

void NativeFrontend::set_device_config(const EspectreDeviceConfig &config) {
  device_config_ = config;
  if (bindings_ != nullptr) {
    const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(device_config_),
                                                         device_info_.chip.empty() ? nullptr
                                                                                   : device_info_.chip.c_str());
    bindings_->set_device_name(device_name.c_str());
  }
}

void NativeFrontend::set_device_info(const EspectreDeviceInfo &info) {
  device_info_ = info;
  if (client_connected_) {
    send_system_info_();
  }
}

void NativeFrontend::set_wifi_provisioning_info(const WifiProvisioningInfo &info) { wifi_info_ = info; }

void NativeFrontend::set_provisioning_command_callback(ProvisioningCommandCallback callback) {
  provisioning_command_callback_ = std::move(callback);
}

void NativeFrontend::set_device_config_change_callback(DeviceConfigChangeCallback callback) {
  device_config_change_callback_ = std::move(callback);
}

bool NativeFrontend::setup() {
  if (bindings_ == nullptr) {
    ESP_LOGE(TAG, "BLE bindings are not configured");
    return false;
  }

  bindings_->set_connection_state_callback([this](bool connected) { this->handle_connection_state_(connected); });
  bindings_->set_control_write_callback([this](const std::string &command) { this->handle_control_command_(command); });
  bindings_->set_telemetry_subscription_callback(
      [this](bool subscribed) { this->handle_live_telemetry_subscription_(subscribed); });
  const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(device_config_),
                                                       device_info_.chip.empty() ? nullptr
                                                                                 : device_info_.chip.c_str());
  bindings_->set_device_name(device_name.c_str());
  runtime_.set_live_telemetry_enabled(false);
  if (!bindings_->setup()) {
    ESP_LOGE(TAG, "BLE bindings setup failed");
    return false;
  }

  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    bindings_->shutdown();
    return false;
  }

  if (ota_service_ != nullptr) {
    ota_service_->set_prepare_for_update_callback([this]() { this->runtime_.quiesce_for_ota(); });
    ota_service_->set_status_callback([this](const EspectreOtaStatus &status) { this->publish_mqtt_ota_status_(status); });
  }

  setup_mqtt_();
  ESP_LOGI(TAG, "Native frontend initialized");
  return true;
}

void NativeFrontend::loop() {
  const int64_t loop_started_us = esp_timer_get_time();
  runtime_.loop();
  if (bindings_ != nullptr) {
    bindings_->loop();
  }
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->loop();
  }
  if (ota_service_ != nullptr) {
    ota_service_->loop();
  }
  last_loop_time_ms_ = static_cast<float>(esp_timer_get_time() - loop_started_us) / 1000.0f;
}

void NativeFrontend::shutdown() {
  publish_mqtt_status_(false);
  runtime_.shutdown();
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->shutdown();
  }
  if (ota_service_ != nullptr) {
    ota_service_->shutdown();
  }
  if (bindings_ != nullptr) {
    bindings_->shutdown();
  }
  client_connected_ = false;
}

NativeFrontend::~NativeFrontend() { shutdown(); }

void NativeFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
}

void NativeFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  runtime_.record_snapshot(snapshot);
  const uint32_t now = now_ms_();
  publish_mqtt_telemetry_(snapshot, now);
  status_logger_.log_status(TAG, snapshot, packets_received);
}

void NativeFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  runtime_.config().segmentation_threshold = snapshot.threshold;
  send_system_info_();
  publish_mqtt_telemetry_(snapshot, now_ms_());
}

void NativeFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  send_system_info_();
}

void NativeFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  finalize_frontend_calibration(runtime_, snapshot, [this]() { status_logger_.reset(); }, success, TAG);
  send_system_info_();
}

void NativeFrontend::on_live_telemetry(float movement, float threshold) {
  if (!client_connected_ || bindings_ == nullptr) {
    return;
  }

  uint8_t payload[sizeof(float) * 2 + 1] = {0};
  std::memcpy(payload, &movement, sizeof(float));
  std::memcpy(payload + sizeof(float), &threshold, sizeof(float));
  payload[sizeof(float) * 2] =
      runtime_.snapshot().motion_state == MotionState::MOTION ? kTelemetryMotionStateMotion : kTelemetryMotionStateIdle;
  bindings_->publish_telemetry(payload, sizeof(payload));
}

void NativeFrontend::on_runtime_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW(TAG, "Runtime fault: %s", message);
  }
  if (bindings_ != nullptr) {
    bindings_->report_fault(message);
  }
}

bool NativeFrontend::handle_control_command_(const std::string &command) {
  if (command == "REQ_SYSINFO") {
    send_system_info_();
    return true;
  }
  DeviceConfigBleCommandResult device_config_result = handle_ble_device_config_command(
      command,
      device_config_,
      [this](EspectreDeviceConfig *cleared_config, std::string *message) {
        if (!device_config_change_callback_) {
          ESP_LOGW(TAG, "Device config change callback is not configured");
          return false;
        }
        const bool accepted = device_config_change_callback_(EspectreDeviceConfig{}, true, message);
        if (accepted && cleared_config != nullptr) {
          cleared_config->device_id = espectre_effective_device_id_u64(device_config_);
        }
        return accepted;
      },
      [this](EspectreDeviceConfig *updated_config, std::string *message) {
        if (updated_config == nullptr) {
          return false;
        }
        if (!device_config_change_callback_) {
          return true;
        }
        return device_config_change_callback_(*updated_config, false, message);
      });
  if (device_config_result.handled) {
    if (!device_config_result.message.empty()) {
      ESP_LOGI(TAG,
               "Device config command %s: %s",
               device_config_result.accepted ? "accepted" : "rejected",
               device_config_result.message.c_str());
    }
    if (!device_config_result.accepted) {
      return false;
    }
    publish_mqtt_status_(false);
    if (device_config_result.config_changed) {
      set_device_config(device_config_result.config);
    }
    setup_mqtt_();
    send_system_info_();
    return true;
  }
  if (command.rfind("SET_WIFI_CONFIG:", 0) == 0 || command == "CLEAR_WIFI") {
    if (!provisioning_command_callback_) {
      ESP_LOGW(TAG, "Provisioning command callback is not configured");
      return false;
    }
    std::string message;
    const bool accepted = provisioning_command_callback_(command, &message);
    if (!message.empty()) {
      ESP_LOGI(TAG, "Provisioning command %s: %s", accepted ? "accepted" : "rejected", message.c_str());
    }
    return accepted;
  }
  if (command.rfind("SET_THRESHOLD:", 0) == 0) {
    const char *value_str = command.c_str() + 14;
    char *end_ptr = nullptr;
    errno = 0;
    const float threshold = strtof(value_str, &end_ptr);
    const bool parse_ok = (end_ptr != value_str) && (end_ptr != nullptr) && (*end_ptr == '\0') &&
                          (errno != ERANGE) && std::isfinite(threshold);
    if (!parse_ok || !validate_runtime_threshold(threshold)) {
      ESP_LOGW(TAG, "Invalid BLE threshold command: %s", command.c_str());
      return false;
    }
    return handle_threshold_write_(threshold);
  }

  ESP_LOGW(TAG, "Unknown BLE control command: %s", command.c_str());
  return false;
}

void NativeFrontend::handle_mqtt_command_(const std::string &payload) {
  const FrontendMqttCommandResult result = handle_frontend_mqtt_command(
      payload,
      ota_service_,
      device_info_.firmware_version.c_str(),
      FrontendMqttCommandCapabilities{
          true,
          true,
          runtime_.capabilities().supports_runtime_threshold_updates,
          ota_service_ != nullptr,
      },
      [this]() { this->publish_mqtt_info_(); },
      [this]() { this->publish_mqtt_stats_(); },
      [this](float threshold, std::string *message) {
        const bool accepted = this->handle_threshold_write_(threshold);
        if (message != nullptr && message->empty()) {
          *message = accepted ? "threshold updated" : "threshold rejected";
        }
        return accepted;
      },
      [this](const EspectreOtaStatus &status) { this->publish_mqtt_ota_status_(status); });
  if (result.handled) {
    publish_mqtt_command_result_(result.command, result.accepted, result.message.c_str());
  }
}

bool NativeFrontend::handle_threshold_write_(float threshold) {
  if (!runtime_.capabilities().supports_runtime_threshold_updates) {
    ESP_LOGW(TAG, "Runtime threshold updates are not supported");
    return false;
  }

  if (!runtime_.set_threshold_runtime(threshold)) {
    return false;
  }
  send_system_info_();
  return true;
}

void NativeFrontend::handle_connection_state_(bool connected) {
  client_connected_ = connected;
  if (connected) {
    telemetry_subscribed_ = false;
    runtime_.set_live_telemetry_enabled(false);
    send_system_info_();
  } else {
    telemetry_subscribed_ = false;
    runtime_.set_live_telemetry_enabled(false);
  }
}

void NativeFrontend::handle_live_telemetry_subscription_(bool subscribed) {
  telemetry_subscribed_ = subscribed;
  runtime_.set_live_telemetry_enabled(client_connected_ && telemetry_subscribed_);
}

void NativeFrontend::setup_mqtt_() {
  (void) setup_frontend_mqtt_transport(mqtt_transport_,
                                       device_config_,
                                       [this](const std::string &payload) { this->handle_mqtt_command_(payload); },
                                       [this]() {
                                         this->publish_mqtt_info_();
                                         this->publish_mqtt_status_(true);
                                       },
                                       TAG);
}

void NativeFrontend::publish_mqtt_info_() {
  const EspectreDeviceInfo info =
      normalize_protocol_device_info(device_info_, &runtime_.snapshot(), ota_service_ != nullptr, "native", CONFIG_IDF_TARGET);
  (void) publish_frontend_mqtt_message(
      mqtt_transport_, device_config_, "info", espectre_info_payload(device_config_, info), false);
}

void NativeFrontend::publish_mqtt_status_(bool online) {
  (void) publish_frontend_mqtt_status(mqtt_transport_, device_config_, online, now_ms_());
}

void NativeFrontend::publish_mqtt_telemetry_(const RuntimeSnapshot &snapshot, uint32_t now) {
  const char *frontend = device_info_.frontend.empty() ? "native" : device_info_.frontend.c_str();
  (void) publish_frontend_mqtt_message(mqtt_transport_,
                                       device_config_,
                                       "telemetry",
                                       espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U, frontend),
                                       false);
}

void NativeFrontend::publish_mqtt_stats_() {
  const uint32_t now = now_ms_();
  (void) publish_frontend_mqtt_message(
      mqtt_transport_,
      device_config_,
      "stats",
      espectre_stats_payload(
          device_config_, runtime_.snapshot(), now, now / 1000U, current_free_memory_kb(), last_loop_time_ms_),
      false);
}

void NativeFrontend::publish_mqtt_ota_status_(const EspectreOtaStatus &status) {
  (void) publish_frontend_mqtt_ota_status(mqtt_transport_, device_config_, status, now_ms_());
}

void NativeFrontend::publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message) {
  (void) publish_frontend_mqtt_command_result(mqtt_transport_, device_config_, command, accepted, message);
}

void NativeFrontend::send_system_info_() {
  if (!client_connected_ || bindings_ == nullptr) {
    return;
  }

  std::vector<std::string> lines;
  EspectreDeviceInfo sysinfo_device_info = device_info_;
  if (sysinfo_device_info.chip.empty()) {
    sysinfo_device_info.chip = CONFIG_IDF_TARGET;
  }
  lines = build_frontend_sysinfo_lines(FrontendSysinfoBase{
      "native",
      SysinfoCapabilities{true,
                          true,
                          true,
                          runtime_.capabilities().supports_runtime_threshold_updates,
                          runtime_.capabilities().supports_ble_telemetry,
                          runtime_.capabilities().supports_extended_diagnostics,
                          ota_service_ != nullptr},
      device_config_,
      sysinfo_device_info,
      true,
      false,
      mqtt_transport_ != nullptr && mqtt_transport_->connected(),
      SysinfoWifiState{
          wifi_info_.ssid, wifi_info_.bssid, wifi_info_.channel, wifi_info_.password_set, device_info_.network.channel > 0U,
      },
  });
  char line[96];
  visit_runtime_diagnostics(runtime_.config(), runtime_.snapshot(), [&line, &lines](const char *key, const char *value) {
    std::snprintf(line, sizeof(line), "%s=%s", key, value);
    lines.emplace_back(line);
  });
  append_sysinfo_end_line(&lines);
  bindings_->replace_sysinfo_lines(std::move(lines));
}

uint32_t NativeFrontend::now_ms_() const { return monotonic_now_ms(); }

}  // namespace espectre
}  // namespace esphome
