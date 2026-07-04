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

#include "ble_protocol.h"
#include "espectre_log.h"
#include "esp_timer.h"
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

void NativeFrontend::set_device_info(const EspectreDeviceInfo &info) { device_info_ = info; }

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
  flush_pending_system_info_();
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
  if (command == "CLEAR_MQTT_CONFIG") {
    if (!device_config_change_callback_) {
      ESP_LOGW(TAG, "Device config change callback is not configured");
      return false;
    }
    EspectreDeviceConfig updated = device_config_;
    clear_espectre_mqtt_config(&updated);
    std::string message;
    if (!device_config_change_callback_(updated, false, &message)) {
      ESP_LOGW(TAG, "MQTT config clear failed: %s", message.c_str());
      return false;
    }
    if (!message.empty()) {
      ESP_LOGI(TAG, "MQTT config cleared: %s", message.c_str());
    }
    set_device_config(updated);
    publish_mqtt_status_(false);
    setup_mqtt_();
    send_system_info_();
    return true;
  }
  if (command == "CLEAR_DEVICE_CONFIG") {
    if (!device_config_change_callback_) {
      ESP_LOGW(TAG, "Device config change callback is not configured");
      return false;
    }
    std::string message;
    const bool accepted = device_config_change_callback_(EspectreDeviceConfig{}, true, &message);
    if (!message.empty()) {
      ESP_LOGI(TAG, "Device config clear %s: %s", accepted ? "accepted" : "rejected", message.c_str());
    }
    if (!accepted) {
      return false;
    }
    publish_mqtt_status_(false);
    EspectreDeviceConfig cleared{};
    cleared.device_id = espectre_effective_device_id_u64(device_config_);
    set_device_config(cleared);
    setup_mqtt_();
    send_system_info_();
    return true;
  }
  if (command.rfind("SET_DEVICE_CONFIG:", 0) == 0) {
    EspectreDeviceConfig updated = device_config_;
    std::string error;
    if (!parse_espectre_config_command(command, &updated, &error)) {
      ESP_LOGW(TAG, "Invalid device config command: %s", error.c_str());
      return false;
    }
    if (device_config_change_callback_) {
      std::string message;
      if (!device_config_change_callback_(updated, false, &message)) {
        ESP_LOGW(TAG, "Device config persistence failed: %s", message.c_str());
        return false;
      }
      if (!message.empty()) {
        ESP_LOGI(TAG, "Device config persisted: %s", message.c_str());
      }
    }
    set_device_config(updated);
    setup_mqtt_();
    send_system_info_();
    return true;
  }
  if (command.rfind("SET_WIFI_", 0) == 0 || command == "APPLY_WIFI" || command == "CLEAR_WIFI") {
    if (!provisioning_command_callback_) {
      ESP_LOGW(TAG, "Provisioning command callback is not configured");
      return false;
    }
    std::string message;
    const bool accepted = provisioning_command_callback_(command, &message);
    if (!message.empty()) {
      ESP_LOGI(TAG, "Provisioning command %s: %s", accepted ? "accepted" : "rejected", message.c_str());
    }
    if (accepted) {
      send_system_info_();
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
  EspectreCommand command;
  std::string error;
  if (!parse_espectre_command(payload, &command, &error)) {
    command.command = "unknown";
    publish_mqtt_command_result_(command, false, error.c_str());
    return;
  }

  if (command.command == "set_threshold") {
    if (!command.has_threshold || !validate_runtime_threshold(command.threshold)) {
      publish_mqtt_command_result_(command, false, "invalid threshold");
      return;
    }
    const bool accepted = handle_threshold_write_(command.threshold);
    publish_mqtt_command_result_(command, accepted, accepted ? "threshold updated" : "threshold rejected");
    return;
  }

  if (command.command == "info") {
    publish_mqtt_info_();
    publish_mqtt_command_result_(command, true, "info published");
    return;
  }

  if (command.command == "stats") {
    publish_mqtt_stats_();
    publish_mqtt_command_result_(command, true, "stats published");
    return;
  }

  if (command.command == "ota_check" || command.command == "ota_start" || command.command == "ota_status") {
    const bool accepted = handle_ota_command_(command);
    publish_mqtt_command_result_(command, accepted, accepted ? "ota command accepted" : "ota command rejected");
    return;
  }

  publish_mqtt_command_result_(command, false, "unsupported command");
}

bool NativeFrontend::handle_ota_command_(const EspectreCommand &command) {
  if (ota_service_ == nullptr) {
    return false;
  }

  const std::string current_version =
      device_info_.firmware_version.empty() ? "unknown" : device_info_.firmware_version;
  if (command.command == "ota_status") {
    publish_mqtt_ota_status_(ota_service_->status());
    return true;
  }
  if (command.command == "ota_check") {
    return command.has_manifest_url && ota_service_->start_check(command.manifest_url, current_version);
  }
  if (command.command == "ota_start") {
    return ota_service_->start_update(command.has_manifest_url ? command.manifest_url : "",
                                      command.has_image_url ? command.image_url : "",
                                      command.has_version ? command.version : "",
                                      current_version);
  }
  return false;
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
    pending_sysinfo_lines_.clear();
    next_sysinfo_line_index_ = 0;
    last_sysinfo_line_ms_ = 0;
  }
}

void NativeFrontend::handle_live_telemetry_subscription_(bool subscribed) {
  telemetry_subscribed_ = subscribed;
  runtime_.set_live_telemetry_enabled(client_connected_ && telemetry_subscribed_);
}

void NativeFrontend::setup_mqtt_() {
  if (mqtt_transport_ == nullptr) {
    return;
  }
  if (device_config_.mqtt_host.empty()) {
    mqtt_transport_->shutdown();
    return;
  }

  mqtt_transport_->set_command_callback([this](const std::string &payload) { this->handle_mqtt_command_(payload); });
  mqtt_transport_->set_connection_callback([this](bool connected) {
    if (connected) {
      this->publish_mqtt_info_();
      this->publish_mqtt_status_(true);
    }
  });
  if (!mqtt_transport_->setup(device_config_)) {
    ESP_LOGW(TAG, "MQTT transport setup failed");
    return;
  }
}

void NativeFrontend::publish_mqtt_info_() {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  EspectreDeviceInfo info = device_info_;
  info.frontend = info.frontend.empty() ? "native" : info.frontend;
  info.firmware_version = info.firmware_version.empty() ? "unknown" : info.firmware_version;
  info.chip = info.chip.empty() ? CONFIG_IDF_TARGET : info.chip;
  info.supports_ota = ota_service_ != nullptr;
  if (info.detector.empty() && runtime_.snapshot().detector_name != nullptr) {
    info.detector = runtime_.snapshot().detector_name;
  }
  mqtt_transport_->publish(espectre_topic(device_config_, "info"),
                           espectre_info_payload(device_config_, info),
                           true);
}

void NativeFrontend::publish_mqtt_status_(bool online) {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  mqtt_transport_->publish(espectre_topic(device_config_, "status"),
                           espectre_status_payload(device_config_, online, now_ms_()),
                           true);
}

void NativeFrontend::publish_mqtt_telemetry_(const RuntimeSnapshot &snapshot, uint32_t now) {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  const char *frontend = device_info_.frontend.empty() ? "native" : device_info_.frontend.c_str();
  mqtt_transport_->publish(espectre_topic(device_config_, "telemetry"),
                           espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U, frontend),
                           false);
}

void NativeFrontend::publish_mqtt_stats_() {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  const uint32_t now = now_ms_();
  mqtt_transport_->publish(espectre_topic(device_config_, "stats"),
                           espectre_stats_payload(device_config_,
                                                  runtime_.snapshot(),
                                                  now,
                                                  now / 1000U,
                                                  current_free_memory_kb(),
                                                  last_loop_time_ms_),
                           false);
}

void NativeFrontend::publish_mqtt_ota_status_(const EspectreOtaStatus &status) {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  mqtt_transport_->publish(espectre_topic(device_config_, "ota/state"),
                           espectre_ota_status_payload(device_config_, status, now_ms_()),
                           true);
}

void NativeFrontend::publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message) {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  mqtt_transport_->publish(espectre_topic(device_config_, accepted ? "commands/accepted" : "commands/rejected"),
                           espectre_command_result_payload(device_config_, command, accepted, message),
                           false);
}

void NativeFrontend::queue_system_info_line_(const char *line) {
  pending_sysinfo_lines_.emplace_back(line != nullptr ? line : "");
}

void NativeFrontend::flush_pending_system_info_(bool force) {
  if (!client_connected_ || bindings_ == nullptr || next_sysinfo_line_index_ >= pending_sysinfo_lines_.size()) {
    return;
  }

  const uint32_t now = now_ms_();
  if (!force && last_sysinfo_line_ms_ != 0 && (now - last_sysinfo_line_ms_) < sysinfo_line_interval_ms_) {
    return;
  }

  bindings_->publish_sysinfo_line(pending_sysinfo_lines_[next_sysinfo_line_index_].c_str());
  last_sysinfo_line_ms_ = now;
  next_sysinfo_line_index_ += 1;

  if (next_sysinfo_line_index_ >= pending_sysinfo_lines_.size()) {
    pending_sysinfo_lines_.clear();
    next_sysinfo_line_index_ = 0;
  }
}

void NativeFrontend::send_system_info_() {
  if (!client_connected_ || bindings_ == nullptr) {
    return;
  }

  char line[96];
  const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(device_config_),
                                                       device_info_.chip.empty() ? nullptr
                                                                                 : device_info_.chip.c_str());
  pending_sysinfo_lines_.clear();
  next_sysinfo_line_index_ = 0;
  last_sysinfo_line_ms_ = 0;

  queue_system_info_line_("proto_version=1");
  queue_system_info_line_("frontend=native");
  std::snprintf(line, sizeof(line), "espectre_protocol_version=%s", ESPECTRE_PROTOCOL_VERSION);
  queue_system_info_line_(line);
  queue_system_info_line_("supports_wifi_provisioning=true");
  queue_system_info_line_("supports_mqtt_config=true");
  queue_system_info_line_("supports_device_config=true");
  std::snprintf(line,
                sizeof(line),
                "supports_runtime_threshold=%s",
                runtime_.capabilities().supports_runtime_threshold_updates ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line,
                sizeof(line),
                "supports_live_telemetry=%s",
                runtime_.capabilities().supports_ble_telemetry ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line,
                sizeof(line),
                "supports_extended_diagnostics=%s",
                runtime_.capabilities().supports_extended_diagnostics ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "supports_ota=%s", ota_service_ != nullptr ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "device_id=%s", espectre_effective_device_id(device_config_).c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "device_label=%s", espectre_effective_device_label(device_config_).c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "device_name=%s", device_name.c_str());
  queue_system_info_line_(line);
  std::snprintf(line,
                sizeof(line),
                "mqtt_connected=%s",
                mqtt_transport_ != nullptr && mqtt_transport_->connected() ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "mqtt_host=%s", device_config_.mqtt_host.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "mqtt_port=%u", static_cast<unsigned>(device_config_.mqtt_port));
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "mqtt_username=%s", device_config_.mqtt_username.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "topic_prefix=%s", device_config_.topic_prefix.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "wifi_connected=%s", device_info_.network.channel > 0U ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "wifi_ssid=%s", wifi_info_.ssid.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "wifi_bssid=%s", wifi_info_.bssid.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "wifi_channel=%u", static_cast<unsigned>(wifi_info_.channel));
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "wifi_password_set=%s", wifi_info_.password_set ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "chip=%s", CONFIG_IDF_TARGET);
  queue_system_info_line_(line);
  visit_runtime_diagnostics(runtime_.config(), runtime_.snapshot(), [this, &line](const char *key, const char *value) {
    std::snprintf(line, sizeof(line), "%s=%s", key, value);
    queue_system_info_line_(line);
  });
  queue_system_info_line_("END");
  flush_pending_system_info_(true);
}

uint32_t NativeFrontend::now_ms_() const { return static_cast<uint32_t>(esp_timer_get_time() / 1000ULL); }

}  // namespace espectre
}  // namespace esphome
