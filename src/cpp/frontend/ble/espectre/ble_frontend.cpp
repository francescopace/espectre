/*
 * ESPectre - BLE Frontend Adapter
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "ble_frontend.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "espectre_log.h"
#include "esp_timer.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "sdkconfig.h"

#if __has_include("esp_wifi.h")
#include "esp_wifi.h"
#define ESPECTRE_HAVE_ESP_WIFI 1
#endif

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "espectre.ble";
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

BleFrontend::BleFrontend(IBleBindings *bindings) : bindings_(bindings) {}

BleFrontend::BleFrontend(IBleBindings *bindings, IMqttTransport *mqtt_transport)
    : bindings_(bindings), mqtt_transport_(mqtt_transport) {}

void BleFrontend::set_runtime_config(const RuntimeConfig &config) { runtime_.set_config(config); }

void BleFrontend::set_device_config(const EspectreDeviceConfig &config) {
  device_config_ = config;
  if (bindings_ != nullptr) {
    const std::string ble_name = espectre_ble_device_name(device_config_);
    bindings_->set_device_name(ble_name.c_str());
  }
}

void BleFrontend::set_device_info(const EspectreDeviceInfo &info) { device_info_ = info; }

void BleFrontend::set_wifi_provisioning_info(const WifiProvisioningInfo &info) { wifi_info_ = info; }

void BleFrontend::set_provisioning_command_callback(ProvisioningCommandCallback callback) {
  provisioning_command_callback_ = std::move(callback);
}

void BleFrontend::set_device_config_change_callback(DeviceConfigChangeCallback callback) {
  device_config_change_callback_ = std::move(callback);
}

bool BleFrontend::setup() {
  if (bindings_ == nullptr) {
    ESP_LOGE(TAG, "BLE bindings are not configured");
    return false;
  }

  bindings_->set_connection_state_callback([this](bool connected) { this->handle_connection_state_(connected); });
  bindings_->set_control_write_callback([this](const std::string &command) { this->handle_control_command_(command); });
  bindings_->set_telemetry_subscription_callback(
      [this](bool subscribed) { this->handle_live_telemetry_subscription_(subscribed); });
  bindings_->set_device_name(espectre_ble_device_name(device_config_).c_str());
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

  setup_mqtt_();
  ESP_LOGI(TAG, "BLE frontend initialized");
  return true;
}

void BleFrontend::loop() {
  const int64_t loop_started_us = esp_timer_get_time();
  runtime_.loop();
  flush_pending_system_info_();
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->loop();
  }
  last_loop_time_ms_ = static_cast<float>(esp_timer_get_time() - loop_started_us) / 1000.0f;
}

void BleFrontend::shutdown() {
  publish_mqtt_status_(false);
  runtime_.shutdown();
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->shutdown();
  }
  if (bindings_ != nullptr) {
    bindings_->shutdown();
  }
  client_connected_ = false;
}

BleFrontend::~BleFrontend() { shutdown(); }

void BleFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
}

void BleFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  runtime_.record_snapshot(snapshot);
  const uint32_t now = now_ms_();
  publish_mqtt_telemetry_(snapshot, now);
  log_runtime_rates_(now, packets_received);
}

void BleFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  runtime_.config().segmentation_threshold = snapshot.threshold;
  send_system_info_();
  publish_mqtt_telemetry_(snapshot, now_ms_());
}

void BleFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  send_system_info_();
}

void BleFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  runtime_.record_snapshot(snapshot);
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
  send_system_info_();
}

void BleFrontend::on_live_telemetry(float movement, float threshold) {
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

void BleFrontend::on_runtime_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW(TAG, "Runtime fault: %s", message);
  }
  if (bindings_ != nullptr) {
    bindings_->report_fault(message);
  }
}

bool BleFrontend::handle_control_command_(const std::string &command) {
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
    cleared.device_id = espectre_effective_device_id(device_config_);
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

void BleFrontend::handle_mqtt_command_(const std::string &payload) {
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

  publish_mqtt_command_result_(command, false, "unsupported command");
}

bool BleFrontend::handle_threshold_write_(float threshold) {
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

void BleFrontend::handle_connection_state_(bool connected) {
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

void BleFrontend::handle_live_telemetry_subscription_(bool subscribed) {
  telemetry_subscribed_ = subscribed;
  runtime_.set_live_telemetry_enabled(client_connected_ && telemetry_subscribed_);
}

void BleFrontend::setup_mqtt_() {
  if (mqtt_transport_ == nullptr) {
    return;
  }
  if (!device_config_.mqtt_enabled || device_config_.mqtt_host.empty()) {
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

void BleFrontend::publish_mqtt_info_() {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  EspectreDeviceInfo info = device_info_;
  info.frontend = info.frontend.empty() ? "ble" : info.frontend;
  info.firmware_version = info.firmware_version.empty() ? "unknown" : info.firmware_version;
  info.chip = info.chip.empty() ? CONFIG_IDF_TARGET : info.chip;
  if (info.detector.empty() && runtime_.snapshot().detector_name != nullptr) {
    info.detector = runtime_.snapshot().detector_name;
  }
  mqtt_transport_->publish(espectre_topic(device_config_, "info"),
                           espectre_info_payload(device_config_, info),
                           true);
}

void BleFrontend::publish_mqtt_status_(bool online) {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  mqtt_transport_->publish(espectre_topic(device_config_, "status"),
                           espectre_status_payload(device_config_, online, now_ms_()),
                           true);
}

void BleFrontend::publish_mqtt_telemetry_(const RuntimeSnapshot &snapshot, uint32_t now) {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  if (mqtt_transport_->publish(espectre_topic(device_config_, "telemetry"),
                               espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U),
                               false)) {
    mqtt_publish_count_ += 1;
  }
}

void BleFrontend::publish_mqtt_stats_() {
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

void BleFrontend::publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message) {
  if (mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  mqtt_transport_->publish(espectre_topic(device_config_, accepted ? "commands/accepted" : "commands/rejected"),
                           espectre_command_result_payload(device_config_, command, accepted, message),
                           false);
}

void BleFrontend::queue_system_info_line_(const char *line) {
  pending_sysinfo_lines_.emplace_back(line != nullptr ? line : "");
}

void BleFrontend::flush_pending_system_info_(bool force) {
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

void BleFrontend::log_runtime_rates_(uint32_t now_ms, uint32_t packets_received) {
  const RuntimeSnapshot &snapshot = runtime_.snapshot();
  const uint32_t elapsed_ms = (last_rate_log_ms_ > 0 && now_ms > last_rate_log_ms_) ? (now_ms - last_rate_log_ms_) : 0U;
  const uint32_t mqtt_delta = (last_rate_log_ms_ > 0) ? (mqtt_publish_count_ - last_rate_mqtt_publish_count_) : 0U;
  const uint32_t csi_rate_pps =
      elapsed_ms > 0 ? static_cast<uint32_t>((static_cast<uint64_t>(packets_received) * 1000U) / elapsed_ms) : 0U;
  const float mqtt_rate_hz = elapsed_ms > 0 ? (static_cast<float>(mqtt_delta) * 1000.0F / static_cast<float>(elapsed_ms)) : 0.0F;
  const float motion_metric = snapshot.movement_metric;
  const float threshold = snapshot.threshold;
  const bool is_motion = (snapshot.motion_state == MotionState::MOTION);
  const float progress = (threshold > 0.0F) ? (motion_metric / threshold) : 0.0F;
  const int percent = static_cast<int>(progress * 100.0F);

  int8_t rssi = -127;
  uint8_t channel = 0;
#ifdef ESPECTRE_HAVE_ESP_WIFI
  wifi_ap_record_t ap_info{};
  if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
    rssi = ap_info.rssi;
    channel = ap_info.primary;
  }
#endif

  log_progress_bar(TAG, progress, 20, 15,
                   "%d%% | mvmt:%.4f thr:%.4f | %s | %u pkt/s | ch:%u rssi:%d | mqtt:%.2f msg/s",
                   percent, motion_metric, threshold,
                   is_motion ? "MOTION" : "IDLE",
                   static_cast<unsigned>(csi_rate_pps),
                   static_cast<unsigned>(channel),
                   static_cast<int>(rssi),
                   static_cast<double>(mqtt_rate_hz));

  last_rate_log_ms_ = now_ms;
  last_rate_mqtt_publish_count_ = mqtt_publish_count_;
}

void BleFrontend::send_system_info_() {
  if (!client_connected_ || bindings_ == nullptr) {
    return;
  }

  char line[96];
  const std::string ble_device_name = espectre_ble_device_name(device_config_);
  pending_sysinfo_lines_.clear();
  next_sysinfo_line_index_ = 0;
  last_sysinfo_line_ms_ = 0;

  queue_system_info_line_("proto_version=1");
  std::snprintf(line, sizeof(line), "espectre_protocol_version=%s", ESPECTRE_PROTOCOL_VERSION);
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "device_id=%s", espectre_effective_device_id(device_config_).c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "device_name=%s", espectre_effective_device_name(device_config_).c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "ble_device_name=%s", ble_device_name.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "mqtt_enabled=%s", device_config_.mqtt_enabled ? "true" : "false");
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "mqtt_host=%s", device_config_.mqtt_host.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "mqtt_port=%u", static_cast<unsigned>(device_config_.mqtt_port));
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "mqtt_username=%s", device_config_.mqtt_username.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "topic_prefix=%s", device_config_.topic_prefix.c_str());
  queue_system_info_line_(line);
  std::snprintf(line, sizeof(line), "wifi_saved=%s", wifi_info_.has_saved_config ? "true" : "false");
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

uint32_t BleFrontend::now_ms_() const { return static_cast<uint32_t>(esp_timer_get_time() / 1000ULL); }

}  // namespace espectre
}  // namespace esphome
