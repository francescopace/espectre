/*
 * ESPectre - Native Frontend Adapter
 *
 * Bridges runtime events and control flows to BLE, MQTT, and OTA services.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "native_frontend.h"

#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "ble_protocol.h"
#include "espectre_log.h"
#include "esp_timer.h"
#include "frontend_control_helpers.h"
#include "frontend_mqtt_helpers.h"
#include "frontend_sysinfo_helpers.h"
#include "protocol_json.h"
#include "runtime_time.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "runtime_listener_utils.h"
#include "sdkconfig.h"
#include "wifi_band_helpers.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace espectre {

namespace {

static const char *const TAG = "espectre.native";
constexpr const char *kHaOnlinePayload = "online";

const char *motion_state_payload(MotionState state) {
  return state == MotionState::MOTION ? "ON" : "OFF";
}

std::string float_payload(float value) {
  char buffer[24];
  std::snprintf(buffer, sizeof(buffer), "%.4f", static_cast<double>(value));
  return buffer;
}

std::string diagnostic_state_payload(const std::string &key, const RuntimeDiagnosticsSample &sample) {
  char buffer[24];
  if (key == "traffic_tx_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.traffic_tx_pps));
  } else if (key == "csi_callback_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_callback_pps));
  } else if (key == "csi_accepted_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_accepted_pps));
  } else if (key == "csi_admitted_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_admitted_pps));
  } else if (key == "csi_filtered_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_filtered_pps));
  } else if (key == "csi_missing_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_missing_slots_pps));
  } else if (key == "csi_excess_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_excess_pps));
  } else if (key == "csi_stale_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_stale_pps));
  } else if (key == "csi_out_of_order_rate") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_out_of_order_pps));
  } else if (key == "csi_occupancy") {
    std::snprintf(buffer, sizeof(buffer), "%.1f", static_cast<double>(sample.csi_occupancy_ratio * 100.0f));
  } else if (key == "wifi_channel") {
    std::snprintf(buffer, sizeof(buffer), "%u", static_cast<unsigned>(sample.wifi_channel));
  } else if (key == "wifi_rssi") {
    if (sample.wifi_rssi_dbm == INT8_MIN) {
      return {};
    }
    std::snprintf(buffer, sizeof(buffer), "%d", static_cast<int>(sample.wifi_rssi_dbm));
  } else {
    return {};
  }
  return buffer;
}

std::string normalize_text_token(const std::string &value) {
  std::string normalized;
  normalized.reserve(value.size());
  for (const char ch : value) {
    if (!std::isspace(static_cast<unsigned char>(ch))) {
      normalized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
  }
  return normalized;
}

const char *ota_state_name(EspectreOtaState state) {
  switch (state) {
    case EspectreOtaState::IDLE:
      return "idle";
    case EspectreOtaState::CHECKING:
      return "checking";
    case EspectreOtaState::UPDATE_AVAILABLE:
      return "update_available";
    case EspectreOtaState::UP_TO_DATE:
      return "up_to_date";
    case EspectreOtaState::DOWNLOADING:
      return "downloading";
    case EspectreOtaState::APPLYING:
      return "applying";
    case EspectreOtaState::REBOOT_SCHEDULED:
      return "reboot_scheduled";
    case EspectreOtaState::ERROR:
      return "error";
    default:
      return "unknown";
  }
}

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

void NativeFrontend::set_runtime_config(const RuntimeConfig &config) {
  RuntimeConfig native_config = config;
  native_config.runtime_detector_selection_enabled = true;
  runtime_.set_config(native_config);
}

void NativeFrontend::set_device_config(const EspectreDeviceConfig &config) {
  device_config_ = config;
  if (bindings_ != nullptr) {
    const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(device_config_),
                                                         device_info_.chip.empty() ? nullptr
                                                                                   : device_info_.chip.c_str());
    bindings_->set_device_name(device_name.c_str());
  }
  refresh_ble_policy_();
}

void NativeFrontend::set_device_info(const EspectreDeviceInfo &info) {
  device_info_ = info;
  if (client_connected_) {
    system_info_refresh_.request();
  }
}

void NativeFrontend::set_wifi_provisioning_info(const WifiProvisioningInfo &info) {
  wifi_info_ = info;
  if (client_connected_) {
    system_info_refresh_.request();
  }
  refresh_ble_policy_();
}

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
  update_live_telemetry_enabled_();
  if (ble_should_run_()) {
    if (!start_ble_()) {
      ESP_LOGE(TAG, "BLE bindings setup failed");
      return false;
    }
  } else {
    ESP_LOGI(TAG, "BLE idle: Wi-Fi and MQTT are configured; MQTT set_ble on to enter setup");
  }

  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    stop_ble_();
    return false;
  }
  const uint32_t diagnostics_now_ms = now_ms_();
  const RuntimeDiagnosticsSnapshot diagnostics = runtime_.diagnostics();
  diagnostics_sampler_.reset(diagnostics, diagnostics_now_ms);
  latest_diagnostics_ = diagnostics_sampler_.sample(diagnostics, diagnostics_now_ms);

  if (ota_service_ != nullptr) {
    ota_service_->set_prepare_for_update_callback([this]() { this->runtime_.quiesce_for_ota(); });
    ota_service_->set_status_callback([this](const EspectreOtaStatus &status) {
      this->publish_mqtt_ota_status_(status);
      this->system_info_refresh_.request();
    });
  }

  setup_mqtt_();
  ESP_LOGI(TAG, "Native frontend initialized");
  return true;
}

void NativeFrontend::loop() {
  const int64_t loop_started_us = esp_timer_get_time();
  apply_pending_ble_intent_();
  runtime_.loop();
  if (ble_active_ && bindings_ != nullptr) {
    bindings_->loop();
  }
  system_info_refresh_.flush_if([this]() { return this->client_connected_ && this->bindings_ != nullptr; },
                                [this]() { this->send_system_info_(); });
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->loop();
  }
  if (ota_service_ != nullptr) {
    ota_service_->loop();
  }
  last_loop_time_ms_ = static_cast<float>(esp_timer_get_time() - loop_started_us) / 1000.0f;
}

void NativeFrontend::shutdown() {
  mqtt_connected_ = false;
  mqtt_ha_online_ = false;
  update_live_telemetry_enabled_();
  publish_mqtt_status_(false);
  runtime_.shutdown();
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->shutdown();
  }
  if (ota_service_ != nullptr) {
    ota_service_->shutdown();
  }
  pending_ble_intent_ = BleIntent::Unchanged;
  ble_forced_ = false;
  stop_ble_();
  client_connected_ = false;
}

NativeFrontend::~NativeFrontend() { shutdown(); }

void NativeFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  // State-change callbacks contain filtered motion edges, not every detector
  // evaluation. Canonical telemetry already follows evaluation.
  if (!snapshot.ready_to_publish) {
    return;
  }
  publish_ha_motion_(snapshot.motion_state);
}

void NativeFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) packets_received;
  runtime_.record_snapshot(snapshot);
  sample_diagnostics_(now_ms_());
  if (!snapshot.ready_to_publish) {
    return;
  }
  if (runtime_.capabilities().supports_runtime_detector_selection && snapshot.detector_name != nullptr) {
    publish_ha_detector_(snapshot.detector_name);
  }
}

void NativeFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  apply_threshold_snapshot(runtime_, snapshot);
  system_info_refresh_.request();
  publish_mqtt_telemetry_(snapshot, now_ms_());
  if (snapshot.ready_to_publish) {
    publish_ha_threshold_(snapshot.threshold);
  }
}

void NativeFrontend::on_detector_changed(const RuntimeSnapshot &snapshot) {
  apply_detector_snapshot(runtime_, snapshot);
  system_info_refresh_.request();
  publish_mqtt_info_();
  publish_mqtt_telemetry_(snapshot, now_ms_());
  if (snapshot.ready_to_publish) {
    if (snapshot.detector_name != nullptr) {
      publish_ha_detector_(snapshot.detector_name);
    }
    publish_ha_threshold_(snapshot.threshold);
  }
}

void NativeFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  system_info_refresh_.request();
  publish_ha_calibrate_(true);
}

void NativeFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  finalize_frontend_calibration(runtime_, snapshot, success, TAG);
  system_info_refresh_.request();
  publish_ha_calibrate_(false);
  if (snapshot.ready_to_publish) {
    publish_ha_threshold_(snapshot.threshold);
  }
}

void NativeFrontend::on_live_telemetry(float movement, float threshold) {
  if (!runtime_.snapshot().ready_to_publish) {
    return;
  }
  RuntimeSnapshot snapshot = runtime_.snapshot();
  snapshot.movement_metric = movement;
  snapshot.threshold = threshold;
  runtime_.record_snapshot(snapshot);
  publish_mqtt_telemetry_(snapshot, now_ms_());
  publish_ha_movement_(movement);
}

void NativeFrontend::on_runtime_fault(const char *message) {
  if (bindings_ != nullptr) {
    bindings_->report_fault(message);
  }
}

bool NativeFrontend::handle_control_command_(const std::string &command) {
  if (command == "REQ_SYSINFO") {
    system_info_refresh_.request();
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
    system_info_refresh_.request();
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
  if (command == "STOP_BLE") {
    return handle_ble_mode_write_(false, nullptr);
  }
  if (command.rfind("OTA_", 0) == 0) {
    EspectreCommand ota_command;
    std::string ota_error;
    if (!parse_espectre_ble_ota_command(command, &ota_command, &ota_error)) {
      ESP_LOGW(TAG, "BLE OTA command rejected: %s", ota_error.c_str());
      return false;
    }
    return handle_ble_ota_command_(ota_command);
  }

  ESP_LOGW(TAG, "Unknown BLE control command: %s", command.c_str());
  return false;
}

void NativeFrontend::handle_mqtt_command_(const std::string &payload) {
  EspectreCommand ble_command;
  std::string ble_parse_error;
  if (parse_espectre_command(payload, &ble_command, &ble_parse_error) && ble_command.command == "set_ble") {
    std::string message;
    const bool accepted = handle_ble_mode_write_(ble_command.ble == "on", &message);
    publish_mqtt_command_result_(ble_command, accepted, message.c_str());
    return;
  }

  const FrontendMqttCommandResult result = handle_frontend_mqtt_command(
      payload,
      ota_service_,
      device_info_.firmware_version.c_str(),
      FrontendMqttCommandCapabilities{
          true,
          true,
          runtime_.capabilities().supports_runtime_threshold_updates,
          runtime_.capabilities().supports_runtime_motion_hits_updates,
          runtime_.capabilities().supports_traffic_control,
          runtime_.capabilities().supports_runtime_detector_selection,
          runtime_.capabilities().supports_manual_recalibration,
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
      [this](uint8_t motion_on_hits, uint8_t motion_off_hits, std::string *message) {
        const bool accepted = this->handle_motion_hits_write_(motion_on_hits, motion_off_hits);
        if (message != nullptr && message->empty()) {
          *message = accepted ? "motion hits updated" : "motion hits rejected";
        }
        return accepted;
      },
      [this](CsiTrafficMode mode, std::string *message) {
        const bool accepted = this->handle_csi_traffic_mode_write_(mode);
        if (message != nullptr && message->empty()) {
          *message = accepted ? "csi traffic mode updated" : "csi traffic mode rejected";
        }
        return accepted;
      },
      [this](RuntimeTrafficMode mode, std::string *message) {
        const bool accepted = this->handle_traffic_generator_mode_write_(mode);
        if (message != nullptr && message->empty()) {
          *message = accepted ? "traffic generator mode updated" : "traffic generator mode rejected";
        }
        return accepted;
      },
      [this](DetectionAlgorithm algorithm, std::string *message) {
        const bool accepted = this->handle_detector_write_(algorithm);
        if (message != nullptr) {
          *message = accepted ? "detector updated" : "detector rejected";
        }
        return accepted;
      },
      [this](std::string *message) {
        const bool accepted = this->handle_recalibration_write_();
        if (message != nullptr && message->empty()) {
          *message = accepted ? "recalibration started" : "recalibration rejected";
        }
        return accepted;
      },
      [this](const EspectreOtaStatus &status) { this->publish_mqtt_ota_status_(status); },
      [this]() { this->publish_mqtt_commands_(); });
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
  system_info_refresh_.request();
  if (runtime_.snapshot().ready_to_publish) {
    publish_ha_threshold_(threshold);
  }
  return true;
}

bool NativeFrontend::handle_motion_hits_write_(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  if (!runtime_.capabilities().supports_runtime_motion_hits_updates) {
    ESP_LOGW(TAG, "Runtime motion hit updates are not supported");
    return false;
  }
  if (!runtime_.set_motion_hits_runtime(motion_on_hits, motion_off_hits)) {
    return false;
  }
  system_info_refresh_.request();
  publish_ha_motion_hits_(motion_on_hits, motion_off_hits);
  publish_mqtt_info_();
  return true;
}

bool NativeFrontend::handle_csi_traffic_mode_write_(CsiTrafficMode mode) {
  if (!runtime_.capabilities().supports_traffic_control) {
    ESP_LOGW(TAG, "Runtime traffic control is not supported");
    return false;
  }
  if (!csi_traffic_mode_is_sensing_control(mode)) {
    ESP_LOGW(TAG, "CSI traffic mode pacing is not selectable");
    return false;
  }
  if (!runtime_.set_csi_traffic_mode_runtime(mode)) {
    return false;
  }
  system_info_refresh_.request();
  publish_ha_traffic_control_(runtime_.config().csi_traffic_mode, runtime_.config().traffic_generator_mode);
  publish_mqtt_info_();
  return true;
}

bool NativeFrontend::handle_traffic_generator_mode_write_(RuntimeTrafficMode mode) {
  if (!runtime_.capabilities().supports_traffic_control) {
    ESP_LOGW(TAG, "Runtime traffic control is not supported");
    return false;
  }
  if (!runtime_.set_traffic_generator_mode_runtime(mode)) {
    return false;
  }
  system_info_refresh_.request();
  publish_ha_traffic_control_(runtime_.config().csi_traffic_mode, runtime_.config().traffic_generator_mode);
  publish_mqtt_info_();
  return true;
}

bool NativeFrontend::handle_detector_write_(DetectionAlgorithm algorithm) {
  if (!runtime_.capabilities().supports_runtime_detector_selection) {
    ESP_LOGW(TAG, "Runtime detector selection is not supported");
    return false;
  }
  if (!runtime_.set_detection_algorithm_runtime(algorithm)) {
    return false;
  }
  system_info_refresh_.request();
  return true;
}

bool NativeFrontend::handle_ble_ota_command_(const EspectreCommand &command) {
  std::string payload = "{\"command\":\"";
  payload += command.command;
  payload += "\"";
  if (command.has_ota_channel) {
    payload += ",\"channel\":\"";
    payload += command.ota_channel;
    payload += "\"";
  }
  payload += "}";
  const FrontendMqttCommandResult result = handle_frontend_mqtt_command(
      payload,
      ota_service_,
      device_info_.firmware_version.c_str(),
      FrontendMqttCommandCapabilities{
          false,
          false,
          false,
          false,
          false,
          false,
          false,
          ota_service_ != nullptr,
      },
      {},
      {},
      {},
      {},
      {},
      {},
      {},
      {},
      [this](const EspectreOtaStatus &status) {
        this->publish_mqtt_ota_status_(status);
        this->system_info_refresh_.request();
      });
  if (!result.accepted) {
    ESP_LOGW(TAG, "BLE OTA command rejected: %s", result.message.c_str());
    return false;
  }
  system_info_refresh_.request();
  return true;
}

bool NativeFrontend::handle_recalibration_write_() {
  if (!runtime_.capabilities().supports_manual_recalibration) {
    ESP_LOGW(TAG, "Manual recalibration is not supported");
    return false;
  }
  const bool accepted = runtime_.trigger_recalibration();
  if (!accepted) {
    return false;
  }
  publish_ha_calibrate_(runtime_.is_calibrating());
  system_info_refresh_.request();
  return true;
}

void NativeFrontend::handle_ha_threshold_command_(const std::string &payload) {
  const std::string token = normalize_text_token(payload);
  char *end_ptr = nullptr;
  errno = 0;
  const float threshold = strtof(token.c_str(), &end_ptr);
  const bool parse_ok = (end_ptr != token.c_str()) && (end_ptr != nullptr) && (*end_ptr == '\0') &&
                        (errno != ERANGE) && std::isfinite(threshold);
  if (!parse_ok) {
    ESP_LOGW(TAG, "Invalid HA threshold command: %s", payload.c_str());
    return;
  }
  (void) handle_threshold_write_(threshold);
}

void NativeFrontend::handle_ha_motion_hits_command_(bool motion_on, const std::string &payload) {
  const std::string token = normalize_text_token(payload);
  char *end_ptr = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(token.c_str(), &end_ptr, 10);
  if (end_ptr == token.c_str() || end_ptr == nullptr || *end_ptr != '\0' || errno == ERANGE || parsed > UINT8_MAX) {
    ESP_LOGW(TAG, "Invalid HA motion hits command: %s", payload.c_str());
    return;
  }
  const uint8_t value = static_cast<uint8_t>(parsed);
  const uint8_t motion_on_hits = motion_on ? value : runtime_.config().motion_on_hits;
  const uint8_t motion_off_hits = motion_on ? runtime_.config().motion_off_hits : value;
  (void) handle_motion_hits_write_(motion_on_hits, motion_off_hits);
}

void NativeFrontend::handle_ha_calibrate_command_(const std::string &payload) {
  const std::string token = normalize_text_token(payload);
  if (token == "off") {
    publish_ha_calibrate_(runtime_.is_calibrating());
    return;
  }
  if (token != "on") {
    ESP_LOGW(TAG, "Invalid HA calibrate command: %s", payload.c_str());
    return;
  }
  if (runtime_.is_calibrating()) {
    publish_ha_calibrate_(true);
    return;
  }
  if (!handle_recalibration_write_()) {
    publish_ha_calibrate_(runtime_.is_calibrating());
  }
}

void NativeFrontend::handle_ha_csi_traffic_mode_command_(const std::string &payload) {
  const std::string mode = normalize_text_token(payload);
  if (mode != RUNTIME_CSI_TRAFFIC_MODE_INTERNAL_NAME &&
      mode != RUNTIME_CSI_TRAFFIC_MODE_EXTERNAL_NAME &&
      mode != RUNTIME_CSI_TRAFFIC_MODE_DISABLED_NAME) {
    ESP_LOGW(TAG, "Invalid HA CSI traffic mode command: %s", payload.c_str());
    return;
  }
  (void) handle_csi_traffic_mode_write_(parse_csi_traffic_mode(mode.c_str()));
}

void NativeFrontend::handle_ha_traffic_generator_mode_command_(const std::string &payload) {
  const std::string mode = normalize_text_token(payload);
  if (mode != RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME &&
      mode != RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME) {
    ESP_LOGW(TAG, "Invalid HA traffic generator mode command: %s", payload.c_str());
    return;
  }
  (void) handle_traffic_generator_mode_write_(parse_traffic_mode(mode.c_str()));
}

void NativeFrontend::handle_ha_diagnostics_command_(const std::string &payload) {
  (void) payload;
  publish_ha_diagnostics_();
}

void NativeFrontend::handle_ha_birth_message_(const std::string &topic, const std::string &payload) {
  if (topic != ha_settings_.birth_topic || !frontend_ha_mqtt_enabled()) {
    return;
  }
  if (normalize_text_token(payload) == kHaOnlinePayload) {
    publish_ha_discovery_();
    publish_mqtt_status_(true);
    publish_current_ha_state_();
  }
}

void NativeFrontend::handle_connection_state_(bool connected) {
  client_connected_ = connected;
  if (connected) {
    system_info_refresh_.request();
  }
}

bool NativeFrontend::wifi_configured_() const { return !wifi_info_.ssid.empty(); }

bool NativeFrontend::provisioning_complete_() const {
  return wifi_configured_() && !device_config_.mqtt_host.empty();
}

bool NativeFrontend::ble_should_run_() const { return ble_forced_ || !provisioning_complete_(); }

void NativeFrontend::request_ble_recovery() {
  ble_forced_ = true;
  pending_ble_intent_ = BleIntent::Start;
  ESP_LOGI(TAG, "BLE recovery requested");
}

bool NativeFrontend::start_ble_() {
  if (ble_active_) {
    return true;
  }
  runtime_.set_services_armed(false);
  if (bindings_ == nullptr || !bindings_->setup()) {
    if (wifi_configured_()) {
      runtime_.set_services_armed(true);
    }
    return false;
  }
  // Once setup opens, keep it available across GATT and Wi-Fi reconnects.
  // Only an explicit BLE-off command may hand the radio back to sensing.
  ble_forced_ = true;
  ble_active_ = true;
  ESP_LOGI(TAG, "BLE setup mode; CSI paused");
  return true;
}

void NativeFrontend::stop_ble_() {
  if (!ble_active_) {
    return;
  }
  if (bindings_ != nullptr) {
    bindings_->shutdown();
  }
  ble_active_ = false;
  client_connected_ = false;
  update_live_telemetry_enabled_();
  if (wifi_configured_()) {
    runtime_.set_services_armed(true);
  }
  ESP_LOGI(TAG, "BLE stopped");
}

void NativeFrontend::refresh_ble_policy_() {
  if (!runtime_.is_setup_complete()) {
    return;
  }
  if (ble_should_run_()) {
    pending_ble_intent_ = BleIntent::Start;
    return;
  }
  if (ble_active_ && !client_connected_) {
    pending_ble_intent_ = BleIntent::Stop;
  }
}

void NativeFrontend::apply_pending_ble_intent_() {
  const BleIntent intent = pending_ble_intent_;
  pending_ble_intent_ = BleIntent::Unchanged;
  if (intent == BleIntent::Start) {
    if (!start_ble_()) {
      ESP_LOGE(TAG, "BLE start failed");
    }
  } else if (intent == BleIntent::Stop) {
    stop_ble_();
  }
}

bool NativeFrontend::handle_ble_mode_write_(bool enable, std::string *message) {
  if (enable) {
    request_ble_recovery();
    if (message != nullptr) {
      *message = ble_active_ ? "ble already active" : "ble start requested";
    }
    return true;
  }
  if (!wifi_configured_()) {
    ESP_LOGW(TAG, "Refusing to stop BLE while Wi-Fi is unconfigured");
    if (message != nullptr) {
      *message = "cannot stop BLE while Wi-Fi is unconfigured";
    }
    return false;
  }
  if (device_config_.mqtt_host.empty()) {
    ESP_LOGW(TAG, "Refusing to stop BLE while MQTT is unconfigured");
    if (message != nullptr) {
      *message = "cannot stop BLE while MQTT is unconfigured";
    }
    return false;
  }
  ble_forced_ = false;
  pending_ble_intent_ = BleIntent::Stop;
  if (message != nullptr) {
    *message = "ble stop requested";
  }
  return true;
}

void NativeFrontend::handle_live_telemetry_subscription_(bool subscribed) {
  (void) subscribed;
}

void NativeFrontend::update_live_telemetry_enabled_() {
  runtime_.set_live_telemetry_enabled(mqtt_connected_);
}

void NativeFrontend::setup_mqtt_() {
  (void) setup_frontend_mqtt_transport(mqtt_transport_,
                                       device_config_,
                                       [this](const std::string &payload) { this->handle_mqtt_command_(payload); },
                                       [this](bool connected) {
                                         this->mqtt_connected_ = connected;
                                         this->mqtt_ha_online_ = connected && frontend_ha_mqtt_enabled();
                                         this->update_live_telemetry_enabled_();
                                         this->system_info_refresh_.request();
                                         if (connected) {
                                           this->publish_mqtt_info_();
                                           this->publish_mqtt_status_(true);
                                           this->publish_current_mqtt_ota_status_();
                                           this->setup_ha_mqtt_();
                                           this->publish_ha_discovery_();
                                           this->publish_current_ha_state_();
                                         }
                                       },
                                       TAG);
}

void NativeFrontend::setup_ha_mqtt_() {
  if (!frontend_ha_mqtt_enabled() || mqtt_transport_ == nullptr) {
    return;
  }
  ha_settings_ = build_frontend_ha_mqtt_settings(device_config_, device_info_, "native");
  (void) mqtt_transport_->subscribe(
      ha_settings_.birth_topic,
      [this](const std::string &topic, const std::string &payload) { this->handle_ha_birth_message_(topic, payload); });
  (void) mqtt_transport_->subscribe(ha_settings_.threshold_command_topic,
                                    [this](const std::string &, const std::string &payload) {
                                      this->handle_ha_threshold_command_(payload);
                                    });
  if (runtime_.capabilities().supports_runtime_motion_hits_updates) {
    (void) mqtt_transport_->subscribe(ha_settings_.motion_on_hits_command_topic,
                                      [this](const std::string &, const std::string &payload) {
                                        this->handle_ha_motion_hits_command_(true, payload);
                                      });
    (void) mqtt_transport_->subscribe(ha_settings_.motion_off_hits_command_topic,
                                      [this](const std::string &, const std::string &payload) {
                                        this->handle_ha_motion_hits_command_(false, payload);
                                      });
  }
  (void) mqtt_transport_->subscribe(ha_settings_.calibrate_command_topic,
                                    [this](const std::string &, const std::string &payload) {
                                      this->handle_ha_calibrate_command_(payload);
                                    });
  if (runtime_.capabilities().supports_runtime_detector_selection) {
    (void) mqtt_transport_->subscribe(ha_settings_.detector_command_topic,
                                      [this](const std::string &, const std::string &payload) {
                                        const std::string detector = normalize_text_token(payload);
                                        if (detector == RUNTIME_DETECTION_ALGORITHM_LIGHTWEIGHT_NAME ||
                                            detector == RUNTIME_DETECTION_ALGORITHM_HIGH_ACCURACY_NAME) {
                                          (void) this->handle_detector_write_(parse_detection_algorithm(detector.c_str()));
                                        }
                                      });
  }
  if (runtime_.capabilities().supports_traffic_control) {
    (void) mqtt_transport_->subscribe(ha_settings_.csi_traffic_mode_command_topic,
                                      [this](const std::string &, const std::string &payload) {
                                        this->handle_ha_csi_traffic_mode_command_(payload);
                                      });
    (void) mqtt_transport_->subscribe(ha_settings_.traffic_generator_mode_command_topic,
                                      [this](const std::string &, const std::string &payload) {
                                        this->handle_ha_traffic_generator_mode_command_(payload);
                                      });
  }
  (void) mqtt_transport_->subscribe(ha_settings_.diagnostics_command_topic,
                                    [this](const std::string &, const std::string &payload) {
                                      this->handle_ha_diagnostics_command_(payload);
                                    });
}

void NativeFrontend::publish_ha_discovery_() {
  if (!frontend_ha_mqtt_enabled() || mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  ha_settings_ = build_frontend_ha_mqtt_settings(device_config_, device_info_, "native");
  const auto messages = build_frontend_ha_discovery_messages(ha_settings_,
                                                             device_info_,
                                                             runtime_.capabilities().supports_runtime_detector_selection,
                                                             runtime_.capabilities().supports_runtime_motion_hits_updates,
                                                             runtime_.capabilities().supports_traffic_control);
  for (const auto &message : messages) {
    (void) mqtt_transport_->publish(message.topic, message.payload, true);
  }
}

void NativeFrontend::publish_ha_motion_(MotionState state) {
  if (!ha_mqtt_ready_()) {
    return;
  }
  (void) mqtt_transport_->publish(ha_settings_.motion_state_topic, motion_state_payload(state), false);
}

void NativeFrontend::publish_ha_movement_(float movement) {
  if (!ha_mqtt_ready_()) {
    return;
  }
  (void) mqtt_transport_->publish(ha_settings_.movement_state_topic, float_payload(movement), false);
}

void NativeFrontend::publish_ha_threshold_(float threshold) {
  if (!ha_mqtt_ready_()) {
    return;
  }
  (void) mqtt_transport_->publish(ha_settings_.threshold_state_topic, float_payload(threshold), false);
}

void NativeFrontend::publish_ha_motion_hits_(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  if (!ha_mqtt_ready_()) {
    return;
  }
  if (!runtime_.capabilities().supports_runtime_motion_hits_updates) {
    return;
  }
  char buffer[8];
  std::snprintf(buffer, sizeof(buffer), "%u", static_cast<unsigned>(motion_on_hits));
  (void) mqtt_transport_->publish(ha_settings_.motion_on_hits_state_topic, buffer, false);
  std::snprintf(buffer, sizeof(buffer), "%u", static_cast<unsigned>(motion_off_hits));
  (void) mqtt_transport_->publish(ha_settings_.motion_off_hits_state_topic, buffer, false);
}

void NativeFrontend::publish_ha_calibrate_(bool calibrating) {
  if (!ha_mqtt_ready_()) {
    return;
  }
  (void) mqtt_transport_->publish(ha_settings_.calibrate_state_topic, calibrating ? "ON" : "OFF", false);
}

void NativeFrontend::publish_ha_detector_(const char *detector_name) {
  if (!ha_mqtt_ready_() || detector_name == nullptr) {
    return;
  }
  if (!runtime_.capabilities().supports_runtime_detector_selection) {
    return;
  }
  (void) mqtt_transport_->publish(ha_settings_.detector_state_topic, detector_name, false);
}

void NativeFrontend::publish_ha_traffic_control_(CsiTrafficMode csi_traffic_mode,
                                                 RuntimeTrafficMode traffic_generator_mode) {
  if (!ha_mqtt_ready_() || !runtime_.capabilities().supports_traffic_control) {
    return;
  }
  (void) mqtt_transport_->publish(ha_settings_.csi_traffic_mode_state_topic, csi_traffic_mode_name(csi_traffic_mode),
                                  false);
  (void) mqtt_transport_->publish(ha_settings_.traffic_generator_mode_state_topic,
                                  traffic_mode_name(traffic_generator_mode), false);
}

void NativeFrontend::publish_ha_diagnostics_() {
  if (!ha_mqtt_ready_()) {
    return;
  }
  for (const FrontendHaDiagnosticSensor &sensor : ha_settings_.diagnostic_sensors) {
    const std::string payload = diagnostic_state_payload(sensor.key, latest_diagnostics_);
    if (payload.empty()) {
      continue;
    }
    (void) mqtt_transport_->publish(sensor.state_topic, payload, false);
  }
}

bool NativeFrontend::ha_mqtt_ready_() {
  if (!mqtt_ha_online_ || mqtt_transport_ == nullptr) {
    return false;
  }
  if (ha_settings_.movement_state_topic.empty()) {
    ha_settings_ = build_frontend_ha_mqtt_settings(device_config_, device_info_, "native");
  }
  return !ha_settings_.movement_state_topic.empty();
}

void NativeFrontend::publish_ha_state_(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    return;
  }
  publish_ha_motion_(snapshot.motion_state);
  publish_ha_movement_(snapshot.movement_metric);
  publish_ha_threshold_(snapshot.threshold);
  publish_ha_motion_hits_(runtime_.config().motion_on_hits, runtime_.config().motion_off_hits);
  publish_ha_calibrate_(runtime_.is_calibrating() || snapshot.calibrating);
  publish_ha_detector_(snapshot.detector_name);
  publish_ha_traffic_control_(runtime_.config().csi_traffic_mode, runtime_.config().traffic_generator_mode);
}

void NativeFrontend::publish_current_ha_state_() { publish_ha_state_(runtime_.snapshot()); }

void NativeFrontend::publish_mqtt_info_() {
  const EspectreDeviceInfo info = mqtt_protocol_device_info_();
  (void) publish_frontend_mqtt_message(
      mqtt_transport_, device_config_, "info", espectre_info_payload(device_config_, info), true);
}

void NativeFrontend::publish_mqtt_commands_() {
  const EspectreDeviceInfo info = mqtt_protocol_device_info_();
  (void) publish_frontend_mqtt_message(
      mqtt_transport_, device_config_, "commands/catalog", espectre_commands_payload(device_config_, info), false);
}

EspectreDeviceInfo NativeFrontend::mqtt_protocol_device_info_() const {
  EspectreDeviceInfo info =
      normalize_protocol_device_info(device_info_, &runtime_.snapshot(), ota_service_ != nullptr, "native", CONFIG_IDF_TARGET);
  info.supports_info = true;
  info.supports_stats = true;
  info.supports_runtime_threshold = runtime_.capabilities().supports_runtime_threshold_updates;
  info.supports_runtime_motion_hits = runtime_.capabilities().supports_runtime_motion_hits_updates;
  info.supports_runtime_detector = runtime_.capabilities().supports_runtime_detector_selection;
  info.supports_manual_recalibration = runtime_.capabilities().supports_manual_recalibration;
  info.supports_traffic_control = runtime_.capabilities().supports_traffic_control;
  info.supports_ble = true;
  info.csi_traffic_mode = csi_traffic_mode_name(runtime_.config().csi_traffic_mode);
  info.traffic_mode = traffic_mode_name(runtime_.config().traffic_generator_mode);
  info.csi_target_pps = runtime_.config().csi_target_pps;
  return info;
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
          device_config_,
          runtime_.snapshot(),
          now,
          now / 1000U,
          current_free_memory_kb(),
          last_loop_time_ms_,
          &latest_diagnostics_),
      false);
}

void NativeFrontend::sample_diagnostics_(uint32_t now_ms) {
  latest_diagnostics_ = diagnostics_sampler_.sample(runtime_.diagnostics(), now_ms);
}

void NativeFrontend::publish_mqtt_ota_status_(const EspectreOtaStatus &status) {
  (void) publish_frontend_mqtt_ota_status(mqtt_transport_, device_config_, status, now_ms_());
}

void NativeFrontend::publish_current_mqtt_ota_status_() {
  if (ota_service_ == nullptr) {
    return;
  }
  EspectreOtaStatus status = ota_service_->status();
  if ((status.current_version.empty() || status.current_version == "unknown") &&
      !device_info_.firmware_version.empty()) {
    status.current_version = device_info_.firmware_version;
  }
  publish_mqtt_ota_status_(status);
}

void NativeFrontend::publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message) {
  (void) publish_frontend_mqtt_command_result(mqtt_transport_, device_config_, command, accepted, message);
}

void NativeFrontend::append_ota_sysinfo_lines_(std::vector<std::string> *lines) const {
  if (lines == nullptr || ota_service_ == nullptr) {
    return;
  }
  const EspectreOtaStatus status = ota_service_->status();
  lines->emplace_back(std::string("ota_state=") + ota_state_name(status.state));
  lines->emplace_back(std::string("ota_busy=") + (status.busy ? "true" : "false"));
  lines->emplace_back(std::string("ota_update_available=") + (status.update_available ? "true" : "false"));
  lines->emplace_back(std::string("ota_current_version=") + status.current_version);
  lines->emplace_back(std::string("ota_target_version=") + status.target_version);
  lines->emplace_back(std::string("ota_channel=") + status.channel);
  lines->emplace_back(std::string("ota_message=") + status.message);
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
                          false,
                          false,
                          false,
                          false,
                          false,
                          false,
                          runtime_.capabilities().supports_extended_diagnostics,
                          ota_service_ != nullptr,
                          ESPECTRE_WIFI_DUAL_BAND != 0},
      device_config_,
      sysinfo_device_info,
      true,
      true,
      mqtt_transport_ != nullptr && mqtt_transport_->connected(),
      SysinfoWifiState{
          wifi_info_.ssid,
          wifi_info_.bssid,
          wifi_info_.channel,
          device_info_.network.channel > 0U,
          wifi_info_.band_policy,
      },
  });
  lines.emplace_back(std::string("ble_active=") + (ble_active_ ? "true" : "false"));
  char line[96];
  visit_runtime_diagnostics(runtime_.config(), runtime_.snapshot(), [&line, &lines](const char *key, const char *value) {
    std::snprintf(line, sizeof(line), "%s=%s", key, value);
    lines.emplace_back(line);
  });
  append_ota_sysinfo_lines_(&lines);
  append_sysinfo_end_line(&lines);
  bindings_->replace_sysinfo_lines(std::move(lines));
}

uint32_t NativeFrontend::now_ms_() const { return monotonic_now_ms(); }

}  // namespace espectre
