/*
 * ESPectre - Native Frontend Adapter
 *
 * Bridges runtime events and control flows to Direct WebSocket, MQTT, and OTA
 * services.
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

#include "espectre_log.h"
#include "esp_timer.h"
#include "frontend_control_helpers.h"
#include "frontend_mqtt_helpers.h"
#include "protocol_json.h"
#include "runtime_time.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "sdkconfig.h"
#include "wifi_band_helpers.h"

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

#if defined(ESP_PLATFORM)
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
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

float current_free_memory_kb() {
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  return static_cast<float>(heap_caps_get_free_size(MALLOC_CAP_DEFAULT)) / 1024.0f;
#else
  return 0.0f;
#endif
}

uint32_t current_task_stack_high_water_bytes() {
#if defined(ESP_PLATFORM) && INCLUDE_uxTaskGetStackHighWaterMark
  return static_cast<uint32_t>(uxTaskGetStackHighWaterMark(nullptr));
#else
  return 0U;
#endif
}

}  // namespace

NativeFrontend::NativeFrontend(IMqttTransport *mqtt_transport,
                               IOtaService *ota_service,
                               IDirectWebSocketService *direct_service)
    : mqtt_transport_(mqtt_transport),
      ota_service_(ota_service),
      direct_service_(direct_service) {}

void NativeFrontend::set_runtime_config(const RuntimeConfig &config) {
  RuntimeConfig native_config = config;
  native_config.runtime_detector_selection_enabled = true;
  runtime_.set_config(native_config);
}

void NativeFrontend::set_device_config(const EspectreDeviceConfig &config) {
  device_config_ = config;
}

void NativeFrontend::set_device_info(const EspectreDeviceInfo &info) {
  device_info_ = info;
  if (peer_discovery_ != nullptr) {
    peer_discovery_->set_wifi_ready(!device_info_.network.ip_address.empty());
  }
  refresh_direct_service_();
}

void NativeFrontend::set_peer_discovery_service(IPeerDiscoveryService *service) {
  peer_discovery_ = service;
  if (peer_discovery_ != nullptr) {
    peer_discovery_->set_wifi_ready(!device_info_.network.ip_address.empty());
  }
  refresh_direct_service_();
}

void NativeFrontend::set_wifi_provisioning_info(const WifiProvisioningInfo &info) {
  wifi_info_ = info;
  refresh_direct_service_();
}

void NativeFrontend::set_provisioning_command_callback(ProvisioningCommandCallback callback) {
  provisioning_command_callback_ = std::move(callback);
}

void NativeFrontend::set_device_config_change_callback(DeviceConfigChangeCallback callback) {
  device_config_change_callback_ = std::move(callback);
}

void NativeFrontend::prepare_for_wifi_reconfigure() {
  if (wifi_reconfigure_quiesced_) {
    return;
  }
  wifi_reconfigure_quiesced_ = true;
  runtime_.set_services_armed(false);
}

void NativeFrontend::resume_after_wifi_reconfigure() {
  if (!wifi_reconfigure_quiesced_) {
    return;
  }
  wifi_reconfigure_quiesced_ = false;
  if (!ota_frontend_quiesced_ && wifi_configured_()) {
    runtime_.set_services_armed(true);
  }
}

bool NativeFrontend::setup() {
  update_live_telemetry_enabled_();

  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    return false;
  }
  const uint32_t diagnostics_now_ms = now_ms_();
  const RuntimeDiagnosticsSnapshot diagnostics = runtime_.diagnostics();
  diagnostics_sampler_.reset(diagnostics, diagnostics_now_ms);
  latest_diagnostics_ = diagnostics_sampler_.sample(diagnostics, diagnostics_now_ms);

  if (ota_service_ != nullptr) {
    ota_service_->set_prepare_for_update_callback([this]() { this->prepare_for_ota_(); });
    ota_service_->set_status_callback([this](const EspectreOtaStatus &status) {
      if (this->ota_frontend_quiesced_) {
        if (status.state == EspectreOtaState::ERROR) {
          this->resume_after_ota_error_();
        }
        return;
      }
      this->publish_ota_status_(status);
    });
  }

  setup_mqtt_();
  refresh_direct_service_();
  ESP_LOGI(TAG, "Native frontend initialized");
  return true;
}

void NativeFrontend::loop() {
  const int64_t loop_started_us = esp_timer_get_time();
  runtime_.loop();
  drain_pending_runtime_events_();
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->loop();
    drain_pending_ha_snapshot_();
  }
  if (direct_service_ != nullptr && direct_service_->running()) {
    direct_service_->loop();
  }
  if (peer_discovery_ != nullptr) {
    peer_discovery_->loop();
  }
  if (ota_service_ != nullptr) {
    ota_service_->loop();
  }
  last_loop_time_ms_ = static_cast<float>(esp_timer_get_time() - loop_started_us) / 1000.0f;
}

void NativeFrontend::shutdown() {
  live_telemetry_pending_ = false;
  motion_state_pending_ = false;
  mqtt_connected_ = false;
  mqtt_ha_online_ = false;
  pending_ha_discovery_.clear();
  pending_ha_discovery_index_ = 0U;
  pending_ha_state_ = false;
  stop_direct_service_();
  if (peer_discovery_ != nullptr) {
    peer_discovery_->shutdown();
  }
  update_live_telemetry_enabled_();
  publish_mqtt_status_(false);
  runtime_.shutdown();
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->shutdown();
  }
  if (ota_service_ != nullptr) {
    ota_service_->shutdown();
  }
}

NativeFrontend::~NativeFrontend() { shutdown(); }

void NativeFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    return;
  }
  pending_motion_state_ = snapshot;
  motion_state_pending_ = true;
}

void NativeFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) packets_received;
  sample_diagnostics_(now_ms_());
  if (!snapshot.ready_to_publish) {
    return;
  }
  if (runtime_.capabilities().supports_runtime_detector_selection && snapshot.detector_name != nullptr) {
    publish_ha_detector_(snapshot.detector_name);
  }
}

void NativeFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  publish_mqtt_telemetry_(snapshot, now_ms_());
  if (snapshot.ready_to_publish) {
    publish_ha_threshold_(snapshot.threshold);
  }
}

void NativeFrontend::on_detector_changed(const RuntimeSnapshot &snapshot) {
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
  calibration_started_ = true;
  calibration_start_threshold_ = snapshot.threshold;
  publish_ha_calibrate_(true);
  if (!protocol_recalibration_command_active_) {
    publish_runtime_status_state_();
  }
}

void NativeFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  const bool threshold_changed =
      !calibration_started_ || std::fabs(snapshot.threshold - calibration_start_threshold_) > 1.0e-6f;
  calibration_started_ = false;
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
  publish_ha_calibrate_(false);
  if (snapshot.ready_to_publish) {
    publish_ha_threshold_(snapshot.threshold);
  }
  publish_runtime_status_state_();
  if (success && threshold_changed) {
    publish_runtime_config_state_();
  }
}

void NativeFrontend::on_live_telemetry(float movement, float threshold) {
  if (!runtime_.snapshot().ready_to_publish) {
    return;
  }
  RuntimeSnapshot snapshot = runtime_.snapshot();
  snapshot.movement_metric = movement;
  snapshot.threshold = threshold;
  pending_live_telemetry_ = snapshot;
  live_telemetry_pending_ = true;
}

void NativeFrontend::drain_pending_runtime_events_() {
  if (motion_state_pending_) {
    const RuntimeSnapshot snapshot = pending_motion_state_;
    motion_state_pending_ = false;
    publish_ha_motion_(snapshot.motion_state);
    const uint32_t now = now_ms_();
    const std::string payload =
        espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U, "native");
    fan_out_payload_(nullptr, "telemetry", payload, false, true);
  }
  if (live_telemetry_pending_) {
    const RuntimeSnapshot snapshot = pending_live_telemetry_;
    live_telemetry_pending_ = false;
    const uint32_t now = now_ms_();
    const char *frontend = device_info_.frontend.empty() ? "native" : device_info_.frontend.c_str();
    const std::string payload =
        espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U, frontend);
    fan_out_payload_("telemetry", "telemetry", payload, false, true);
    publish_ha_movement_(snapshot.movement_metric);
  }
}

void NativeFrontend::on_runtime_fault(const char *message) {
  std::string data{"{"};
  append_json_pair(&data, "message", message != nullptr ? message : "runtime fault", true);
  data += "}";
  publish_direct_event_("fault", data);
}

void NativeFrontend::handle_mqtt_command_(const std::string &payload) {
  EspectreCommand command;
  std::string parse_error;
  if (!parse_espectre_command(payload, &command, &parse_error)) {
    if (command.command.empty()) {
      command.command = "unknown";
    }
    FrontendCommandResult result;
    result.handled = true;
    result.command = std::move(command);
    result.code = "invalid_params";
    result.message = std::move(parse_error);
    publish_mqtt_command_result_(result);
    return;
  }
  const FrontendCommandResult result =
      dispatch_command_(command, FrontendCommandOrigin::MQTT, false);
  publish_mqtt_command_result_(result);
}

FrontendCommandResult NativeFrontend::dispatch_command_(const EspectreCommand &command,
                                                         FrontendCommandOrigin origin,
                                                         bool allow_local_config) {
  const FrontendCommandCapabilities capabilities{
      true,
      true,
      true,
      true,
      true,
      allow_local_config,
      allow_local_config,
      true,
      runtime_.capabilities().supports_runtime_threshold_updates,
      runtime_.capabilities().supports_runtime_motion_hits_updates,
      runtime_.capabilities().supports_traffic_control,
      runtime_.capabilities().supports_runtime_detector_selection,
      runtime_.capabilities().supports_manual_recalibration,
      ota_service_ != nullptr,
      allow_local_config && peer_discovery_enabled_,
  };
  FrontendCommandResult result = command_engine_.execute(
      command,
      FrontendCommandContext{origin},
      ota_service_,
      device_info_.firmware_version.c_str(),
      capabilities,
      [this, allow_local_config, capabilities](const EspectreCommand &read) {
        if (read.command == "capabilities") {
          return espectre_capabilities_payload(this->device_config_,
                                               this->mqtt_protocol_device_info_(),
                                               true,
                                               true,
                                               true,
                                               allow_local_config,
                                               allow_local_config,
                                               capabilities.supports_peer_discovery);
        }
        if (read.command == "info") {
          return espectre_info_payload(this->device_config_, this->mqtt_protocol_device_info_());
        }
        if (read.command == "status") {
          return this->direct_status_payload_(!this->device_info_.network.ip_address.empty());
        }
        if (read.command == "config") return this->direct_config_payload_(allow_local_config);
        if (read.command == "diagnostics") return this->direct_diagnostics_payload_();
        if (read.command == "ota_status" && this->ota_service_ != nullptr) {
          return espectre_ota_status_payload(this->device_config_, this->current_ota_status_(), this->now_ms_());
        }
        return std::string{};
      },
      [this](const std::string &device_label, std::string *message) {
        EspectreDeviceConfig updated_config = this->device_config_;
        updated_config.device_label = device_label;
        if (this->device_config_change_callback_ &&
            !this->device_config_change_callback_(updated_config, false, message)) {
          return false;
        }
        this->set_device_config(updated_config);
        this->publish_ha_discovery_();
        this->publish_current_ha_state_();
        if (message != nullptr && message->empty()) {
          *message = "device label updated";
        }
        return true;
      },
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
        this->protocol_recalibration_command_active_ = true;
        const bool accepted = this->handle_recalibration_write_();
        this->protocol_recalibration_command_active_ = false;
        if (message != nullptr && message->empty()) {
          *message = accepted ? "recalibration started" : "recalibration rejected";
        }
        return accepted;
      },
      [this](const EspectreCommand &wifi_command, bool clear, std::string *message) {
        if (!this->provisioning_command_callback_) {
          if (message != nullptr) {
            *message = "Wi-Fi provisioning is unavailable";
          }
          return false;
        }
        if (clear) {
          return this->provisioning_command_callback_("CLEAR_WIFI", message);
        }
        const std::string ssid = wifi_command.has_wifi_ssid ? wifi_command.wifi_ssid : this->wifi_info_.ssid;
        if (ssid.empty()) {
          if (message != nullptr) {
            *message = "SSID is required when no Wi-Fi configuration exists";
          }
          return false;
        }
        std::string encoded = "SET_WIFI_CONFIG:ssid=" + encode_urlencoded_component(ssid);
        const auto append_field = [&encoded](const char *name, const std::string &value) {
          encoded += "&";
          encoded += name;
          encoded += "=";
          encoded += encode_urlencoded_component(value);
        };
        if (wifi_command.has_wifi_password) {
          append_field("password", wifi_command.wifi_password);
        }
        if (wifi_command.has_wifi_bssid) {
          append_field("bssid", wifi_command.wifi_bssid);
        }
        if (wifi_command.has_wifi_channel) {
          append_field("channel", std::to_string(wifi_command.wifi_channel));
        }
        if (wifi_command.has_wifi_band_policy) {
          append_field("band_policy", wifi_command.wifi_band_policy);
        }
        return this->provisioning_command_callback_(encoded, message);
      },
      [this](const EspectreCommand &mqtt_command, bool clear, std::string *message) {
        EspectreDeviceConfig updated = this->device_config_;
        if (clear) {
          clear_espectre_mqtt_config(&updated);
        } else {
          updated.mqtt_host = mqtt_command.mqtt_host;
          if (mqtt_command.has_mqtt_port) {
            updated.mqtt_port = mqtt_command.mqtt_port;
          }
          if (mqtt_command.has_mqtt_username) {
            updated.mqtt_username = mqtt_command.mqtt_username;
          }
          if (mqtt_command.has_mqtt_password) {
            updated.mqtt_password = mqtt_command.mqtt_password;
          }
          if (mqtt_command.has_mqtt_topic_prefix) {
            updated.topic_prefix = mqtt_command.mqtt_topic_prefix.empty() ? ESPECTRE_TOPIC_PREFIX
                                                                          : mqtt_command.mqtt_topic_prefix;
          }
        }
        if (this->device_config_change_callback_ &&
            !this->device_config_change_callback_(updated, false, message)) {
          return false;
        }
        this->set_device_config(updated);
        this->setup_mqtt_();
        if (message != nullptr && message->empty()) {
          *message = clear ? "MQTT configuration cleared" : "MQTT configuration saved";
        }
        return true;
      },
      [this](bool enabled, std::string *message) {
        if (this->ota_frontend_quiesced_) {
          if (message != nullptr) {
            *message = "sensing is unavailable during OTA";
          }
          return false;
        }
        this->runtime_.set_services_armed(enabled);
        this->update_live_telemetry_enabled_();
        if (message != nullptr) {
          *message = enabled ? "sensing started" : "sensing stopped";
        }
        return true;
      });
  if (result.accepted) {
    if ((static_cast<uint8_t>(result.changes) & static_cast<uint8_t>(FrontendCommandChange::STATUS)) != 0U) {
      publish_runtime_status_state_();
    }
    if ((static_cast<uint8_t>(result.changes) & static_cast<uint8_t>(FrontendCommandChange::CONFIG)) != 0U) {
      publish_runtime_config_state_();
    }
    if ((static_cast<uint8_t>(result.changes) & static_cast<uint8_t>(FrontendCommandChange::INFO)) != 0U) {
      const std::string payload = espectre_info_payload(device_config_, mqtt_protocol_device_info_());
      (void) publish_frontend_mqtt_message(
          mqtt_transport_, device_config_, "info", payload, true);
      publish_direct_event_("info", payload);
    }
  }
  return result;
}

std::string NativeFrontend::handle_direct_request_(const DirectWebSocketRequest &request) {
  EspectreCommand command;
  std::string parse_error;
  if (!direct_websocket_request_to_command(request, &command, &parse_error)) {
    return direct_websocket_error_response(request.id, "invalid_params", parse_error.c_str());
  }

  const FrontendCommandResult result =
      dispatch_command_(command, FrontendCommandOrigin::DIRECT, true);
  if (!result.accepted) {
    return direct_websocket_error_response(request.id, result.code.c_str(), result.message.c_str());
  }
  std::string response{"{"};
  append_json_pair(&response, "command", result.command.command.c_str(), true);
  append_json_pair(&response, "code", result.code.c_str());
  append_json_pair(&response, "message", result.message.c_str());
  if (!result.data_json.empty()) response += ",\"data\":" + result.data_json;
  response += "}";
  return direct_websocket_success_response(request.id, response);
}

IDirectWebSocketService::DeferredRequestResult NativeFrontend::handle_deferred_direct_request_(
    uint64_t connection_token,
    const DirectWebSocketRequest &request) {
  if (request.method != ESPECTRE_PEER_DISCOVERY_METHOD) {
    return {false, handle_direct_request_(request)};
  }
  if (!peer_discovery_enabled_ || peer_discovery_ == nullptr) {
    return {false,
            direct_websocket_error_response(
                request.id, "unsupported", "peer discovery is unavailable")};
  }
  std::vector<JsonObjectField> params;
  if (!parse_json_object_fields(request.params, &params) || !params.empty()) {
    return {false,
            direct_websocket_error_response(
                request.id, "invalid_params", "discover_peers does not accept parameters")};
  }
  if (peer_discovery_->active()) {
    return {false,
            direct_websocket_error_response(
                request.id, "conflict", "a peer discovery request is already active")};
  }
  const bool started = peer_discovery_->start(
      [this, connection_token, request_id = request.id](PeerDiscoverySnapshot snapshot) {
        if (this->direct_service_ == nullptr) {
          return;
        }
        std::string result{"{\"command\":\"discover_peers\",\"code\":\"ok\""};
        append_json_pair(&result, "message", "peer discovery completed");
        result += ",\"data\":" + peer_discovery_snapshot_json(snapshot) + "}";
        (void) this->direct_service_->complete_deferred_response(
            connection_token, direct_websocket_success_response(request_id, result));
      });
  if (!started) {
    return {false,
            direct_websocket_error_response(
                request.id, "unavailable", "peer discovery could not be started")};
  }
  return {true, {}};
}

std::string NativeFrontend::direct_capabilities_payload_() const {
  return espectre_capabilities_payload(device_config_,
                                       mqtt_protocol_device_info_(),
                                       true,
                                       true,
                                       true,
                                       true,
                                       true,
                                       peer_discovery_enabled_);
}

std::string NativeFrontend::direct_status_payload_(bool online) const {
  const uint32_t now = now_ms_();
  std::string out = espectre_status_payload(device_config_, online, now);
  if (!out.empty() && out.back() == '}') {
    out.pop_back();
  }
  out += ",\"wifi_connected\":";
  out += device_info_.network.ip_address.empty() ? "false" : "true";
  out += ",\"mqtt_configured\":";
  out += device_config_.mqtt_host.empty() ? "false" : "true";
  out += ",\"mqtt_connected\":";
  out += mqtt_connected_ ? "true" : "false";
  out += ",\"sensing_enabled\":";
  out += runtime_.services_armed() ? "true" : "false";
  out += ",\"ready_to_publish\":";
  out += runtime_.snapshot().ready_to_publish ? "true" : "false";
  out += ",\"calibrating\":";
  out += runtime_.is_calibrating() ? "true" : "false";
  out += "}";
  return out;
}

std::string NativeFrontend::direct_config_payload_(bool include_local) const {
  const RuntimeConfig &runtime_config = runtime_.config();
  std::string out{"{"};
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", espectre_effective_device_id(device_config_).c_str());
  out += ",\"runtime\":{";
  out += "\"threshold\":" + std::to_string(runtime_.snapshot().threshold);
  append_json_pair(&out, "detector", detection_algorithm_name(runtime_config.detection_algorithm));
  out += ",\"motion_on_hits\":" + std::to_string(runtime_config.motion_on_hits);
  out += ",\"motion_off_hits\":" + std::to_string(runtime_config.motion_off_hits);
  append_json_pair(&out, "csi_traffic_mode", csi_traffic_mode_name(runtime_config.csi_traffic_mode));
  append_json_pair(&out, "traffic_generator_mode", traffic_mode_name(runtime_config.traffic_generator_mode));
  out += ",\"csi_target_pps\":" + std::to_string(runtime_config.csi_target_pps) + "}";
  if (!include_local) {
    out += "}";
    return out;
  }
  out += ",\"device\":{";
  append_json_pair(&out, "device_label", device_config_.device_label.c_str(), true);
  out += "},\"wifi\":{\"configured\":";
  out += wifi_configured_() ? "true" : "false";
  append_json_pair(&out, "ssid", wifi_info_.ssid.c_str());
  append_json_pair(&out, "bssid", wifi_info_.bssid.c_str());
  out += ",\"channel\":" + std::to_string(static_cast<unsigned>(wifi_info_.channel));
  append_json_pair(&out, "band_policy", wifi_band_policy_name(wifi_info_.band_policy));
  append_json_pair(&out, "apply_state", wifi_info_.apply_state.c_str());
  append_json_pair(&out, "apply_message", wifi_info_.apply_message.c_str());
  out += "},\"mqtt\":{\"configured\":";
  out += device_config_.mqtt_host.empty() ? "false" : "true";
  append_json_pair(&out, "host", device_config_.mqtt_host.c_str());
  out += ",\"port\":" + std::to_string(static_cast<unsigned>(device_config_.mqtt_port));
  out += ",\"username_configured\":";
  out += device_config_.mqtt_username.empty() ? "false" : "true";
  append_json_pair(&out, "topic_prefix", device_config_.topic_prefix.c_str());
  out += "}}";
  return out;
}

std::string NativeFrontend::direct_diagnostics_payload_() const {
  const uint32_t now = now_ms_();
  const RuntimeDiagnosticsSnapshot runtime_diagnostics = runtime_.diagnostics();
  std::string out = espectre_diagnostics_payload(device_config_,
                                                 runtime_.snapshot(),
                                                 now,
                                                 now / 1000U,
                                                 current_free_memory_kb(),
                                                 last_loop_time_ms_,
                                                 &latest_diagnostics_);
  if (!out.empty() && out.back() == '}') {
    out.pop_back();
  }
  const DirectWebSocketServiceDiagnostics direct =
      direct_service_ != nullptr ? direct_service_->diagnostics() : DirectWebSocketServiceDiagnostics{};
  const MqttTransportDiagnostics mqtt =
      mqtt_transport_ != nullptr ? mqtt_transport_->diagnostics() : MqttTransportDiagnostics{};
  append_runtime_performance_diagnostics_json(&out, runtime_diagnostics, false);
  out += ",\"task_stack_high_water_bytes\":" + std::to_string(current_task_stack_high_water_bytes());
  out += ",\"direct\":{\"clients\":" + std::to_string(direct_client_count_);
  out += ",\"client_limit\":" + std::to_string(direct.client_limit);
  out += ",\"queue_capacity\":" + std::to_string(direct.queue_capacity);
  out += ",\"queued_messages\":" + std::to_string(direct.queued_messages);
  out += ",\"accepted_connections\":" + std::to_string(direct.accepted_connections);
  out += ",\"rejected_connections\":" + std::to_string(direct.rejected_connections);
  out += ",\"malformed_frames\":" + std::to_string(direct.malformed_frames);
  out += ",\"oversized_frames\":" + std::to_string(direct.oversized_frames);
  out += ",\"rate_limited_requests\":" + std::to_string(direct.rate_limited_requests);
  out += ",\"dropped_telemetry_events\":" + std::to_string(direct.dropped_telemetry_events);
  out += ",\"send_failures\":" + std::to_string(direct.send_failures);
  out += ",\"slow_client_disconnects\":" + std::to_string(direct.slow_client_disconnects) + "}";
  out += ",\"mqtt\":{\"connected\":";
  out += mqtt_transport_ != nullptr && mqtt_transport_->connected() ? "true" : "false";
  out += ",\"queue_capacity\":" + std::to_string(mqtt.queue_capacity);
  out += ",\"outbox_capacity_bytes\":" + std::to_string(mqtt.outbox_capacity_bytes);
  out += ",\"queued_publishes\":" + std::to_string(mqtt.queued_publishes);
  out += ",\"dropped_publishes\":" + std::to_string(mqtt.dropped_publishes);
  out += ",\"publish_failures\":" + std::to_string(mqtt.publish_failures);
  out += ",\"reconnects\":" + std::to_string(mqtt.reconnects) + "}}";
  return out;
}

bool NativeFrontend::handle_threshold_write_(float threshold) {
  if (!runtime_.capabilities().supports_runtime_threshold_updates) {
    ESP_LOGW(TAG, "Runtime threshold updates are not supported");
    return false;
  }

  if (!runtime_.set_threshold_runtime(threshold)) {
    return false;
  }
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
  if (handle_threshold_write_(threshold)) {
    publish_runtime_config_state_();
  }
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
  if (handle_motion_hits_write_(motion_on_hits, motion_off_hits)) {
    publish_runtime_config_state_();
  }
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
  if (handle_csi_traffic_mode_write_(parse_csi_traffic_mode(mode.c_str()))) {
    publish_runtime_config_state_();
  }
}

void NativeFrontend::handle_ha_traffic_generator_mode_command_(const std::string &payload) {
  const std::string mode = normalize_text_token(payload);
  if (mode != RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME &&
      mode != RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME) {
    ESP_LOGW(TAG, "Invalid HA traffic generator mode command: %s", payload.c_str());
    return;
  }
  if (handle_traffic_generator_mode_write_(parse_traffic_mode(mode.c_str()))) {
    publish_runtime_config_state_();
  }
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
  }
}

bool NativeFrontend::wifi_configured_() const { return !wifi_info_.ssid.empty(); }

void NativeFrontend::update_live_telemetry_enabled_() {
  runtime_.set_live_telemetry_enabled(mqtt_connected_ || direct_client_count_ > 0U);
}

void NativeFrontend::refresh_direct_service_() {
  if (direct_service_ == nullptr) {
    return;
  }
  const bool has_station_address = !device_info_.network.ip_address.empty();
  if (!runtime_.is_setup_complete() || !has_station_address || ota_frontend_quiesced_) {
    stop_direct_service_();
    return;
  }
  if (direct_service_->running()) {
    return;
  }
  peer_discovery_enabled_ = false;

  DirectWebSocketServiceConfig config = DirectWebSocketServiceConfig::for_first_party_portals();
#if defined(CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED) && CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED
  config.allow_http_loopback_origins = true;
#endif
  const auto client_count_changed = [this](size_t client_count) {
    this->direct_client_count_ = client_count;
    this->update_live_telemetry_enabled_();
  };
  bool setup = false;
  if (peer_discovery_ != nullptr) {
    setup = direct_service_->setup_deferred(
        config,
        [this](uint64_t token, const DirectWebSocketRequest &request) {
          return this->handle_deferred_direct_request_(token, request);
        },
        client_count_changed);
    peer_discovery_enabled_ = setup;
  }
  if (!setup) {
    setup = direct_service_->setup(
        config,
        [this](const DirectWebSocketRequest &request) { return this->handle_direct_request_(request); },
        client_count_changed);
  }
  if (!setup) {
    ESP_LOGE(TAG, "Direct WebSocket service setup failed");
  }
}

void NativeFrontend::stop_direct_service_() {
  if (direct_service_ != nullptr && direct_service_->running()) {
    direct_service_->shutdown();
  }
  direct_client_count_ = 0U;
  peer_discovery_enabled_ = false;
}

void NativeFrontend::publish_direct_event_(const char *event_name,
                                           const std::string &data_json,
                                           bool replaceable_telemetry) {
  if (direct_service_ == nullptr || !direct_service_->running() || direct_client_count_ == 0U || event_name == nullptr) {
    return;
  }
  (void) direct_service_->publish_event(event_name, data_json, replaceable_telemetry);
}

void NativeFrontend::fan_out_payload_(const char *mqtt_suffix,
                                      const char *direct_event_name,
                                      const std::string &payload,
                                      bool mqtt_retain,
                                      bool replaceable_telemetry) {
  if (mqtt_suffix != nullptr) {
    (void) publish_frontend_mqtt_message(
        mqtt_transport_, device_config_, mqtt_suffix, payload, mqtt_retain);
  }
  if (direct_event_name != nullptr) {
    publish_direct_event_(direct_event_name, payload, replaceable_telemetry);
  }
}

void NativeFrontend::setup_mqtt_() {
  const bool was_connected = mqtt_connected_;
  mqtt_connected_ = false;
  mqtt_ha_online_ = false;
  if (was_connected) {
    update_live_telemetry_enabled_();
    publish_direct_event_("status", direct_status_payload_(!device_info_.network.ip_address.empty()));
  }
  (void) setup_frontend_mqtt_transport(mqtt_transport_,
                                       device_config_,
                                       [this](const std::string &payload) { this->handle_mqtt_command_(payload); },
                                       [this](bool connected) {
                                         this->mqtt_connected_ = connected;
                                         this->mqtt_ha_online_ = connected && frontend_ha_mqtt_enabled();
                                         if (!connected) {
                                           this->pending_ha_discovery_.clear();
                                           this->pending_ha_discovery_index_ = 0U;
                                           this->pending_ha_state_ = false;
                                         }
                                         this->update_live_telemetry_enabled_();
                                         if (connected) {
                                           this->publish_mqtt_capabilities_();
                                           this->publish_mqtt_info_();
                                           this->publish_mqtt_status_(true);
                                           this->publish_mqtt_config_();
                                           this->publish_current_mqtt_ota_status_();
                                           this->setup_ha_mqtt_();
                                           this->publish_ha_discovery_();
                                         }
                                         this->publish_direct_event_(
                                             "status",
                                             this->direct_status_payload_(!this->device_info_.network.ip_address.empty()));
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
                                          const DetectionAlgorithm algorithm =
                                              parse_detection_algorithm(detector.c_str());
                                          if (this->handle_detector_write_(algorithm)) {
                                            this->publish_runtime_config_state_();
                                          }
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
  pending_ha_discovery_ = build_frontend_ha_discovery_messages(
      ha_settings_,
      device_info_,
      runtime_.capabilities().supports_runtime_detector_selection,
      runtime_.capabilities().supports_runtime_motion_hits_updates,
      runtime_.capabilities().supports_traffic_control);
  pending_ha_discovery_index_ = 0U;
  pending_ha_state_ = true;
  drain_pending_ha_snapshot_();
}

void NativeFrontend::drain_pending_ha_snapshot_() {
  if (!mqtt_ha_online_ || mqtt_transport_ == nullptr || !mqtt_transport_->connected()) {
    return;
  }
  while (pending_ha_discovery_index_ < pending_ha_discovery_.size()) {
    const MqttTransportDiagnostics diagnostics = mqtt_transport_->diagnostics();
    if (diagnostics.queue_capacity > 0U && diagnostics.queued_publishes >= diagnostics.queue_capacity) {
      return;
    }
    const FrontendHaDiscoveryMessage &message = pending_ha_discovery_[pending_ha_discovery_index_];
    if (!mqtt_transport_->publish(message.topic, message.payload, true)) {
      return;
    }
    pending_ha_discovery_index_ += 1U;
  }
  if (!pending_ha_discovery_.empty()) {
    pending_ha_discovery_.clear();
    pending_ha_discovery_index_ = 0U;
  }
  if (pending_ha_state_ && mqtt_transport_->diagnostics().queued_publishes == 0U) {
    pending_ha_state_ = false;
    publish_current_ha_state_();
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

void NativeFrontend::publish_runtime_config_state_() {
  publish_mqtt_config_();
  publish_direct_event_("config", direct_config_payload_());
}

void NativeFrontend::publish_runtime_status_state_() {
  const std::string payload = direct_status_payload_(!device_info_.network.ip_address.empty());
  (void) publish_frontend_mqtt_message(mqtt_transport_, device_config_, "status", payload, true);
  publish_direct_event_("status", payload);
}

void NativeFrontend::publish_mqtt_info_() {
  const EspectreDeviceInfo info = mqtt_protocol_device_info_();
  (void) publish_frontend_mqtt_message(
      mqtt_transport_, device_config_, "info", espectre_info_payload(device_config_, info), true);
}

void NativeFrontend::publish_mqtt_capabilities_() {
  const EspectreDeviceInfo info = mqtt_protocol_device_info_();
  (void) publish_frontend_mqtt_message(
      mqtt_transport_,
      device_config_,
      "capabilities",
      espectre_capabilities_payload(device_config_, info, true, true, true),
      true);
}

EspectreDeviceInfo NativeFrontend::mqtt_protocol_device_info_() const {
  EspectreDeviceInfo info =
      normalize_protocol_device_info(device_info_, &runtime_.snapshot(), ota_service_ != nullptr, "native", CONFIG_IDF_TARGET);
  info.supports_info = true;
  info.supports_diagnostics = true;
  info.supports_device_config = true;
  info.supports_runtime_threshold = runtime_.capabilities().supports_runtime_threshold_updates;
  info.supports_runtime_motion_hits = runtime_.capabilities().supports_runtime_motion_hits_updates;
  info.supports_runtime_detector = runtime_.capabilities().supports_runtime_detector_selection;
  info.supports_manual_recalibration = runtime_.capabilities().supports_manual_recalibration;
  info.supports_traffic_control = runtime_.capabilities().supports_traffic_control;
  info.csi_traffic_mode = csi_traffic_mode_name(runtime_.config().csi_traffic_mode);
  info.traffic_mode = traffic_mode_name(runtime_.config().traffic_generator_mode);
  info.csi_target_pps = runtime_.config().csi_target_pps;
  info.evaluation_interval_ms = runtime_.config().evaluation_interval_ms;
  info.publish_interval_ms = runtime_.config().publish_interval_ms;
  return info;
}

void NativeFrontend::publish_mqtt_status_(bool online) {
  (void) publish_frontend_mqtt_message(
      mqtt_transport_, device_config_, "status", direct_status_payload_(online), true);
}

void NativeFrontend::publish_mqtt_telemetry_(const RuntimeSnapshot &snapshot, uint32_t now) {
  const char *frontend = device_info_.frontend.empty() ? "native" : device_info_.frontend.c_str();
  (void) publish_frontend_mqtt_message(mqtt_transport_,
                                       device_config_,
                                       "telemetry",
                                       espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U, frontend),
                                       false);
}

void NativeFrontend::publish_mqtt_config_() {
  (void) publish_frontend_mqtt_message(
      mqtt_transport_,
      device_config_,
      "config",
      direct_config_payload_(false),
      true);
}

EspectreOtaStatus NativeFrontend::current_ota_status_() const {
  EspectreOtaStatus status = ota_service_ != nullptr ? ota_service_->status() : EspectreOtaStatus{};
  if ((status.current_version.empty() || status.current_version == "unknown") &&
      !device_info_.firmware_version.empty()) {
    status.current_version = device_info_.firmware_version;
  }
  return status;
}

void NativeFrontend::publish_ota_status_(const EspectreOtaStatus &status) {
  EspectreOtaStatus normalized = status;
  if ((normalized.current_version.empty() || normalized.current_version == "unknown") &&
      !device_info_.firmware_version.empty()) {
    normalized.current_version = device_info_.firmware_version;
  }
  const std::string payload = espectre_ota_status_payload(device_config_, normalized, now_ms_());
  fan_out_payload_("ota_status", "ota_status", payload, true);
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
  publish_mqtt_ota_status_(current_ota_status_());
}

void NativeFrontend::publish_mqtt_command_result_(const FrontendCommandResult &result) {
  (void) publish_frontend_mqtt_command_result(mqtt_transport_, device_config_, result);
}

void NativeFrontend::prepare_for_ota_() {
  if (ota_frontend_quiesced_) {
    return;
  }
  ota_frontend_quiesced_ = true;
  mqtt_connected_ = false;
  mqtt_ha_online_ = false;
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->shutdown();
  }
  stop_direct_service_();
  runtime_.quiesce_for_ota();
}

void NativeFrontend::resume_after_ota_error_() {
  if (!ota_frontend_quiesced_) {
    return;
  }
  ota_frontend_quiesced_ = false;
  if (wifi_configured_()) {
    runtime_.set_services_armed(true);
  }
  setup_mqtt_();
  refresh_direct_service_();
}

uint32_t NativeFrontend::now_ms_() const { return monotonic_now_ms(); }

}  // namespace espectre
