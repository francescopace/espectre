/*
 * ESPectre - Native Frontend Adapter
 *
 * Bridges runtime events and control flows to Direct HTTP, MQTT, and OTA
 * services.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "native_frontend.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

#include "direct_wifi_snapshot_esp_idf.h"
#include "esp_timer.h"
#include "espectre_log.h"
#include "frontend_command_engine.h"
#include "frontend_mqtt_helpers.h"
#include "protocol_json.h"
#include "runtime_time.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "sdkconfig.h"
#include "wifi_band_helpers.h"

#if defined(ESP_PLATFORM)
#include <esp_random.h>
#endif

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
                               IDirectHttpService *direct_service)
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
  refresh_peer_candidate_();
}

void NativeFrontend::set_device_info(const EspectreDeviceInfo &info) {
  device_info_ = info;
  if (peer_discovery_ != nullptr) {
    peer_discovery_->set_wifi_ready(!device_info_.network.ip_address.empty());
  }
  refresh_peer_candidate_();
  refresh_direct_service_();
}

void NativeFrontend::set_peer_discovery_service(IPeerDiscoveryService *service) {
  peer_discovery_ = service;
  if (peer_discovery_ != nullptr) {
    peer_discovery_->set_wifi_ready(!device_info_.network.ip_address.empty());
  }
  refresh_peer_candidate_();
  refresh_direct_service_();
}

void NativeFrontend::set_wifi_provisioning_info(const WifiProvisioningInfo &info) {
  wifi_info_ = info;
  refresh_direct_service_();
}

void NativeFrontend::set_provisioning_command_callback(ProvisioningCommandCallback callback) {
  provisioning_command_callback_ = std::move(callback);
}

void NativeFrontend::set_wifi_scan_callback(WifiScanCallback callback) {
  wifi_scan_callback_ = std::move(callback);
}

void NativeFrontend::set_device_config_change_callback(DeviceConfigChangeCallback callback) {
  device_config_change_callback_ = std::move(callback);
}

void NativeFrontend::prepare_for_wifi_reconfigure() {
  if (wifi_reconfigure_quiesced_) {
    return;
  }
  wifi_reconfigure_quiesced_ = true;
  wifi_reconfigure_resume_pending_ = false;
  runtime_.set_services_armed(false);
}

void NativeFrontend::resume_after_wifi_reconfigure() {
  if (!wifi_reconfigure_quiesced_) {
    return;
  }
  // StandaloneWifiService and the runtime receive the same GOT_IP transition
  // through separate queues. Defer the resume until loop() has drained the
  // runtime queue, otherwise Native can arm CSI against the old association
  // and immediately disable and rearm it a second time.
  wifi_reconfigure_resume_pending_ = true;
}

bool NativeFrontend::setup() {
  update_live_telemetry_enabled_();

  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    return false;
  }

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
  if (wifi_reconfigure_resume_pending_) {
    wifi_reconfigure_resume_pending_ = false;
    wifi_reconfigure_quiesced_ = false;
    if (!ota_frontend_quiesced_ && wifi_configured_()) {
      runtime_.set_services_armed(true);
    }
  }
  drain_pending_runtime_events_();
  if (mqtt_transport_ != nullptr) {
    mqtt_transport_->loop();
    drain_pending_ha_snapshot_();
  }
  if (direct_service_ != nullptr && direct_service_->running()) {
    raw_session_controller_.ensure_runtime_consistency();
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
  runtime_events_.clear();
  wifi_reconfigure_resume_pending_ = false;
  wifi_reconfigure_quiesced_ = false;
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
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION ||
      !snapshot.ready_to_publish) {
    return;
  }
  (void) runtime_events_.post_motion_state(snapshot);
}

void NativeFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) packets_received;
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION) return;
  if (!snapshot.ready_to_publish) {
    return;
  }
}

void NativeFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION) return;
  publish_mqtt_telemetry_(snapshot, now_ms_());
  if (snapshot.ready_to_publish) {
    publish_ha_threshold_(snapshot.threshold);
  }
}

void NativeFrontend::on_detector_changed(const RuntimeSnapshot &snapshot) {
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION) return;
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
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION) return;
  calibration_started_ = true;
  calibration_start_threshold_ = snapshot.threshold;
  publish_ha_calibrate_(true);
  if (!protocol_recalibration_command_active_) {
    publish_runtime_status_state_();
  }
}

void NativeFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION) return;
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
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION ||
      !runtime_.snapshot().ready_to_publish) {
    return;
  }
  RuntimeSnapshot snapshot = runtime_.snapshot();
  snapshot.movement_metric = movement;
  snapshot.threshold = threshold;
  runtime_events_.post_live_telemetry(snapshot);
}

void NativeFrontend::drain_pending_runtime_events_() {
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION) {
    runtime_events_.clear();
    return;
  }
  RuntimeSnapshot snapshot;
  while (runtime_events_.take_motion_state(snapshot)) {
    publish_ha_motion_(snapshot.motion_state);
    const uint32_t now = now_ms_();
    const std::string payload =
        espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U, "native");
    fan_out_payload_(nullptr, "telemetry", payload, false, true);
  }
  if (runtime_events_.take_live_telemetry(snapshot)) {
    const uint32_t now = now_ms_();
    const char *frontend = device_info_.frontend.empty() ? "native" : device_info_.frontend.c_str();
    const std::string payload =
        espectre_telemetry_payload(device_config_, snapshot, now, now / 1000U, frontend);
    fan_out_payload_("telemetry", "telemetry", payload, false, true);
    publish_ha_movement_(snapshot.movement_metric);
  }
}

void NativeFrontend::on_runtime_fault(const char *message) {
  publish_direct_event_("fault", espectre_fault_payload(device_config_, message, now_ms_()));
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
    result.code = frontend_command_parse_error_code(parse_error);
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
                                                         bool allow_local_config,
                                                         uint64_t connection_token,
                                                         std::string authorization) {
  if (runtime_.operation_state() == RuntimeOperationState::RAW_COLLECTION &&
      !frontend_command_allowed_during_raw_collection(command.command)) {
    FrontendCommandResult busy;
    busy.handled = true;
    busy.command = command;
    busy.code = "busy_raw_collection";
    busy.message = "mutation is unavailable during raw CSI collection";
    return busy;
  }
  const FrontendCommandCapabilities capabilities = command_capability_profile_(allow_local_config);
  FrontendCommandResult result = command_engine_.execute(
      command,
      FrontendCommandContext{origin, connection_token, std::move(authorization)},
      ota_service_,
      device_info_.firmware_version.c_str(),
      capabilities,
      [this, allow_local_config, capabilities](const EspectreCommand &read) {
        if (read.command == "capabilities") {
          return espectre_capabilities_payload(
              this->device_config_, this->mqtt_protocol_device_info_(), capabilities);
        }
        if (read.command == "info") {
          return espectre_info_payload(this->device_config_, this->mqtt_protocol_device_info_());
        }
        if (read.command == "status") {
          return this->direct_status_payload_(!this->device_info_.network.ip_address.empty());
        }
        if (read.command == "config") return this->direct_config_payload_(allow_local_config);
        if (read.command == "wifi_access_points" && allow_local_config) {
          return this->direct_wifi_access_points_payload_();
        }
        if (read.command == "diagnostics") {
          // Keep the generic engine responsible for capability and read-command
          // validation, then build this large payload after its stack frame has
          // unwound.
          return std::string{"{}"};
        }
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
      [this](const EspectreCommand &wifi_command, std::string *message) {
        if (wifi_command.command == "scan_wifi_access_points") {
          if (!this->wifi_scan_callback_) {
            if (message != nullptr) *message = "Wi-Fi scanning is unavailable";
            return false;
          }
          return this->wifi_scan_callback_(message);
        }
        if (wifi_command.command == "clear_wifi_config") {
          if (!this->provisioning_command_callback_) {
            if (message != nullptr) *message = "Wi-Fi configuration removal is unavailable";
            return false;
          }
          return this->provisioning_command_callback_("CLEAR_WIFI", message);
        }
        if (wifi_command.command == "clear_wifi_bssid") {
          if (!this->provisioning_command_callback_) {
            if (message != nullptr) *message = "Wi-Fi BSSID removal is unavailable";
            return false;
          }
          return this->provisioning_command_callback_("SET_WIFI_BSSID:bssid=", message);
        }
        if (!this->provisioning_command_callback_ || !wifi_command.has_wifi_bssid) {
          if (message != nullptr) {
            *message = "Wi-Fi BSSID selection is unavailable";
          }
          return false;
        }
        const std::string encoded =
            "SET_WIFI_BSSID:bssid=" + encode_urlencoded_component(wifi_command.wifi_bssid);
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
      },
      [this](const EspectreCommand &raw_command,
             const FrontendCommandContext &context,
             std::string *code,
             std::string *message,
             std::string *data_json) {
        return this->handle_raw_stream_command_(
            raw_command, context, code, message, data_json);
      });
  if (result.accepted && result.command.command == "diagnostics") {
    result.data_json = direct_diagnostics_payload_();
  }
  if (result.accepted) {
    if (command.command != "start_raw_stream" &&
        (static_cast<uint8_t>(result.changes) & static_cast<uint8_t>(FrontendCommandChange::STATUS)) != 0U) {
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

std::string NativeFrontend::handle_direct_request_(const DirectRequest &request,
                                                   uint64_t connection_token) {
  EspectreCommand command;
  std::string parse_error;
  if (!direct_http_request_to_command(request, &command, &parse_error)) {
    command.command_id = request.command_id;
    command.command = request.command;
    return espectre_command_result_payload(
        device_config_, command, false, frontend_command_parse_error_code(parse_error), parse_error.c_str());
  }

  const FrontendCommandResult result =
      dispatch_command_(command,
                        FrontendCommandOrigin::DIRECT,
                        true,
                        connection_token,
                        request.authorization);
  if (!result.accepted) {
    return espectre_command_result_payload(device_config_,
                                           result.command,
                                           false,
                                           result.code.c_str(),
                                           result.message.c_str(),
                                           result.data_json);
  }
  return espectre_command_result_payload(device_config_,
                                         result.command,
                                         true,
                                         result.code.c_str(),
                                         result.message.c_str(),
                                         result.data_json);
}

IDirectHttpService::DeferredRequestResult NativeFrontend::handle_deferred_direct_request_(
    uint64_t connection_token,
    const DirectRequest &request) {
  if (request.command != ESPECTRE_PEER_DISCOVERY_METHOD) {
    return {false, handle_direct_request_(request, connection_token)};
  }
  if (!peer_discovery_enabled_ || peer_discovery_ == nullptr) {
    EspectreCommand command;
    command.command_id = request.command_id;
    command.command = request.command;
    return {false, espectre_command_result_payload(device_config_, command, false, "unsupported",
                                                   "peer discovery is unavailable")};
  }
  std::vector<JsonObjectField> params;
  if (!parse_json_object_fields(request.params, &params) || !params.empty()) {
    EspectreCommand command;
    command.command_id = request.command_id;
    command.command = request.command;
    return {false, espectre_command_result_payload(device_config_, command, false, "invalid_params",
                                                   "discover_peers does not accept parameters")};
  }
  if (peer_discovery_->active()) {
    EspectreCommand command;
    command.command_id = request.command_id;
    command.command = request.command;
    return {false, espectre_command_result_payload(device_config_, command, false, "conflict",
                                                   "a peer discovery request is already active")};
  }
  const bool started = peer_discovery_->start(
      [this, connection_token, request_id = request.command_id, command_name = request.command](PeerDiscoverySnapshot snapshot) {
        if (this->direct_service_ == nullptr) {
          return;
        }
        EspectreCommand command;
        command.command_id = request_id;
        command.command = command_name;
        (void) this->direct_service_->complete_deferred_response(
            connection_token,
            espectre_command_result_payload(this->device_config_, command, true, "ok",
                                            "peer discovery completed",
                                            peer_discovery_snapshot_json(snapshot)));
      });
  if (!started) {
    EspectreCommand command;
    command.command_id = request.command_id;
    command.command = request.command;
    return {false, espectre_command_result_payload(device_config_, command, false, "unavailable",
                                                   "peer discovery could not be started")};
  }
  return {true, {}};
}

EspectreCapabilityProfile NativeFrontend::command_capability_profile_(bool allow_local_config) const {
  EspectreCapabilityProfile profile;
  using Method = EspectreDirectMethod;
  profile.set(Method::CAPABILITIES);
  profile.set(Method::INFO);
  profile.set(Method::STATUS);
  profile.set(Method::CONFIG);
  profile.set(Method::DIAGNOSTICS);
  profile.set(Method::SET_SENSING);
  profile.set(Method::SET_DEVICE_LABEL);
  profile.set(Method::SET_THRESHOLD, runtime_.capabilities().supports_runtime_threshold_updates);
  profile.set(Method::SET_MOTION_HITS, runtime_.capabilities().supports_runtime_motion_hits_updates);
  profile.set(Method::SET_DETECTOR, runtime_.capabilities().supports_runtime_detector_selection);
  profile.set(Method::RECALIBRATE, runtime_.capabilities().supports_manual_recalibration);
  profile.set(Method::SET_CSI_TRAFFIC_MODE, runtime_.capabilities().supports_traffic_control);
  profile.set(Method::SET_TRAFFIC_GENERATOR_MODE, runtime_.capabilities().supports_traffic_control);
  profile.set(Method::WIFI_ACCESS_POINTS, allow_local_config);
  profile.set(Method::SCAN_WIFI_ACCESS_POINTS, allow_local_config);
  profile.set(Method::SET_WIFI_BSSID, allow_local_config);
  profile.set(Method::CLEAR_WIFI_BSSID, allow_local_config);
  profile.set(Method::CLEAR_WIFI_CONFIG, allow_local_config);
  profile.set(Method::SET_MQTT_CONFIG, allow_local_config);
  profile.set(Method::CLEAR_MQTT_CONFIG, allow_local_config);
  profile.set(Method::OTA_STATUS, ota_service_ != nullptr);
  profile.set(Method::OTA_CHECK, ota_service_ != nullptr);
  profile.set(Method::OTA_START, ota_service_ != nullptr);
  profile.set(Method::DISCOVER_PEERS, allow_local_config && peer_discovery_enabled_);
  const bool raw_csi = allow_local_config && direct_session_tokens_enabled_ &&
                       runtime_.capabilities().supports_raw_csi;
  profile.set(Method::START_RAW_STREAM, raw_csi);
  profile.set(Method::STOP_RAW_STREAM, raw_csi);
  profile.set(EspectreConfigSection::RUNTIME);
  profile.set(EspectreConfigSection::DEVICE);
  profile.set(EspectreConfigSection::WIFI, allow_local_config);
  profile.set(EspectreConfigSection::MQTT, allow_local_config);
  return profile;
}

std::string NativeFrontend::direct_capabilities_payload_() const {
  return espectre_capabilities_payload(
      device_config_, mqtt_protocol_device_info_(), command_capability_profile_(true));
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
  append_json_pair(&out,
                   "operation_state",
                   runtime_operation_state_name(runtime_.operation_state()));
  const RawCsiSessionDiagnostics raw =
      direct_service_ != nullptr ? direct_service_->raw_diagnostics()
                                 : RawCsiSessionDiagnostics{};
  out += ",\"raw_session\":{\"active\":";
  out += raw.active ? "true" : "false";
  out += ",\"binary_bound\":";
  out += raw.binary_bound ? "true" : "false";
  out += ",\"authorized\":";
  out += raw_session_controller_.active() ? "true" : "false";
  out += ",\"fresh_records\":" + std::to_string(raw.fresh_record_total) + "}";
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
  out += ",\"csi_target_pps\":" + std::to_string(runtime_config.csi_target_pps);
  out += ",\"csi_traffic_udp_port\":" + std::to_string(runtime_config.csi_traffic_udp_port);
  append_json_pair(&out, "csi_traffic_multicast_group", runtime_config.csi_traffic_multicast_group.c_str());
  out += "}";
  if (!include_local) {
    out += "}";
    return out;
  }
  out += ",\"device\":{";
  append_json_pair(&out, "device_label", device_config_.device_label.c_str(), true);
  DirectWifiSnapshot wifi = read_direct_wifi_snapshot();
  wifi.configured = wifi.configured || wifi_configured_();
  wifi.connected = wifi.connected || !device_info_.network.ip_address.empty();
  if (wifi.ssid.empty()) wifi.ssid = wifi_info_.ssid;
  if (wifi.bssid.empty()) wifi.bssid = wifi_info_.bssid;
  if (wifi.channel == 0U) {
    wifi.channel = device_info_.network.channel != 0U
                       ? device_info_.network.channel
                       : wifi_info_.channel;
  }
  if (wifi.band.empty() && wifi.channel > 0U) {
    wifi.band = wifi.channel <= WIFI_CHANNEL_2G_MAX ? "2g" : "5g";
  }
  out += "},\"wifi\":{\"configured\":";
  out += wifi.configured ? "true" : "false";
  out += ",\"connected\":";
  out += wifi.connected ? "true" : "false";
  append_json_pair(&out, "ssid", wifi.ssid.c_str());
  append_json_pair(&out, "bssid", wifi.bssid.c_str());
  append_json_pair(&out, "band", wifi.band.c_str());
  out += ",\"channel\":" + std::to_string(static_cast<unsigned>(wifi.channel));
  out += ",\"rssi_dbm\":";
  out += wifi.rssi_dbm == INT16_MIN ? "null" : std::to_string(wifi.rssi_dbm);
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

std::string NativeFrontend::direct_wifi_access_points_payload_() const {
  std::string out{"{\"scanning\":"};
  out += wifi_info_.scan_pending ? "true" : "false";
  append_json_pair(&out, "message", wifi_info_.scan_message.c_str());
  out += ",\"access_points\":[";
  bool first = true;
  for (const WifiProvisioningInfo::AccessPoint &access_point : wifi_info_.access_points) {
    if (!first) out += ",";
    first = false;
    out += "{";
    append_json_pair(&out, "bssid", access_point.bssid.c_str(), true);
    out += ",\"rssi_dbm\":" + std::to_string(static_cast<int>(access_point.rssi_dbm));
    out += ",\"channel\":" + std::to_string(static_cast<unsigned>(access_point.channel));
    out += "}";
  }
  out += "]}";
  return out;
}

std::string NativeFrontend::direct_diagnostics_payload_() const {
  const uint32_t now = now_ms_();
  std::string out = espectre_diagnostics_payload(device_config_,
                                                 runtime_.snapshot(),
                                                 now,
                                                 now / 1000U,
                                                 current_free_memory_kb(),
                                                 last_loop_time_ms_,
                                                 runtime_.diagnostics_sample());
  if (!out.empty() && out.back() == '}') {
    out.pop_back();
  }
  {
    const RuntimeDiagnosticsSnapshot runtime_diagnostics = runtime_.diagnostics();
    out += ",\"csi_classified_total\":" +
           std::to_string(runtime_diagnostics.csi_classified_total);
    out += ",\"csi_provenance_rejected_total\":" +
           std::to_string(runtime_diagnostics.csi_provenance_rejected_total);
    out += ",\"csi_pending_frame_drops_total\":" +
           std::to_string(runtime_diagnostics.csi_pending_frame_drops_total);
    out += ",\"csi_pending_frames\":" +
           std::to_string(runtime_diagnostics.csi_pending_frames);
    out += ",\"csi_pending_frame_capacity\":" +
           std::to_string(runtime_diagnostics.csi_pending_frame_capacity);
    out += ",\"runtime_motion_event_drops_total\":" +
           std::to_string(runtime_events_.motion_state_drops_total());
    append_runtime_performance_diagnostics_json(&out, runtime_diagnostics, false);
  }
  out += ",\"task_stack_high_water_bytes\":" +
         std::to_string(current_task_stack_high_water_bytes());
  {
    const DirectHttpServiceDiagnostics direct =
        direct_service_ != nullptr ? direct_service_->diagnostics() : DirectHttpServiceDiagnostics{};
    out += ",\"direct_http\":{\"event_clients\":" + std::to_string(direct_client_count_);
    out += ",\"event_client_limit\":" + std::to_string(direct.event_client_limit);
    out += ",\"queue_capacity\":" + std::to_string(direct.queue_capacity);
    out += ",\"queued_messages\":" + std::to_string(direct.queued_messages);
    out += ",\"accepted_connections\":" + std::to_string(direct.accepted_connections);
    out += ",\"rejected_connections\":" + std::to_string(direct.rejected_connections);
    out += ",\"malformed_requests\":" + std::to_string(direct.malformed_requests);
    out += ",\"oversized_requests\":" + std::to_string(direct.oversized_requests);
    out += ",\"rate_limited_requests\":" + std::to_string(direct.rate_limited_requests);
    out += ",\"dropped_telemetry_events\":" + std::to_string(direct.dropped_telemetry_events);
    out += ",\"send_failures\":" + std::to_string(direct.send_failures);
    out += ",\"slow_client_disconnects\":" +
           std::to_string(direct.slow_client_disconnects) + "}";
  }
  {
    const RawCsiSessionDiagnostics raw =
        direct_service_ != nullptr ? direct_service_->raw_diagnostics()
                                   : RawCsiSessionDiagnostics{};
    out += ",\"raw_csi\":{\"active\":";
    out += raw.active ? "true" : "false";
    out += ",\"binary_bound\":";
    out += raw.binary_bound ? "true" : "false";
    out += ",\"raw_drop_total\":" + std::to_string(raw.raw_drop_total);
    out += ",\"send_backpressure_total\":" +
           std::to_string(raw.raw_send_backpressure_total);
    out += ",\"fresh_record_total\":" + std::to_string(raw.fresh_record_total);
    out += ",\"stream_sequence\":" + std::to_string(raw.stream_sequence) + "}";
  }
  {
    const MqttTransportDiagnostics mqtt =
        mqtt_transport_ != nullptr ? mqtt_transport_->diagnostics() : MqttTransportDiagnostics{};
    out += ",\"mqtt\":{\"connected\":";
    out += mqtt_transport_ != nullptr && mqtt_transport_->connected() ? "true" : "false";
    out += ",\"queue_capacity\":" + std::to_string(mqtt.queue_capacity);
    out += ",\"outbox_capacity_bytes\":" + std::to_string(mqtt.outbox_capacity_bytes);
    out += ",\"queued_publishes\":" + std::to_string(mqtt.queued_publishes);
    out += ",\"dropped_publishes\":" + std::to_string(mqtt.dropped_publishes);
    out += ",\"publish_failures\":" + std::to_string(mqtt.publish_failures);
    out += ",\"reconnects\":" + std::to_string(mqtt.reconnects) + "}}";
  }
  return out;
}

void NativeFrontend::refresh_peer_candidate_() {
  if (peer_discovery_ == nullptr) return;
  const std::string device_id = espectre_effective_device_id(device_config_);
  const std::string generated_name = espectre_device_name(
      espectre_effective_device_id_u64(device_config_),
      device_info_.chip.empty() ? nullptr : device_info_.chip.c_str());
  const std::string display_name = device_config_.device_label.empty()
                                       ? generated_name
                                       : device_config_.device_label;
  PeerDiscoveryCandidate candidate;
  candidate.instance = device_config_.device_label.empty()
                           ? "ESPectre " + device_id
                           : device_config_.device_label + " " + device_id;
  candidate.hostname = "espectre-" + device_id;
  candidate.device_id = device_id;
  candidate.name = display_name;
  candidate.frontend = "native";
  candidate.txt_version = ESPECTRE_DNS_SD_TXT_SCHEMA_VERSION;
  candidate.protocol_version = ESPECTRE_PROTOCOL_VERSION;
  candidate.transport = ESPECTRE_DIRECT_HTTP_TRANSPORT;
  candidate.path = ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT;
  candidate.events = ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT;
  candidate.firmware = device_info_.firmware_version;
  candidate.chip = device_info_.chip;
  candidate.capabilities = "config,monitor,raw_csi";
  candidate.port = ESPECTRE_DIRECT_HTTP_PORT;
  peer_discovery_->set_local_candidate(std::move(candidate));
}

bool NativeFrontend::handle_raw_stream_command_(const EspectreCommand &command,
                                                const FrontendCommandContext &context,
                                                std::string *code,
                                                std::string *message,
                                                std::string *data_json) {
  uint64_t device_id = 0U;
  if (!parse_espectre_device_id(espectre_effective_device_id(device_config_), &device_id)) {
    if (code != nullptr) *code = "internal_error";
    if (message != nullptr) *message = "device identity is unavailable";
    return false;
  }
  raw_session_controller_.configure(
      direct_service_, &runtime_, device_id, device_info_.chip,
      [this](RawCsiStopReason) {
        this->runtime_events_.clear();
        this->pending_ha_state_ = false;
      });
  const bool accepted = raw_session_controller_.handle_command(
      command, context, code, message, data_json);
  if (accepted && command.command == "start_raw_stream") {
    runtime_events_.clear();
    pending_ha_state_ = false;
  }
  return accepted;
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
    ESP_LOGW(TAG, "CSI traffic mode is not selectable");
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
      mode != RUNTIME_CSI_TRAFFIC_MODE_EXTERNAL_NAME) {
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
      mode != RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME &&
      mode != RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_TCP_NAME) {
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
  direct_session_tokens_enabled_ = false;

  DirectHttpServiceConfig config = DirectHttpServiceConfig::for_first_party_portals();
  config.device_id = device_config_.device_id;
#if defined(CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED) && CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED
  config.allow_http_loopback_origins = true;
#endif
  const auto client_count_changed = [this](size_t client_count) {
    this->direct_client_count_ = client_count;
    this->update_live_telemetry_enabled_();
  };
  bool setup = direct_service_->setup_deferred(
      config,
      [this](uint64_t token, const DirectRequest &request) {
        return this->handle_deferred_direct_request_(token, request);
      },
      client_count_changed);
  direct_session_tokens_enabled_ = setup;
  peer_discovery_enabled_ = setup && peer_discovery_ != nullptr;
  if (!setup) {
    setup = direct_service_->setup(
        config,
        [this](const DirectRequest &request) {
          return this->handle_direct_request_(request, 0U);
        },
        client_count_changed);
  }
  if (!setup) {
    ESP_LOGE(TAG, "Direct HTTP service setup failed");
  }
}

void NativeFrontend::stop_direct_service_() {
  if (direct_service_ != nullptr && direct_service_->running()) {
    direct_service_->shutdown();
  }
  direct_client_count_ = 0U;
  peer_discovery_enabled_ = false;
  direct_session_tokens_enabled_ = false;
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
  const RuntimeDiagnosticsSample *sample = runtime_.diagnostics_sample();
  if (sample == nullptr) {
    return;
  }
  for (const FrontendHaDiagnosticSensor &sensor : ha_settings_.diagnostic_sensors) {
    const std::string payload = diagnostic_state_payload(sensor.key, *sample);
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
      espectre_capabilities_payload(device_config_, info, command_capability_profile_(false)),
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
  info.csi_traffic_udp_port = runtime_.config().csi_traffic_udp_port;
  info.csi_traffic_multicast_group = runtime_.config().csi_traffic_multicast_group;
  info.evaluation_interval_ms = runtime_.config().evaluation_interval_ms;
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
