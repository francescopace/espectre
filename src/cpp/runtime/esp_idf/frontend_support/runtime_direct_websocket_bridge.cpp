/*
 * ESPectre - Runtime Direct WebSocket Bridge
 *
 * Shared Direct WebSocket control surface for firmware frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_direct_websocket_bridge.h"

#include "direct_websocket_protocol.h"
#include "espectre_protocol.h"
#include "protocol_json.h"
#include "runtime_config_utils.h"

namespace espectre {

namespace {

void append_bool(std::string *out, const char *key, bool value, bool first = false) {
  if (out == nullptr || key == nullptr) {
    return;
  }
  *out += first ? "\"" : ",\"";
  *out += key;
  *out += "\":";
  *out += value ? "true" : "false";
}

void append_uint(std::string *out, const char *key, uint64_t value, bool first = false) {
  if (out == nullptr || key == nullptr) {
    return;
  }
  *out += first ? "\"" : ",\"";
  *out += key;
  *out += "\":" + std::to_string(value);
}

}  // namespace

bool RuntimeDirectWebSocketBridge::setup(IDirectWebSocketService *service,
                                         RuntimeFrontendController *runtime,
                                         const RuntimeDirectWebSocketBridgeConfig &config,
                                         ConfigChangedCallback config_changed) {
  shutdown();
  if (service == nullptr || runtime == nullptr || !runtime->is_setup_complete() || config.frontend.empty() ||
      config.device_id == 0U || config.port == 0U) {
    return false;
  }
  service_ = service;
  runtime_ = runtime;
  config_ = config;
  config_changed_ = std::move(config_changed);

  DirectWebSocketServiceConfig service_config;
  service_config.port = config.port;
  service_config.allowed_origins = {
      "https://espectre.dev",
      "https://www.espectre.dev",
      "https://test.espectre.dev",
  };
  service_config.allow_missing_origin = config.allow_missing_origin;
#if defined(CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED) && CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED
  service_config.allow_http_loopback_origins = true;
#endif
  if (!service_->setup(
          service_config,
          [this](const DirectWebSocketRequest &request) { return this->handle_request_(request); },
          [](size_t client_count) { (void) client_count; })) {
    service_ = nullptr;
    runtime_ = nullptr;
    config_changed_ = {};
    return false;
  }
  return true;
}

void RuntimeDirectWebSocketBridge::loop() {
  if (service_ != nullptr && service_->running()) {
    service_->loop();
  }
}

void RuntimeDirectWebSocketBridge::shutdown() {
  if (service_ != nullptr && service_->running()) {
    service_->shutdown();
  }
  service_ = nullptr;
  runtime_ = nullptr;
  config_changed_ = {};
}

bool RuntimeDirectWebSocketBridge::running() const { return service_ != nullptr && service_->running(); }

size_t RuntimeDirectWebSocketBridge::client_count() const {
  return service_ != nullptr ? service_->client_count() : 0U;
}

bool RuntimeDirectWebSocketBridge::publish_event(const char *event_name,
                                                 const std::string &data_json,
                                                 bool replaceable_telemetry) {
  return event_name != nullptr && running() && service_->publish_event(event_name, data_json, replaceable_telemetry);
}

std::string RuntimeDirectWebSocketBridge::handle_request_(const DirectWebSocketRequest &request) {
  EspectreCommand command;
  std::string parse_error;
  if (!direct_websocket_request_to_command(request, &command, &parse_error)) {
    return direct_websocket_error_response(request.id, "invalid_params", parse_error.c_str());
  }
  if (command.command == "capabilities" || command.command == "commands") {
    return direct_websocket_success_response(request.id, capabilities_payload_());
  }
  if (command.command == "info") {
    return direct_websocket_success_response(request.id, info_payload_());
  }
  if (command.command == "status") {
    return direct_websocket_success_response(request.id, status_payload_());
  }
  if (command.command == "config") {
    return direct_websocket_success_response(request.id, config_payload_());
  }
  if (command.command == "diagnostics" || command.command == "stats") {
    return direct_websocket_success_response(request.id, diagnostics_payload_());
  }

  bool accepted = false;
  const char *message = "unsupported method";
  if (command.command == "set_threshold" && command.has_threshold) {
    accepted = runtime_->set_threshold_runtime(command.threshold);
    message = accepted ? "threshold updated" : "threshold rejected";
  } else if (command.command == "set_motion_hits" && command.has_motion_hits) {
    accepted = runtime_->set_motion_hits_runtime(command.motion_on_hits, command.motion_off_hits);
    message = accepted ? "motion hits updated" : "motion hits rejected";
  } else if (command.command == "set_detector" && command.has_detector) {
    accepted = runtime_->set_detection_algorithm_runtime(parse_detection_algorithm(command.detector.c_str()));
    message = accepted ? "detector updated" : "detector rejected";
  } else if (command.command == "set_csi_traffic_mode" && command.has_csi_traffic_mode) {
    accepted = runtime_->set_csi_traffic_mode_runtime(parse_csi_traffic_mode(command.csi_traffic_mode.c_str()));
    message = accepted ? "CSI traffic mode updated" : "CSI traffic mode rejected";
  } else if (command.command == "set_traffic_generator_mode" && command.has_traffic_generator_mode) {
    accepted = runtime_->set_traffic_generator_mode_runtime(
        parse_traffic_mode(command.traffic_generator_mode.c_str()));
    message = accepted ? "traffic generator mode updated" : "traffic generator mode rejected";
  } else if (command.command == "recalibrate") {
    accepted = runtime_->trigger_recalibration();
    message = accepted ? "recalibration started" : "recalibration rejected";
  } else if (command.command == "start_sensing" || command.command == "stop_sensing") {
    runtime_->set_services_armed(command.command == "start_sensing");
    accepted = true;
    message = command.command == "start_sensing" ? "sensing started" : "sensing stopped";
  }

  if (accepted) {
    notify_config_changed_();
  }
  return mutation_result_(request, accepted, message);
}

std::string RuntimeDirectWebSocketBridge::capabilities_payload_() const {
  const RuntimeCapabilities &capabilities = runtime_->capabilities();
  std::string out{"{\"protocol_version\":1"};
  append_json_pair(&out, "subprotocol", ESPECTRE_DIRECT_WEBSOCKET_SUBPROTOCOL);
  out += ",\"methods\":[\"capabilities\",\"commands\",\"info\",\"status\",\"config\",\"diagnostics\",\"stats\",\"start_sensing\",\"stop_sensing\"";
  if (capabilities.supports_runtime_threshold_updates) out += ",\"set_threshold\"";
  if (capabilities.supports_runtime_motion_hits_updates) out += ",\"set_motion_hits\"";
  if (capabilities.supports_runtime_detector_selection) out += ",\"set_detector\"";
  if (capabilities.supports_manual_recalibration) out += ",\"recalibrate\"";
  if (capabilities.supports_traffic_control) {
    out += ",\"set_csi_traffic_mode\",\"set_traffic_generator_mode\"";
  }
  out += "],\"events\":[\"status\",\"telemetry\",\"config\",\"fault\"]";
  append_bool(&out, "raw_csi", config_.raw_csi);
  out += "}";
  return out;
}

std::string RuntimeDirectWebSocketBridge::info_payload_() const {
  std::string out{"{"};
  append_json_pair(&out, "device_id", format_espectre_device_id(config_.device_id).c_str(), true);
  append_json_pair(&out, "name", config_.device_name.c_str());
  append_json_pair(&out, "frontend", config_.frontend.c_str());
  append_json_pair(&out, "firmware", config_.firmware_version.c_str());
  append_json_pair(&out, "chip", config_.chip.c_str());
  append_json_pair(&out, "path", ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT);
  append_bool(&out, "raw_csi", config_.raw_csi);
  out += "}";
  return out;
}

std::string RuntimeDirectWebSocketBridge::status_payload_() const {
  const RuntimeSnapshot &snapshot = runtime_->snapshot();
  std::string out{"{"};
  append_bool(&out, "sensing_enabled", runtime_->services_armed(), true);
  append_bool(&out, "ready_to_publish", snapshot.ready_to_publish);
  append_bool(&out, "calibrating", runtime_->is_calibrating());
  append_bool(&out, "motion", snapshot.motion_state == MotionState::MOTION);
  out += ",\"movement\":" + std::to_string(snapshot.movement_metric);
  out += ",\"threshold\":" + std::to_string(snapshot.threshold);
  append_json_pair(&out, "detector", snapshot.detector_name != nullptr ? snapshot.detector_name : "");
  out += "}";
  return out;
}

std::string RuntimeDirectWebSocketBridge::config_payload_() const {
  const RuntimeConfig &config = runtime_->config();
  std::string out{"{"};
  out += "\"threshold\":" + std::to_string(runtime_->snapshot().threshold);
  append_json_pair(&out, "detector", detection_algorithm_name(config.detection_algorithm));
  append_uint(&out, "motion_on_hits", config.motion_on_hits);
  append_uint(&out, "motion_off_hits", config.motion_off_hits);
  append_json_pair(&out, "csi_traffic_mode", csi_traffic_mode_name(config.csi_traffic_mode));
  append_json_pair(&out, "traffic_generator_mode", traffic_mode_name(config.traffic_generator_mode));
  append_uint(&out, "csi_target_pps", config.csi_target_pps);
  out += "}";
  return out;
}

std::string RuntimeDirectWebSocketBridge::diagnostics_payload_() const {
  const RuntimeDiagnosticsSnapshot diagnostics = runtime_->diagnostics();
  std::string out{"{"};
  append_uint(&out, "traffic_packets_total", diagnostics.traffic_packets_total, true);
  append_uint(&out, "csi_callbacks_total", diagnostics.csi_callbacks_total);
  append_uint(&out, "csi_accepted_total", diagnostics.csi_accepted_total);
  append_uint(&out, "csi_admitted_total", diagnostics.csi_admitted_total);
  append_uint(&out, "csi_filtered_total", diagnostics.csi_filtered_total);
  append_uint(&out, "csi_missing_slots_total", diagnostics.csi_missing_slots_total);
  append_uint(&out, "csi_excess_total", diagnostics.csi_excess_total);
  append_uint(&out, "csi_stale_total", diagnostics.csi_stale_total);
  append_uint(&out, "csi_out_of_order_total", diagnostics.csi_out_of_order_total);
  append_uint(&out, "csi_occupancy_slots", diagnostics.csi_occupancy_slots);
  append_uint(&out, "csi_window_slots", diagnostics.csi_window_slots);
  out += ",\"wifi_rssi_dbm\":" + std::to_string(static_cast<int>(diagnostics.wifi_rssi_dbm));
  append_uint(&out, "wifi_channel", diagnostics.wifi_channel);
  if (service_ != nullptr) {
    const DirectWebSocketServiceDiagnostics direct = service_->diagnostics();
    append_uint(&out, "direct_clients", service_->client_count());
    append_uint(&out, "direct_rejected_connections", direct.rejected_connections);
    append_uint(&out, "direct_dropped_telemetry_events", direct.dropped_telemetry_events);
  }
  out += "}";
  return out;
}

std::string RuntimeDirectWebSocketBridge::mutation_result_(const DirectWebSocketRequest &request,
                                                           bool accepted,
                                                           const char *message) {
  if (!accepted) {
    const char *code = std::string(message) == "unsupported method" ? "unsupported_method" : "rejected";
    return direct_websocket_error_response(request.id, code, message);
  }
  std::string result{"{"};
  append_json_pair(&result, "message", message, true);
  result += "}";
  return direct_websocket_success_response(request.id, result);
}

void RuntimeDirectWebSocketBridge::notify_config_changed_() {
  if (config_changed_) {
    config_changed_();
  }
}

}  // namespace espectre
