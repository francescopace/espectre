/*
 * ESPectre - Runtime Direct HTTP Bridge
 *
 * Shared Direct HTTP control surface for firmware frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_direct_http_bridge.h"

#include "direct_http_protocol.h"
#include "espectre_protocol.h"
#include "protocol_json.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "runtime_time.h"

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

bool RuntimeDirectHttpBridge::setup(IDirectHttpService *service,
                                         RuntimeFrontendController *runtime,
                                         const RuntimeDirectHttpBridgeConfig &config,
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

  DirectHttpServiceConfig service_config = DirectHttpServiceConfig::for_first_party_portals();
  service_config.port = config.port;
  service_config.allow_missing_origin = config.allow_missing_origin;
#if defined(CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED) && CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED
  service_config.allow_http_loopback_origins = true;
#endif
  if (!service_->setup(
          service_config,
          [this](const DirectRequest &request) { return this->handle_request_(request); },
          [](size_t client_count) { (void) client_count; })) {
    service_ = nullptr;
    runtime_ = nullptr;
    config_changed_ = {};
    return false;
  }
  return true;
}

void RuntimeDirectHttpBridge::loop() {
  if (service_ != nullptr && service_->running()) {
    service_->loop();
  }
}

void RuntimeDirectHttpBridge::shutdown() {
  if (service_ != nullptr && service_->running()) {
    service_->shutdown();
  }
  service_ = nullptr;
  runtime_ = nullptr;
  config_changed_ = {};
}

bool RuntimeDirectHttpBridge::running() const { return service_ != nullptr && service_->running(); }

size_t RuntimeDirectHttpBridge::event_client_count() const {
  return service_ != nullptr ? service_->event_client_count() : 0U;
}

bool RuntimeDirectHttpBridge::publish_event(const char *event_name,
                                                 const std::string &data_json,
                                                 bool replaceable_telemetry) {
  return event_name != nullptr && running() && service_->publish_event(event_name, data_json, replaceable_telemetry);
}

bool RuntimeDirectHttpBridge::publish_telemetry(const RuntimeSnapshot &snapshot) {
  EspectreDeviceConfig device;
  device.device_id = config_.device_id;
  return publish_event("telemetry",
                       espectre_telemetry_payload(device,
                                                  snapshot,
                                                  monotonic_now_ms(),
                                                  monotonic_now_ms() / 1000U,
                                                  config_.frontend.c_str()),
                       true);
}

bool RuntimeDirectHttpBridge::publish_changes(FrontendCommandChange changes) {
  if (!running() || runtime_ == nullptr) {
    return false;
  }
  bool published = true;
  const uint8_t flags = static_cast<uint8_t>(changes);
  if ((flags & static_cast<uint8_t>(FrontendCommandChange::STATUS)) != 0U) {
    published = publish_event("status", status_payload_()) && published;
  }
  if ((flags & static_cast<uint8_t>(FrontendCommandChange::INFO)) != 0U) {
    published = publish_event("info", info_payload_()) && published;
  }
  if ((flags & static_cast<uint8_t>(FrontendCommandChange::CONFIG)) != 0U) {
    published = publish_event("config", config_payload_()) && published;
  }
  return published;
}

std::string RuntimeDirectHttpBridge::handle_request_(const DirectRequest &request) {
  EspectreCommand command;
  std::string parse_error;
  if (!direct_http_request_to_command(request, &command, &parse_error)) {
    return direct_http_error_response(request.id, "invalid_params", parse_error.c_str());
  }
  const RuntimeCapabilities &runtime_capabilities = runtime_->capabilities();
  const FrontendCommandCapabilities capabilities{
      true,
      true,
      true,
      runtime_capabilities.supports_extended_diagnostics,
      false,
      false,
      false,
      true,
      runtime_capabilities.supports_runtime_threshold_updates,
      runtime_capabilities.supports_runtime_motion_hits_updates,
      runtime_capabilities.supports_traffic_control,
      runtime_capabilities.supports_runtime_detector_selection,
      runtime_capabilities.supports_manual_recalibration,
      false,
      false,
  };
  FrontendCommandResult result = command_engine_.execute(
      command,
      FrontendCommandContext{FrontendCommandOrigin::DIRECT, 0U, request.authorization},
      nullptr,
      config_.firmware_version.c_str(),
      capabilities,
      [this](const EspectreCommand &read) {
        if (read.command == "capabilities") return capabilities_payload_();
        if (read.command == "info") return info_payload_();
        if (read.command == "status") return status_payload_();
        if (read.command == "config") return config_payload_();
        if (read.command == "diagnostics") return diagnostics_payload_();
        return std::string{};
      },
      {},
      [this](float value, std::string *) { return runtime_->set_threshold_runtime(value); },
      [this](uint8_t on, uint8_t off, std::string *) { return runtime_->set_motion_hits_runtime(on, off); },
      [this](CsiTrafficMode mode, std::string *) { return runtime_->set_csi_traffic_mode_runtime(mode); },
      [this](RuntimeTrafficMode mode, std::string *) { return runtime_->set_traffic_generator_mode_runtime(mode); },
      [this](DetectionAlgorithm algorithm, std::string *) {
        return runtime_->set_detection_algorithm_runtime(algorithm);
      },
      [this](std::string *) { return runtime_->trigger_recalibration(); },
      {},
      {},
      [this](bool enabled, std::string *) {
        runtime_->set_services_armed(enabled);
        return true;
      });
  if (!result.accepted) {
    return direct_http_error_response(request.id, result.code.c_str(), result.message.c_str());
  }
  if (result.changes != FrontendCommandChange::NONE) {
    (void) publish_changes(result.changes);
    notify_config_changed_();
  }
  std::string response{"{"};
  append_json_pair(&response, "command", command.command.c_str(), true);
  append_json_pair(&response, "code", result.code.c_str());
  append_json_pair(&response, "message", result.message.c_str());
  if (!result.data_json.empty()) response += ",\"data\":" + result.data_json;
  response += "}";
  return direct_http_success_response(request.id, response);
}

std::string RuntimeDirectHttpBridge::capabilities_payload_() const {
  const RuntimeCapabilities &capabilities = runtime_->capabilities();
  EspectreDeviceConfig device;
  device.device_id = config_.device_id;
  EspectreDeviceInfo info;
  info.frontend = config_.frontend;
  info.firmware_version = config_.firmware_version;
  info.chip = config_.chip;
  info.supports_info = true;
  info.supports_diagnostics = capabilities.supports_extended_diagnostics;
  info.supports_runtime_threshold = capabilities.supports_runtime_threshold_updates;
  info.supports_runtime_motion_hits = capabilities.supports_runtime_motion_hits_updates;
  info.supports_runtime_detector = capabilities.supports_runtime_detector_selection;
  info.supports_manual_recalibration = capabilities.supports_manual_recalibration;
  info.supports_traffic_control = capabilities.supports_traffic_control;
  return espectre_capabilities_payload(device, info, true, true, true);
}

std::string RuntimeDirectHttpBridge::info_payload_() const {
  EspectreDeviceConfig device;
  device.device_id = config_.device_id;
  device.device_label = config_.device_name;
  EspectreDeviceInfo info;
  info.frontend = config_.frontend;
  info.firmware_version = config_.firmware_version;
  info.chip = config_.chip;
  info.evaluation_interval_ms = runtime_->config().evaluation_interval_ms;
  info.publish_interval_ms = runtime_->config().publish_interval_ms;
  return espectre_info_payload(device, info);
}

std::string RuntimeDirectHttpBridge::status_payload_() const {
  const RuntimeSnapshot &snapshot = runtime_->snapshot();
  EspectreDeviceConfig device;
  device.device_id = config_.device_id;
  std::string out = espectre_status_payload(device, true, monotonic_now_ms());
  out.pop_back();
  append_bool(&out, "sensing_enabled", runtime_->services_armed());
  append_bool(&out, "ready_to_publish", snapshot.ready_to_publish);
  append_bool(&out, "calibrating", runtime_->is_calibrating());
  out += "}";
  return out;
}

std::string RuntimeDirectHttpBridge::config_payload_() const {
  const RuntimeConfig &config = runtime_->config();
  std::string out{"{"};
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", format_espectre_device_id(config_.device_id).c_str());
  out += ",\"runtime\":{";
  out += "\"threshold\":" + std::to_string(runtime_->snapshot().threshold);
  append_json_pair(&out, "detector", detection_algorithm_name(config.detection_algorithm));
  append_uint(&out, "motion_on_hits", config.motion_on_hits);
  append_uint(&out, "motion_off_hits", config.motion_off_hits);
  append_json_pair(&out, "csi_traffic_mode", csi_traffic_mode_name(config.csi_traffic_mode));
  append_json_pair(&out, "traffic_generator_mode", traffic_mode_name(config.traffic_generator_mode));
  append_uint(&out, "csi_target_pps", config.csi_target_pps);
  out += "}}";
  return out;
}

std::string RuntimeDirectHttpBridge::diagnostics_payload_() const {
  const RuntimeDiagnosticsSnapshot diagnostics = runtime_->diagnostics();
  const uint32_t now_ms = monotonic_now_ms();
  std::string out{"{"};
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", format_espectre_device_id(config_.device_id).c_str());
  append_uint(&out, "timestamp_ms", now_ms);
  append_uint(&out, "uptime", now_ms / 1000U);
  append_uint(&out, "traffic_packets_total", diagnostics.traffic_packets_total);
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
  append_runtime_performance_diagnostics_json(&out, diagnostics);
  if (service_ != nullptr) {
    const DirectHttpServiceDiagnostics direct = service_->diagnostics();
    append_uint(&out, "direct_event_clients", service_->event_client_count());
    append_uint(&out, "direct_rejected_connections", direct.rejected_connections);
    append_uint(&out, "direct_dropped_telemetry_events", direct.dropped_telemetry_events);
    out += ",\"direct_http\":{";
    append_uint(&out, "event_clients", service_->event_client_count(), true);
    append_uint(&out, "event_client_limit", direct.event_client_limit);
    append_uint(&out, "queue_capacity", direct.queue_capacity);
    append_uint(&out, "queued_messages", direct.queued_messages);
    append_uint(&out, "accepted_connections", direct.accepted_connections);
    append_uint(&out, "rejected_connections", direct.rejected_connections);
    append_uint(&out, "malformed_requests", direct.malformed_requests);
    append_uint(&out, "oversized_requests", direct.oversized_requests);
    append_uint(&out, "rate_limited_requests", direct.rate_limited_requests);
    append_uint(&out, "dropped_telemetry_events", direct.dropped_telemetry_events);
    append_uint(&out, "send_failures", direct.send_failures);
    append_uint(&out, "slow_client_disconnects", direct.slow_client_disconnects);
    out += "}";
  }
  out += "}";
  return out;
}

void RuntimeDirectHttpBridge::notify_config_changed_() {
  if (config_changed_) {
    config_changed_();
  }
}

}  // namespace espectre
