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

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "direct_http_protocol.h"
#include "espectre_protocol.h"
#include "protocol_json.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "runtime_time.h"

#if defined(ESP_PLATFORM)
#include "esp_wifi.h"
#endif

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
  raw_session_controller_.configure(service_, runtime_, config_.device_id, config_.chip);
  refresh_peer_candidate_();

  DirectHttpServiceConfig service_config = DirectHttpServiceConfig::for_first_party_portals();
  service_config.port = config.port;
  service_config.allow_missing_origin = config.allow_missing_origin;
#if defined(CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED) && CONFIG_ESPECTRE_DIRECT_DEV_ORIGINS_ENABLED
  service_config.allow_http_loopback_origins = true;
#endif
  deferred_requests_enabled_ = false;
  bool setup = false;
  if (config_.peer_discovery != nullptr) {
    setup = service_->setup_deferred(
        service_config,
        [this](uint64_t token, const DirectRequest &request) {
          return this->handle_deferred_request_(token, request);
        },
        [](size_t client_count) { (void) client_count; });
    deferred_requests_enabled_ = setup;
  }
  if (!setup) {
    setup = service_->setup(
        service_config,
        [this](const DirectRequest &request) { return this->handle_request_(request); },
        [](size_t client_count) { (void) client_count; });
  }
  if (!setup) {
    service_ = nullptr;
    runtime_ = nullptr;
    config_changed_ = {};
    deferred_requests_enabled_ = false;
    return false;
  }
  return true;
}

void RuntimeDirectHttpBridge::loop() {
  if (service_ != nullptr && service_->running()) {
    raw_session_controller_.ensure_runtime_consistency();
    if (config_.peer_discovery != nullptr) {
      config_.peer_discovery->set_wifi_ready(wifi_snapshot_().connected);
      config_.peer_discovery->loop();
    }
    service_->loop();
  }
}

void RuntimeDirectHttpBridge::shutdown() {
  raw_session_controller_.shutdown();
  if (config_.peer_discovery != nullptr) config_.peer_discovery->shutdown();
  if (service_ != nullptr && service_->running()) {
    service_->shutdown();
  }
  service_ = nullptr;
  runtime_ = nullptr;
  config_changed_ = {};
  deferred_requests_enabled_ = false;
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
    refresh_peer_candidate_();
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
  const FrontendCommandCapabilities capabilities = capability_profile_();
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
        if (read.command == "wifi_access_points") return wifi_access_points_payload_();
        return std::string{};
      },
      config_.device_label_setter,
      [this](float value, std::string *) { return runtime_->set_threshold_runtime(value); },
      [this](uint8_t on, uint8_t off, std::string *) { return runtime_->set_motion_hits_runtime(on, off); },
      [this](CsiTrafficMode mode, std::string *) { return runtime_->set_csi_traffic_mode_runtime(mode); },
      [this](RuntimeTrafficMode mode, std::string *) { return runtime_->set_traffic_generator_mode_runtime(mode); },
      [this](DetectionAlgorithm algorithm, std::string *) {
        return runtime_->set_detection_algorithm_runtime(algorithm);
      },
      [this](std::string *) { return runtime_->trigger_recalibration(); },
      [this](const EspectreCommand &wifi, std::string *message) {
        return handle_wifi_control_(wifi, message);
      },
      {},
      [this](bool enabled, std::string *) {
        runtime_->set_services_armed(enabled);
        return true;
      },
      [this](const EspectreCommand &raw,
             const FrontendCommandContext &context,
             std::string *code,
             std::string *message,
             std::string *data_json) {
        return handle_raw_stream_(raw, context, code, message, data_json);
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

IDirectHttpService::DeferredRequestResult RuntimeDirectHttpBridge::handle_deferred_request_(
    uint64_t request_token,
    const DirectRequest &request) {
  if (request.method != ESPECTRE_PEER_DISCOVERY_METHOD) {
    return {false, handle_request_(request)};
  }
  if (!deferred_requests_enabled_ || config_.peer_discovery == nullptr) {
    return {false, direct_http_error_response(request.id, "unsupported", "peer discovery is unavailable")};
  }
  std::vector<JsonObjectField> params;
  if (!parse_json_object_fields(request.params, &params) || !params.empty()) {
    return {false,
            direct_http_error_response(
                request.id, "invalid_params", "discover_peers does not accept parameters")};
  }
  if (config_.peer_discovery->active()) {
    return {false,
            direct_http_error_response(
                request.id, "conflict", "a peer discovery request is already active")};
  }
  const bool started = config_.peer_discovery->start(
      [this, request_token, request_id = request.id](PeerDiscoverySnapshot snapshot) {
        if (service_ == nullptr) return;
        std::string result{"{\"command\":\"discover_peers\",\"code\":\"ok\""};
        append_json_pair(&result, "message", "peer discovery completed");
        result += ",\"data\":" + peer_discovery_snapshot_json(snapshot) + "}";
        (void) service_->complete_deferred_response(
            request_token, direct_http_success_response(request_id, result));
      });
  if (!started) {
    return {false,
            direct_http_error_response(
                request.id, "unavailable", "peer discovery could not be started")};
  }
  return {true, {}};
}

EspectreCapabilityProfile RuntimeDirectHttpBridge::capability_profile_() const {
  EspectreCapabilityProfile profile;
  if (runtime_ == nullptr) return profile;
  const RuntimeCapabilities &runtime_capabilities = runtime_->capabilities();
  using Method = EspectreDirectMethod;
  profile.set(Method::CAPABILITIES);
  profile.set(Method::INFO);
  profile.set(Method::STATUS);
  profile.set(Method::CONFIG);
  profile.set(Method::DIAGNOSTICS, runtime_capabilities.supports_extended_diagnostics);
  profile.set(Method::SET_SENSING);
  profile.set(Method::SET_DEVICE_LABEL, static_cast<bool>(config_.device_label_setter));
  profile.set(Method::SET_THRESHOLD, runtime_capabilities.supports_runtime_threshold_updates);
  profile.set(Method::SET_MOTION_HITS, runtime_capabilities.supports_runtime_motion_hits_updates);
  profile.set(Method::SET_DETECTOR, runtime_capabilities.supports_runtime_detector_selection);
  profile.set(Method::RECALIBRATE, runtime_capabilities.supports_manual_recalibration);
  profile.set(Method::SET_CSI_TRAFFIC_MODE, runtime_capabilities.supports_traffic_control);
  profile.set(Method::SET_TRAFFIC_GENERATOR_MODE, runtime_capabilities.supports_traffic_control);
  profile.set(Method::WIFI_ACCESS_POINTS);
  profile.set(Method::SCAN_WIFI_ACCESS_POINTS);
  profile.set(Method::SET_WIFI_BSSID);
  profile.set(Method::CLEAR_WIFI_BSSID);
  const bool raw_csi = config_.raw_csi && runtime_capabilities.supports_raw_csi;
  profile.set(Method::START_RAW_STREAM, raw_csi);
  profile.set(Method::STOP_RAW_STREAM, raw_csi);
  profile.set(Method::DISCOVER_PEERS,
              deferred_requests_enabled_ && config_.peer_discovery != nullptr);
  profile.set(EspectreConfigSection::RUNTIME);
  profile.set(EspectreConfigSection::DEVICE);
  profile.set(EspectreConfigSection::WIFI);
  return profile;
}

std::string RuntimeDirectHttpBridge::device_label_() const {
  return config_.device_label_getter ? config_.device_label_getter() : config_.device_name;
}

DirectWifiSnapshot RuntimeDirectHttpBridge::wifi_snapshot_() const {
  return config_.wifi_snapshot_getter ? config_.wifi_snapshot_getter() : read_direct_wifi_snapshot();
}

std::string RuntimeDirectHttpBridge::wifi_access_points_payload_() const {
  std::string out{"{\"scanning\":false,\"message\":\"\",\"access_points\":["};
#if defined(ESP_PLATFORM)
  uint16_t count = 0U;
  const esp_err_t count_result = esp_wifi_scan_get_ap_num(&count);
  if (count_result == ESP_OK && count > 0U) {
    count = std::min<uint16_t>(count, 32U);
    std::vector<wifi_ap_record_t> records(count);
    if (esp_wifi_scan_get_ap_records(&count, records.data()) == ESP_OK) {
      bool first = true;
      for (uint16_t index = 0U; index < count; ++index) {
        if (!first) out += ',';
        first = false;
        char bssid[18]{};
        std::snprintf(bssid,
                      sizeof(bssid),
                      "%02X:%02X:%02X:%02X:%02X:%02X",
                      records[index].bssid[0],
                      records[index].bssid[1],
                      records[index].bssid[2],
                      records[index].bssid[3],
                      records[index].bssid[4],
                      records[index].bssid[5]);
        out += '{';
        append_json_pair(&out, "bssid", bssid, true);
        out += ",\"rssi_dbm\":" + std::to_string(static_cast<int>(records[index].rssi));
        append_uint(&out, "channel", records[index].primary);
        out += '}';
      }
    }
  } else if (count_result == ESP_ERR_WIFI_STATE) {
    out.replace(12U, 5U, "true");
  }
#endif
  out += "]}";
  return out;
}

bool RuntimeDirectHttpBridge::handle_wifi_control_(const EspectreCommand &command,
                                                    std::string *message) {
#if defined(ESP_PLATFORM)
  if (command.command == "scan_wifi_access_points") {
    wifi_scan_config_t scan{};
    const esp_err_t result = esp_wifi_scan_start(&scan, false);
    if (message != nullptr) {
      *message = result == ESP_OK ? "Wi-Fi access point scan started"
                                  : "Wi-Fi access point scan could not be started";
    }
    return result == ESP_OK;
  }
  wifi_config_t config{};
  if (esp_wifi_get_config(WIFI_IF_STA, &config) != ESP_OK) return false;
  if (command.command == "clear_wifi_bssid") {
    config.sta.bssid_set = false;
    std::memset(config.sta.bssid, 0, sizeof(config.sta.bssid));
    config.sta.channel = 0U;
  } else if (command.command == "set_wifi_bssid" && command.has_wifi_bssid) {
    unsigned int octets[6]{};
    if (std::sscanf(command.wifi_bssid.c_str(),
                    "%2x:%2x:%2x:%2x:%2x:%2x",
                    &octets[0],
                    &octets[1],
                    &octets[2],
                    &octets[3],
                    &octets[4],
                    &octets[5]) != 6) {
      return false;
    }
    for (size_t index = 0U; index < 6U; ++index) {
      config.sta.bssid[index] = static_cast<uint8_t>(octets[index]);
    }
    config.sta.bssid_set = true;
  } else {
    return false;
  }
  const bool updated = esp_wifi_set_config(WIFI_IF_STA, &config) == ESP_OK;
  bool reconnect_started = false;
  if (updated) {
    (void) esp_wifi_disconnect();
    reconnect_started = esp_wifi_connect() == ESP_OK;
  }
  if (message != nullptr) {
    *message = updated && reconnect_started
                   ? (command.command == "clear_wifi_bssid" ? "Wi-Fi BSSID pin cleared"
                                                             : "Wi-Fi BSSID pin updated")
                   : "Wi-Fi BSSID pin update failed";
  }
  return updated && reconnect_started;
#else
  (void) command;
  if (message != nullptr) *message = "Wi-Fi control accepted by host test adapter";
  return true;
#endif
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
  info.csi_traffic_udp_port = runtime_->config().csi_traffic_udp_port;
  info.csi_traffic_multicast_group = runtime_->config().csi_traffic_multicast_group;
  return espectre_capabilities_payload(device, info, capability_profile_());
}

bool RuntimeDirectHttpBridge::handle_raw_stream_(const EspectreCommand &command,
                                                  const FrontendCommandContext &context,
                                                  std::string *code,
                                                  std::string *message,
                                                  std::string *data_json) {
  if (!config_.raw_csi) {
    if (code != nullptr) *code = "unsupported";
    if (message != nullptr) *message = "raw CSI collection is unavailable";
    return false;
  }
  return raw_session_controller_.handle_command(command, context, code, message, data_json);
}

std::string RuntimeDirectHttpBridge::info_payload_() const {
  EspectreDeviceConfig device;
  device.device_id = config_.device_id;
  device.device_label = device_label_();
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
  append_bool(&out, "wifi_connected", wifi_snapshot_().connected);
  out += "}";
  return out;
}

std::string RuntimeDirectHttpBridge::config_payload_() const {
  const RuntimeConfig &config = runtime_->config();
  std::string out{"{"};
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", format_espectre_device_id(config_.device_id).c_str());
  out += ",\"device\":{";
  append_json_pair(&out, "device_label", device_label_().c_str(), true);
  out += "},\"wifi\":{";
  const DirectWifiSnapshot wifi = wifi_snapshot_();
  append_bool(&out, "configured", wifi.configured, true);
  append_bool(&out, "connected", wifi.connected);
  append_json_pair(&out, "ssid", wifi.ssid.c_str());
  append_json_pair(&out, "bssid", wifi.bssid.c_str());
  append_json_pair(&out, "band", wifi.band.c_str());
  append_uint(&out, "channel", wifi.channel);
  out += ",\"rssi_dbm\":";
  out += wifi.rssi_dbm == INT16_MIN ? "null" : std::to_string(wifi.rssi_dbm);
  out += "},\"runtime\":{";
  out += "\"threshold\":" + std::to_string(runtime_->snapshot().threshold);
  append_json_pair(&out, "detector", detection_algorithm_name(config.detection_algorithm));
  append_uint(&out, "motion_on_hits", config.motion_on_hits);
  append_uint(&out, "motion_off_hits", config.motion_off_hits);
  append_json_pair(&out, "csi_traffic_mode", csi_traffic_mode_name(config.csi_traffic_mode));
  append_json_pair(&out, "traffic_generator_mode", traffic_mode_name(config.traffic_generator_mode));
  append_uint(&out, "csi_target_pps", config.csi_target_pps);
  append_uint(&out, "csi_traffic_udp_port", config.csi_traffic_udp_port);
  append_json_pair(&out, "csi_traffic_multicast_group", config.csi_traffic_multicast_group.c_str());
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
  append_uint(&out, "csi_classified_total", diagnostics.csi_classified_total);
  append_uint(&out,
              "csi_provenance_rejected_total",
              diagnostics.csi_provenance_rejected_total);
  append_uint(&out, "csi_accepted_total", diagnostics.csi_accepted_total);
  append_uint(&out, "csi_admitted_total", diagnostics.csi_admitted_total);
  append_uint(&out, "csi_filtered_total", diagnostics.csi_filtered_total);
  append_uint(&out, "csi_missing_slots_total", diagnostics.csi_missing_slots_total);
  append_uint(&out, "csi_excess_total", diagnostics.csi_excess_total);
  append_uint(&out, "csi_stale_total", diagnostics.csi_stale_total);
  append_uint(&out, "csi_out_of_order_total", diagnostics.csi_out_of_order_total);
  append_uint(&out, "csi_occupancy_slots", diagnostics.csi_occupancy_slots);
  append_uint(&out, "csi_window_slots", diagnostics.csi_window_slots);
  out += ",\"wifi_rssi_dbm\":";
  out += diagnostics.wifi_rssi_dbm == INT8_MIN
             ? "null"
             : std::to_string(static_cast<int>(diagnostics.wifi_rssi_dbm));
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
    const RawCsiSessionDiagnostics raw = service_->raw_diagnostics();
    out += ",\"raw_csi\":{\"active\":";
    out += raw.active ? "true" : "false";
    out += ",\"binary_bound\":";
    out += raw.binary_bound ? "true" : "false";
    append_uint(&out, "raw_drop_total", raw.raw_drop_total);
    append_uint(&out, "send_backpressure_total", raw.raw_send_backpressure_total);
    append_uint(&out, "fresh_record_total", raw.fresh_record_total);
    append_uint(&out, "stream_sequence", raw.stream_sequence);
    out += "}";
  }
  out += "}";
  return out;
}

void RuntimeDirectHttpBridge::refresh_peer_candidate_() {
  if (config_.peer_discovery == nullptr) return;
  const std::string device_id = format_espectre_device_id(config_.device_id);
  const std::string device_label = device_label_();
  PeerDiscoveryCandidate local;
  local.instance = device_label + " " + device_id;
  local.device_id = device_id;
  local.name = device_label;
  local.frontend = config_.frontend;
  local.txt_version = ESPECTRE_DIRECT_DISCOVERY_TXT_VERSION;
  local.protocol_version = ESPECTRE_PROTOCOL_VERSION;
  local.transport = ESPECTRE_DIRECT_HTTP_TRANSPORT;
  local.path = ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT;
  local.events = ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT;
  local.firmware = config_.firmware_version;
  local.chip = config_.chip;
  local.capabilities = "config,monitor,raw_csi";
  local.port = config_.port;
  config_.peer_discovery->set_local_candidate(std::move(local));
}

void RuntimeDirectHttpBridge::notify_config_changed_() {
  if (config_changed_) {
    config_changed_();
  }
}

}  // namespace espectre
