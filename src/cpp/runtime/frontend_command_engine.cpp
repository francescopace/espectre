/*
 * ESPectre - Frontend Command Engine
 *
 * Parses frontend control commands that update stored device
 * configuration.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "frontend_command_engine.h"

namespace espectre {

const char *frontend_command_parse_error_code(const std::string &error) {
  return error == "unsupported protocol_version" ? "unsupported_version" : "invalid_params";
}

bool frontend_command_allowed_during_raw_collection(const std::string &command) {
  return command == "capabilities" || command == "device" || command == "health" ||
         command == "sensing" || command == "wifi" || command == "mqtt" ||
         command == "read_diagnostics" || command == "ota" || command == "devices" ||
         command == "wifi_access_points" || command == "update_device" ||
         command == "update_mqtt" || command == "clear_mqtt";
}

DeviceConfigCommandResult handle_device_config_command(const std::string &command,
                                                       const EspectreDeviceConfig &current_config,
                                                       DeviceConfigClearHandler clear_handler,
                                                       DeviceConfigUpdateHandler update_handler) {
  DeviceConfigCommandResult result;

  if (command == "CLEAR_DEVICE_CONFIG") {
    result.handled = true;
    EspectreDeviceConfig cleared_config{};
    if (clear_handler) {
      result.accepted = clear_handler(&cleared_config, &result.message);
    }
    if (result.accepted) {
      result.config_changed = true;
      result.config = std::move(cleared_config);
    }
    return result;
  }

  if (command == "CLEAR_MQTT_CONFIG") {
    result.handled = true;
    EspectreDeviceConfig updated_config = current_config;
    clear_espectre_mqtt_config(&updated_config);
    if (update_handler) {
      result.accepted = update_handler(&updated_config, &result.message);
    }
    if (result.accepted) {
      result.config_changed = true;
      result.config = std::move(updated_config);
      if (result.message.empty()) {
        result.message = "mqtt settings cleared";
      }
    }
    return result;
  }

  if (command.rfind("SET_MQTT_CONFIG:", 0) == 0) {
    result.handled = true;
    EspectreDeviceConfig updated_config = current_config;
    std::string error;
    if (!parse_espectre_mqtt_config_command(command, &updated_config, &error)) {
      result.message = error.empty() ? "invalid mqtt config" : error;
      return result;
    }
    if (update_handler) {
      result.accepted = update_handler(&updated_config, &result.message);
    }
    if (result.accepted) {
      result.config_changed = true;
      result.config = std::move(updated_config);
      if (result.message.empty()) {
        result.message = "mqtt settings saved";
      }
    }
    return result;
  }

  if (command.rfind("SET_DEVICE_CONFIG:", 0) == 0) {
    result.handled = true;
    EspectreDeviceConfig updated_config = current_config;
    std::string error;
    if (!parse_espectre_config_command(command, &updated_config, &error)) {
      result.message = error.empty() ? "unsupported device config field" : error;
      return result;
    }
    if (update_handler) {
      result.accepted = update_handler(&updated_config, &result.message);
    }
    if (result.accepted) {
      result.config_changed = true;
      result.config = std::move(updated_config);
    }
    return result;
  }

  return result;
}

FrontendCommandResult FrontendCommandEngine::execute(
    const EspectreCommand &command,
    const FrontendCommandContext &context,
    IOtaService *ota_service,
    const char *current_version,
    const FrontendCommandCapabilities &capabilities,
    FrontendReadPayloadCallback read_payload_callback,
    FrontendDeviceLabelCallback device_label_callback,
    FrontendThresholdCallback threshold_callback,
    FrontendMotionHitsCallback motion_hits_callback,
    FrontendCsiTrafficModeCallback csi_traffic_mode_callback,
    FrontendTrafficGeneratorModeCallback traffic_generator_mode_callback,
    FrontendDetectorCallback detector_callback,
    FrontendRecalibrateCallback recalibrate_callback,
    FrontendWifiBssidCallback wifi_bssid_callback,
    FrontendMqttConfigCallback mqtt_config_callback,
    FrontendSensingControlCallback sensing_control_callback,
    FrontendRawStreamCallback raw_stream_callback) const {
  FrontendCommandResult result;
  result.handled = true;
  result.command = command;
  const auto reject = [&result](const char *code, const char *message) {
    result.code = code != nullptr ? code : "internal_error";
    result.message = message != nullptr ? message : "";
    return result;
  };
  const auto accept_read = [&result, &command, &read_payload_callback](const char *message) {
    if (!read_payload_callback) {
      result.code = "unsupported";
      result.message = "unsupported command";
      return result;
    }
    result.data_json = read_payload_callback(command);
    if (result.data_json.empty()) {
      result.code = "unavailable";
      result.message = "command data is unavailable";
      return result;
    }
    result.accepted = true;
    result.code = "ok";
    result.message = message != nullptr ? message : "";
    return result;
  };
  const auto supports = [&capabilities](EspectreDirectMethod method) {
    return capabilities.supports(method);
  };

  if (context.origin == FrontendCommandOrigin::MQTT &&
      command.command != "update_device" && command.command != "update_sensing" &&
      command.command != "recalibrate" && command.command != "read_diagnostics" &&
      command.command != "check_ota" && command.command != "start_ota") {
    return reject("forbidden", "command is not available over MQTT");
  }

  if (command.command == "capabilities") {
    return supports(EspectreDirectMethod::CAPABILITIES) ? accept_read("capabilities returned")
                                                        : reject("unsupported", "unsupported command");
  }
  if (command.command == "device") {
    return supports(EspectreDirectMethod::INFO) ? accept_read("info returned")
                                                : reject("unsupported", "unsupported command");
  }
  if (command.command == "health") {
    return supports(EspectreDirectMethod::STATUS) ? accept_read("status returned")
                                                  : reject("unsupported", "unsupported command");
  }
  if (command.command == "sensing" || command.command == "wifi" || command.command == "mqtt") {
    return supports(EspectreDirectMethod::CONFIG) ? accept_read("config returned")
                                                  : reject("unsupported", "unsupported command");
  }
  if (command.command == "read_diagnostics") {
    return supports(EspectreDirectMethod::DIAGNOSTICS) ? accept_read("diagnostics returned")
                                                       : reject("unsupported", "unsupported command");
  }
  if (command.command == "wifi_access_points") {
    return supports(EspectreDirectMethod::WIFI_ACCESS_POINTS)
               ? accept_read("Wi-Fi access points returned")
               : reject("unsupported", "unsupported command");
  }

  if (command.command == "update_device") {
    if (!supports(EspectreDirectMethod::SET_DEVICE_LABEL) || !device_label_callback ||
        !command.has_device_label) {
      return !supports(EspectreDirectMethod::SET_DEVICE_LABEL) || !device_label_callback
                 ? reject("unsupported", "unsupported command")
                 : reject("invalid_params", "invalid device label (accepted: a single-line string)");
    }
    result.accepted = device_label_callback(command.device_label, &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) {
      result.changes = FrontendCommandChange::DEVICE;
    }
    if (result.message.empty()) {
      result.message = result.accepted ? "device label updated" : "device label rejected";
    }
    return result;
  }

  if (command.command == "scan_wifi" || command.command == "set_wifi_bssid" ||
      command.command == "clear_wifi_bssid" || command.command == "clear_wifi_credentials") {
    EspectreDirectMethod method = EspectreDirectMethod::SCAN_WIFI_ACCESS_POINTS;
    if (command.command == "set_wifi_bssid") method = EspectreDirectMethod::SET_WIFI_BSSID;
    if (command.command == "clear_wifi_bssid") method = EspectreDirectMethod::CLEAR_WIFI_BSSID;
    if (command.command == "clear_wifi_credentials") method = EspectreDirectMethod::CLEAR_WIFI_CONFIG;
    if (!supports(method) || !wifi_bssid_callback) {
      return reject("unsupported", "unsupported command");
    }
    result.accepted = wifi_bssid_callback(command, &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted && command.command != "scan_wifi") {
      result.changes = FrontendCommandChange::WIFI;
    }
    if (result.message.empty()) {
      result.message = result.accepted
                           ? (command.command == "scan_wifi"
                                  ? "Wi-Fi access point scan started"
                                  : command.command == "clear_wifi_credentials"
                                      ? "Wi-Fi configuration cleared"
                                      : command.command == "clear_wifi_bssid"
                                          ? "Wi-Fi BSSID pin cleared"
                                          : "Wi-Fi BSSID accepted")
                           : "Wi-Fi access point request rejected";
    }
    return result;
  }

  if (command.command == "update_mqtt" || command.command == "clear_mqtt") {
    const EspectreDirectMethod method = command.command == "update_mqtt"
                                            ? EspectreDirectMethod::SET_MQTT_CONFIG
                                            : EspectreDirectMethod::CLEAR_MQTT_CONFIG;
    if (!supports(method) || !mqtt_config_callback) {
      return reject("unsupported", "unsupported command");
    }
    result.accepted = mqtt_config_callback(command, command.command == "clear_mqtt", &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::MQTT;
    if (result.message.empty()) {
      result.message = result.accepted ? "MQTT configuration updated" : "MQTT configuration rejected";
    }
    return result;
  }

  if (command.command == "update_sensing") {
    if (command.has_threshold && (!supports(EspectreDirectMethod::SET_THRESHOLD) ||
                                  !threshold_callback || !validate_runtime_threshold(command.threshold))) {
      return reject("invalid_params", "invalid threshold (accepted: 0.0-1.0)");
    }
    if (command.has_motion_hits && (!supports(EspectreDirectMethod::SET_MOTION_HITS) || !motion_hits_callback)) {
      return reject("unsupported", "motion hits update is unsupported");
    }
    if (command.has_detector && (!supports(EspectreDirectMethod::SET_DETECTOR) || !detector_callback)) {
      return reject("unsupported", "detector update is unsupported");
    }
    if (command.has_motion_hits && (command.motion_on_hits < RUNTIME_MOTION_HITS_MIN ||
        command.motion_on_hits > RUNTIME_MOTION_HITS_MAX || command.motion_off_hits < RUNTIME_MOTION_HITS_MIN ||
        command.motion_off_hits > RUNTIME_MOTION_HITS_MAX)) {
      return reject("invalid_params", "invalid motion hits (accepted: motion_on_hits and motion_off_hits in 1-20)");
    }
    if (command.has_csi_traffic_mode &&
        (!supports(EspectreDirectMethod::SET_CSI_TRAFFIC_MODE) || !csi_traffic_mode_callback)) {
      return reject("unsupported", "CSI traffic mode update is unsupported");
    }
    if (command.has_traffic_generator_mode &&
        (!supports(EspectreDirectMethod::SET_TRAFFIC_GENERATOR_MODE) || !traffic_generator_mode_callback)) {
      return reject("unsupported", "traffic generator update is unsupported");
    }
    if (command.has_sensing_enabled &&
        (!supports(EspectreDirectMethod::SET_SENSING) || !sensing_control_callback)) {
      return reject("unsupported", "sensing state update is unsupported");
    }
    result.accepted = true;
    if (command.has_detector) {
      result.accepted = detector_callback(parse_detection_algorithm(command.detector.c_str()), &result.message);
    }
    if (result.accepted && command.has_threshold) {
      result.accepted = threshold_callback(command.threshold, &result.message);
    }
    if (result.accepted && command.has_motion_hits) {
      result.accepted = motion_hits_callback(command.motion_on_hits, command.motion_off_hits, &result.message);
    }
    if (result.accepted && command.has_csi_traffic_mode) {
      result.accepted = csi_traffic_mode_callback(
          parse_csi_traffic_mode(command.csi_traffic_mode.c_str()), &result.message);
    }
    if (result.accepted && command.has_traffic_generator_mode) {
      result.accepted = traffic_generator_mode_callback(
          parse_traffic_mode(command.traffic_generator_mode.c_str()), &result.message);
    }
    if (result.accepted && command.has_sensing_enabled) {
      result.accepted = sensing_control_callback(command.sensing_enabled, &result.message);
    }
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::SENSING;
    if (result.message.empty()) result.message = result.accepted ? "sensing updated" : "sensing update rejected";
    return result;
  }

  if (command.command == "recalibrate") {
    if (!supports(EspectreDirectMethod::RECALIBRATE) || !recalibrate_callback) {
      return reject("unsupported", "unsupported command");
    }
    result.accepted = recalibrate_callback(&result.message);
    result.code = result.accepted ? "ok" : "busy";
    if (result.accepted) result.changes = FrontendCommandChange::SENSING;
    if (result.message.empty()) {
      result.message = result.accepted ? "recalibration started" : "recalibration rejected";
    }
    return result;
  }

  if (command.command == "ota" || command.command == "check_ota" || command.command == "start_ota") {
    EspectreDirectMethod method = EspectreDirectMethod::OTA_STATUS;
    if (command.command == "check_ota") method = EspectreDirectMethod::OTA_CHECK;
    if (command.command == "start_ota") method = EspectreDirectMethod::OTA_START;
    if (!supports(method)) {
      return reject("unsupported", "unsupported command");
    }
    if (ota_service == nullptr) {
      return reject("unavailable", "ota unavailable");
    }
    const std::string normalized_current_version =
        (current_version == nullptr || current_version[0] == '\0') ? "unknown" : current_version;
    if (command.command == "ota") {
      return accept_read("ota status returned");
    }
    if (command.command == "check_ota") {
      result.accepted = ota_service->start_check(normalized_current_version, command.ota_channel);
      result.code = result.accepted ? "ok" : "busy";
      if (result.accepted) result.changes = FrontendCommandChange::OTA;
      result.message = result.accepted ? "ota check started" : "ota check rejected";
      return result;
    }
    result.accepted = ota_service->start_update(normalized_current_version, command.ota_channel);
    result.code = result.accepted ? "ok" : "busy";
    if (result.accepted) result.changes = FrontendCommandChange::OTA;
    result.message = result.accepted ? "ota update started" : "ota update rejected";
    return result;
  }

  return reject("unsupported", "unsupported command");
}

}  // namespace espectre
