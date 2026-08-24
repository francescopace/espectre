/*
 * ESPectre - Frontend Control Helpers
 *
 * Parses frontend control commands that update stored device
 * configuration.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "frontend_control_helpers.h"

namespace espectre {

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
    FrontendWifiConfigCallback wifi_config_callback,
    FrontendMqttConfigCallback mqtt_config_callback,
    FrontendSensingControlCallback sensing_control_callback) const {
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

  if (context.origin == FrontendCommandOrigin::MQTT &&
      (command.command == "set_wifi_config" || command.command == "clear_wifi_config" ||
       command.command == "set_mqtt_config" || command.command == "clear_mqtt_config" ||
       command.command == "discover_peers")) {
    return reject("forbidden", "command is local to Direct WebSocket");
  }

  if (command.command == "capabilities") return accept_read("capabilities returned");
  if (command.command == "info") {
    return capabilities.supports_info ? accept_read("info returned")
                                      : reject("unsupported", "unsupported command");
  }
  if (command.command == "status") {
    return capabilities.supports_status ? accept_read("status returned")
                                        : reject("unsupported", "unsupported command");
  }
  if (command.command == "config") {
    return capabilities.supports_config ? accept_read("config returned")
                                        : reject("unsupported", "unsupported command");
  }
  if (command.command == "diagnostics") {
    return capabilities.supports_diagnostics ? accept_read("diagnostics returned")
                                             : reject("unsupported", "unsupported command");
  }

  if (command.command == "set_device_label") {
    if (!capabilities.supports_device_config || !device_label_callback || !command.has_device_label) {
      return !capabilities.supports_device_config || !device_label_callback
                 ? reject("unsupported", "unsupported command")
                 : reject("invalid_params", "invalid device label (accepted: a single-line string)");
    }
    result.accepted = device_label_callback(command.device_label, &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::INFO;
    if (result.message.empty()) {
      result.message = result.accepted ? "device label updated" : "device label rejected";
    }
    return result;
  }

  if (command.command == "set_wifi_config" || command.command == "clear_wifi_config") {
    if (!capabilities.supports_wifi_config || !wifi_config_callback) {
      return reject("unsupported", "unsupported command");
    }
    result.accepted = wifi_config_callback(command, command.command == "clear_wifi_config", &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::CONFIG;
    if (result.message.empty()) {
      result.message = result.accepted ? "Wi-Fi configuration accepted" : "Wi-Fi configuration rejected";
    }
    return result;
  }

  if (command.command == "set_mqtt_config" || command.command == "clear_mqtt_config") {
    if (!capabilities.supports_mqtt_config || !mqtt_config_callback) {
      return reject("unsupported", "unsupported command");
    }
    result.accepted = mqtt_config_callback(command, command.command == "clear_mqtt_config", &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::CONFIG;
    if (result.message.empty()) {
      result.message = result.accepted ? "MQTT configuration updated" : "MQTT configuration rejected";
    }
    return result;
  }

  if (command.command == "set_sensing") {
    if (!capabilities.supports_sensing_control || !sensing_control_callback) {
      return reject("unsupported", "unsupported command");
    }
    if (!command.has_sensing_enabled) {
      return reject("invalid_params", "invalid sensing state (accepted: boolean enabled)");
    }
    const bool enabled = command.sensing_enabled;
    result.accepted = sensing_control_callback(enabled, &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::STATUS;
    if (result.message.empty()) {
      result.message = result.accepted ? (enabled ? "sensing started" : "sensing stopped")
                                       : (enabled ? "sensing start rejected" : "sensing stop rejected");
    }
    return result;
  }

  if (command.command == "set_threshold") {
    if (!capabilities.supports_threshold || !threshold_callback) {
      return reject("unsupported", "unsupported command");
    }
    if (!command.has_threshold || !validate_runtime_threshold(command.threshold)) {
      return reject("invalid_params", "invalid threshold (accepted: 0.0-1.0)");
    }
    result.accepted = threshold_callback(command.threshold, &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::CONFIG;
    if (result.message.empty()) {
      result.message = result.accepted ? "threshold updated" : "threshold rejected";
    }
    return result;
  }

  if (command.command == "set_motion_hits") {
    if (!capabilities.supports_motion_hits || !motion_hits_callback) {
      return reject("unsupported", "unsupported command");
    }
    if (!command.has_motion_hits || command.motion_on_hits < RUNTIME_MOTION_HITS_MIN ||
        command.motion_on_hits > RUNTIME_MOTION_HITS_MAX || command.motion_off_hits < RUNTIME_MOTION_HITS_MIN ||
        command.motion_off_hits > RUNTIME_MOTION_HITS_MAX) {
      return reject("invalid_params", "invalid motion hits (accepted: motion_on_hits and motion_off_hits in 1-20)");
    }
    result.accepted = motion_hits_callback(command.motion_on_hits, command.motion_off_hits, &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::CONFIG;
    if (result.message.empty()) {
      result.message = result.accepted ? "motion hits updated" : "motion hits rejected";
    }
    return result;
  }

  if (command.command == "set_csi_traffic_mode") {
    if (!capabilities.supports_traffic_control || !csi_traffic_mode_callback) {
      return reject("unsupported", "unsupported command");
    }
    if (!command.has_csi_traffic_mode) {
      return reject("invalid_params", "invalid csi traffic mode (accepted: internal, external, and disabled)");
    }
    result.accepted = csi_traffic_mode_callback(parse_csi_traffic_mode(command.csi_traffic_mode.c_str()), &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::CONFIG;
    if (result.message.empty()) {
      result.message = result.accepted ? "csi traffic mode updated" : "csi traffic mode rejected";
    }
    return result;
  }

  if (command.command == "set_traffic_generator_mode") {
    if (!capabilities.supports_traffic_control || !traffic_generator_mode_callback) {
      return reject("unsupported", "unsupported command");
    }
    if (!command.has_traffic_generator_mode) {
      return reject("invalid_params", "invalid traffic generator mode (accepted: ping and dns)");
    }
    result.accepted =
        traffic_generator_mode_callback(parse_traffic_mode(command.traffic_generator_mode.c_str()), &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::CONFIG;
    if (result.message.empty()) {
      result.message = result.accepted ? "traffic generator mode updated" : "traffic generator mode rejected";
    }
    return result;
  }

  if (command.command == "set_detector") {
    if (!capabilities.supports_detector || !detector_callback) {
      return reject("unsupported", "unsupported command");
    }
    if (!command.has_detector) {
      return reject("invalid_params", "invalid detector (accepted: lightweight and high_accuracy)");
    }
    result.accepted = detector_callback(parse_detection_algorithm(command.detector.c_str()), &result.message);
    result.code = result.accepted ? "ok" : "unavailable";
    if (result.accepted) result.changes = FrontendCommandChange::CONFIG;
    if (result.message.empty()) {
      result.message = result.accepted ? "detector updated" : "detector rejected";
    }
    return result;
  }

  if (command.command == "recalibrate") {
    if (!capabilities.supports_recalibrate || !recalibrate_callback) {
      return reject("unsupported", "unsupported command");
    }
    result.accepted = recalibrate_callback(&result.message);
    result.code = result.accepted ? "ok" : "busy";
    if (result.accepted) result.changes = FrontendCommandChange::STATUS;
    if (result.message.empty()) {
      result.message = result.accepted ? "recalibration started" : "recalibration rejected";
    }
    return result;
  }

  if (command.command == "ota_status" || command.command == "ota_check" || command.command == "ota_start") {
    if (!capabilities.supports_ota || ota_service == nullptr) {
      return reject("unavailable", "ota unavailable");
    }
    const std::string normalized_current_version =
        (current_version == nullptr || current_version[0] == '\0') ? "unknown" : current_version;
    if (command.command == "ota_status") {
      return accept_read("ota status returned");
    }
    if (command.command == "ota_check") {
      result.accepted = ota_service->start_check(normalized_current_version, command.ota_channel);
      result.code = result.accepted ? "ok" : "busy";
      if (result.accepted) result.changes = FrontendCommandChange::OTA_STATUS;
      result.message = result.accepted ? "ota check started" : "ota check rejected";
      return result;
    }
    result.accepted = ota_service->start_update(normalized_current_version, command.ota_channel);
    result.code = result.accepted ? "ok" : "busy";
    if (result.accepted) result.changes = FrontendCommandChange::OTA_STATUS;
    result.message = result.accepted ? "ota update started" : "ota update rejected";
    return result;
  }

  return reject("unsupported", "unsupported command");
}

}  // namespace espectre
