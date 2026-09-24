/*
 * ESPectre - Frontend Command Engine
 *
 * Dispatches parsed protocol commands to frontend callbacks.
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

bool frontend_command_allowed_during_raw_collection(const std::string &command,
                                                   const EspectreProtocolExtension *extension) {
  if (const auto *route = find_extension_route(extension, command)) return route->allowed_during_raw_collection;
  return command == "capabilities" || command == "device" || command == "health" ||
         command == "sensing" || command == "wifi" || command == "mqtt" ||
         command == "read_diagnostics" || command == "devices" ||
         command == "wifi_access_points" || command == "update_device" ||
         command == "update_mqtt" || command == "clear_mqtt";
}

FrontendCommandResult FrontendCommandEngine::execute(
    const EspectreCommand &command,
    const FrontendCommandContext &context,
    const FrontendCommandCapabilities &capabilities,
    FrontendReadPayloadCallback read_payload_callback,
    FrontendDeviceLabelCallback device_label_callback,
    FrontendThresholdCallback threshold_callback,
    FrontendMotionHitsCallback motion_hits_callback,
    FrontendTrafficGeneratorModeCallback traffic_generator_mode_callback,
    FrontendDetectorCallback detector_callback,
    FrontendRecalibrateCallback recalibrate_callback,
    FrontendWifiBssidCallback wifi_bssid_callback,
    FrontendMqttConfigCallback mqtt_config_callback,
    FrontendSensingControlCallback sensing_control_callback,
    FrontendSensingPreflightCallback sensing_preflight_callback) const {
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
      command.command != "recalibrate" && command.command != "read_diagnostics") {
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
    if (!supports(EspectreDirectMethod::SET_DEVICE_LABEL) || !device_label_callback) {
      return reject("unsupported", "unsupported command");
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
                                  !threshold_callback)) {
      return reject("unsupported", "threshold update is unsupported");
    }
    if (command.has_motion_hits && (!supports(EspectreDirectMethod::SET_MOTION_HITS) || !motion_hits_callback)) {
      return reject("unsupported", "motion hits update is unsupported");
    }
    if (command.has_detector && (!supports(EspectreDirectMethod::SET_DETECTOR) || !detector_callback)) {
      return reject("unsupported", "detector update is unsupported");
    }
    if (command.has_traffic_generator_mode &&
        (!supports(EspectreDirectMethod::SET_TRAFFIC_GENERATOR_MODE) || !traffic_generator_mode_callback)) {
      return reject("unsupported", "traffic generator update is unsupported");
    }
    if (command.has_sensing_enabled &&
        (!supports(EspectreDirectMethod::SET_SENSING) || !sensing_control_callback)) {
      return reject("unsupported", "sensing state update is unsupported");
    }
    if (sensing_preflight_callback) {
      RuntimeControlUpdate update;
      update.has_detection_algorithm = command.has_detector;
      update.detection_algorithm = parse_detection_algorithm(command.detector.c_str());
      update.has_threshold = command.has_threshold;
      update.threshold = command.threshold;
      update.has_motion_hits = command.has_motion_hits;
      update.motion_on_hits = command.motion_on_hits;
      update.motion_off_hits = command.motion_off_hits;
      update.has_traffic_generator_mode = command.has_traffic_generator_mode;
      update.traffic_generator_mode = parse_traffic_generator_mode(command.traffic_generator_mode.c_str());
      std::string preflight_message;
      if (!sensing_preflight_callback(update, &preflight_message)) {
        return reject("invalid_params",
                      preflight_message.empty() ? "sensing update is invalid" : preflight_message.c_str());
      }
    }
    // Fields apply one at a time and cannot be rolled back. A backend refusal
    // after an applied field still reports SENSING so transports republish
    // the state the device actually reached.
    bool applied_any = false;
    const char *rejected_field = nullptr;
    const auto apply = [&](bool present, const char *field, const auto &callback) {
      if (!present || rejected_field != nullptr) return;
      if (callback()) {
        applied_any = true;
      } else {
        rejected_field = field;
      }
    };
    apply(command.has_detector, "detector", [&] {
      return detector_callback(parse_detection_algorithm(command.detector.c_str()), &result.message);
    });
    apply(command.has_threshold, "threshold",
          [&] { return threshold_callback(command.threshold, &result.message); });
    apply(command.has_motion_hits, "motion hits", [&] {
      return motion_hits_callback(command.motion_on_hits, command.motion_off_hits, &result.message);
    });
    apply(command.has_traffic_generator_mode, "traffic generator mode", [&] {
      return traffic_generator_mode_callback(
          parse_traffic_generator_mode(command.traffic_generator_mode.c_str()), &result.message);
    });
    apply(command.has_sensing_enabled, "sensing state",
          [&] { return sensing_control_callback(command.sensing_enabled, &result.message); });
    result.accepted = rejected_field == nullptr;
    result.code = result.accepted ? "ok" : "unavailable";
    if (applied_any) result.changes = FrontendCommandChange::SENSING;
    if (result.message.empty()) {
      result.message = result.accepted ? "sensing updated"
                                       : std::string(rejected_field) + " update rejected";
    }
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

  return reject("unsupported", "unsupported command");
}

}  // namespace espectre
