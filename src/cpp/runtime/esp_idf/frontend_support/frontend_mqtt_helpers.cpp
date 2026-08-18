/*
 * ESPectre - Frontend MQTT Helpers
 *
 * Sets up frontend MQTT transport and handles shared command and status
 * payloads.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "frontend_mqtt_helpers.h"

#include <utility>

#include "espectre_log.h"

namespace espectre {

bool setup_frontend_mqtt_transport(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   IMqttTransport::CommandCallback command_callback,
                                   FrontendMqttConnectedCallback connected_callback,
                                   const char *log_tag) {
  if (transport == nullptr) {
    return false;
  }
  if (config.mqtt_host.empty()) {
    transport->shutdown();
    return false;
  }

  transport->set_command_callback(std::move(command_callback));
  transport->set_connection_callback([connected_callback = std::move(connected_callback)](bool connected) {
    if (connected_callback) {
      connected_callback(connected);
    }
  });
  if (!transport->setup(config)) {
    ESP_LOGW(log_tag != nullptr ? log_tag : "espectre.mqtt", "MQTT transport setup failed");
    return false;
  }
  return true;
}

bool publish_frontend_mqtt_message(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   const char *suffix,
                                   const std::string &payload,
                                   bool retain) {
  if (transport == nullptr || !transport->connected()) {
    return false;
  }
  (void) config;
  return transport->publish_suffix(suffix, payload, retain);
}

bool publish_frontend_mqtt_status(IMqttTransport *transport,
                                  const EspectreDeviceConfig &config,
                                  bool online,
                                  uint32_t timestamp_ms) {
  return publish_frontend_mqtt_message(
      transport, config, "status", espectre_status_payload(config, online, timestamp_ms), true);
}

bool publish_frontend_mqtt_command_result(IMqttTransport *transport,
                                          const EspectreDeviceConfig &config,
                                          const EspectreCommand &command,
                                          bool accepted,
                                          const char *message) {
  return publish_frontend_mqtt_message(transport,
                                       config,
                                       accepted ? "commands/accepted" : "commands/rejected",
                                       espectre_command_result_payload(config, command, accepted, message),
                                       false);
}

bool publish_frontend_mqtt_ota_status(IMqttTransport *transport,
                                      const EspectreDeviceConfig &config,
                                      const EspectreOtaStatus &status,
                                      uint32_t timestamp_ms) {
  return publish_frontend_mqtt_message(
      transport, config, "ota/state", espectre_ota_status_payload(config, status, timestamp_ms), false);
}

FrontendMqttCommandResult handle_frontend_mqtt_command(const std::string &payload,
                                                       IOtaService *ota_service,
                                                       const char *current_version,
                                                       const FrontendMqttCommandCapabilities &capabilities,
                                                       FrontendMqttInfoCallback info_callback,
                                                       FrontendMqttStatsCallback stats_callback,
                                                       FrontendMqttThresholdCallback threshold_callback,
                                                       FrontendMqttMotionHitsCallback motion_hits_callback,
                                                       FrontendMqttCsiTrafficModeCallback csi_traffic_mode_callback,
                                                       FrontendMqttTrafficGeneratorModeCallback traffic_generator_mode_callback,
                                                       FrontendMqttDetectorCallback detector_callback,
                                                       FrontendMqttRecalibrateCallback recalibrate_callback,
                                                       FrontendMqttOtaStatusCallback ota_status_callback,
                                                       FrontendMqttCommandsCallback commands_callback) {
  FrontendMqttCommandResult result;
  result.handled = true;
  if (!parse_espectre_command(payload, &result.command, &result.message)) {
    if (result.command.command.empty()) {
      result.command.command = "unknown";
    }
    result.accepted = false;
    return result;
  }

  if (result.command.command == "info") {
    if (!capabilities.supports_info || !info_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    info_callback();
    result.accepted = true;
    result.message = "info published";
    return result;
  }

  if (result.command.command == "commands") {
    if (!commands_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    commands_callback();
    result.accepted = true;
    result.message = "commands published";
    return result;
  }

  if (result.command.command == "stats") {
    if (!capabilities.supports_stats || !stats_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    stats_callback();
    result.accepted = true;
    result.message = "stats published";
    return result;
  }

  if (result.command.command == "set_threshold") {
    if (!capabilities.supports_threshold || !threshold_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    if (!result.command.has_threshold || !validate_runtime_threshold(result.command.threshold)) {
      result.accepted = false;
      result.message = "invalid threshold (accepted: 0.0-1.0)";
      return result;
    }
    result.accepted = threshold_callback(result.command.threshold, &result.message);
    if (result.message.empty()) {
      result.message = result.accepted ? "threshold updated" : "threshold rejected";
    }
    return result;
  }

  if (result.command.command == "set_motion_hits") {
    if (!capabilities.supports_motion_hits || !motion_hits_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    if (!result.command.has_motion_hits ||
        result.command.motion_on_hits < RUNTIME_MOTION_HITS_MIN ||
        result.command.motion_on_hits > RUNTIME_MOTION_HITS_MAX ||
        result.command.motion_off_hits < RUNTIME_MOTION_HITS_MIN ||
        result.command.motion_off_hits > RUNTIME_MOTION_HITS_MAX) {
      result.accepted = false;
      result.message = "invalid motion hits (accepted: motion_on_hits and motion_off_hits in 1-20)";
      return result;
    }
    result.accepted =
        motion_hits_callback(result.command.motion_on_hits, result.command.motion_off_hits, &result.message);
    if (result.message.empty()) {
      result.message = result.accepted ? "motion hits updated" : "motion hits rejected";
    }
    return result;
  }

  if (result.command.command == "set_csi_traffic_mode") {
    if (!capabilities.supports_traffic_control || !csi_traffic_mode_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    if (!result.command.has_csi_traffic_mode) {
      result.accepted = false;
      result.message = "invalid csi traffic mode (accepted: internal, external, pacing, and disabled)";
      return result;
    }
    result.accepted =
        csi_traffic_mode_callback(parse_csi_traffic_mode(result.command.csi_traffic_mode.c_str()), &result.message);
    if (result.message.empty()) {
      result.message = result.accepted ? "csi traffic mode updated" : "csi traffic mode rejected";
    }
    return result;
  }

  if (result.command.command == "set_traffic_generator_mode") {
    if (!capabilities.supports_traffic_control || !traffic_generator_mode_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    if (!result.command.has_traffic_generator_mode) {
      result.accepted = false;
      result.message = "invalid traffic generator mode (accepted: ping and dns)";
      return result;
    }
    result.accepted = traffic_generator_mode_callback(parse_traffic_mode(result.command.traffic_generator_mode.c_str()),
                                                      &result.message);
    if (result.message.empty()) {
      result.message = result.accepted ? "traffic generator mode updated" : "traffic generator mode rejected";
    }
    return result;
  }

  if (result.command.command == "set_detector") {
    if (!capabilities.supports_detector || !detector_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    if (!result.command.has_detector) {
      result.accepted = false;
      result.message = "invalid detector (accepted: lightweight and high_accuracy)";
      return result;
    }
    result.accepted = detector_callback(parse_detection_algorithm(result.command.detector.c_str()), &result.message);
    if (result.message.empty()) {
      result.message = result.accepted ? "detector updated" : "detector rejected";
    }
    return result;
  }

  if (result.command.command == "recalibrate") {
    if (!capabilities.supports_recalibrate || !recalibrate_callback) {
      result.accepted = false;
      result.message = "unsupported command";
      return result;
    }
    result.accepted = recalibrate_callback(&result.message);
    if (result.message.empty()) {
      result.message = result.accepted ? "recalibration started" : "recalibration rejected";
    }
    return result;
  }

  if (result.command.command == "ota_status" || result.command.command == "ota_check" || result.command.command == "ota_start") {
    if (!capabilities.supports_ota || ota_service == nullptr) {
      result.accepted = false;
      result.message = "ota unavailable";
      return result;
    }

    const std::string normalized_current_version =
        (current_version == nullptr || current_version[0] == '\0') ? "unknown" : current_version;

    if (result.command.command == "ota_status") {
      if (ota_status_callback) {
        ota_status_callback(ota_service->status());
      }
      result.accepted = true;
      result.message = "ota status published";
      return result;
    }

    if (result.command.command == "ota_check") {
      result.accepted = ota_service->start_check(normalized_current_version, result.command.ota_channel);
      result.message = result.accepted ? "ota check started" : "ota check rejected";
      return result;
    }

    result.accepted = ota_service->start_update(normalized_current_version, result.command.ota_channel);
    result.message = result.accepted ? "ota update started" : "ota update rejected";
    return result;
  }

  result.accepted = false;
  result.message = "unsupported command";
  return result;
}

}  // namespace espectre
