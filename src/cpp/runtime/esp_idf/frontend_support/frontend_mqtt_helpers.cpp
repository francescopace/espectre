/*
 * ESPectre - Frontend MQTT Helpers
 *
 * Sets up frontend MQTT transport and maps MQTT payloads to the shared
 * frontend command dispatcher.
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
                                                       FrontendMqttDeviceLabelCallback device_label_callback,
                                                       FrontendMqttThresholdCallback threshold_callback,
                                                       FrontendMqttMotionHitsCallback motion_hits_callback,
                                                       FrontendMqttCsiTrafficModeCallback csi_traffic_mode_callback,
                                                       FrontendMqttTrafficGeneratorModeCallback traffic_generator_mode_callback,
                                                       FrontendMqttDetectorCallback detector_callback,
                                                       FrontendMqttRecalibrateCallback recalibrate_callback,
                                                       FrontendMqttOtaStatusCallback ota_status_callback,
                                                       FrontendMqttCommandsCallback commands_callback) {
  EspectreCommand command;
  std::string message;
  if (!parse_espectre_command(payload, &command, &message)) {
    FrontendMqttCommandResult result;
    result.handled = true;
    result.command = std::move(command);
    if (result.command.command.empty()) {
      result.command.command = "unknown";
    }
    result.message = std::move(message);
    return result;
  }
  return handle_frontend_command(command,
                                 ota_service,
                                 current_version,
                                 capabilities,
                                 std::move(info_callback),
                                 std::move(stats_callback),
                                 std::move(device_label_callback),
                                 std::move(threshold_callback),
                                 std::move(motion_hits_callback),
                                 std::move(csi_traffic_mode_callback),
                                 std::move(traffic_generator_mode_callback),
                                 std::move(detector_callback),
                                 std::move(recalibrate_callback),
                                 std::move(ota_status_callback),
                                 std::move(commands_callback));
}

}  // namespace espectre
