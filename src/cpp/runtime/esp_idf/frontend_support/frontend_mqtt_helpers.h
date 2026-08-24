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
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "frontend_control_helpers.h"
#include "mqtt_transport.h"

namespace espectre {

using FrontendMqttConnectedCallback = std::function<void(bool)>;
using FrontendMqttInfoCallback = FrontendInfoCallback;
using FrontendMqttStatsCallback = FrontendStatsCallback;
using FrontendMqttDeviceLabelCallback = FrontendDeviceLabelCallback;
using FrontendMqttThresholdCallback = FrontendThresholdCallback;
using FrontendMqttMotionHitsCallback = FrontendMotionHitsCallback;
using FrontendMqttCsiTrafficModeCallback = FrontendCsiTrafficModeCallback;
using FrontendMqttTrafficGeneratorModeCallback = FrontendTrafficGeneratorModeCallback;
using FrontendMqttDetectorCallback = FrontendDetectorCallback;
using FrontendMqttRecalibrateCallback = FrontendRecalibrateCallback;
using FrontendMqttOtaStatusCallback = FrontendOtaStatusCallback;
using FrontendMqttCommandsCallback = FrontendCommandsCallback;
using FrontendMqttCommandCapabilities = FrontendCommandCapabilities;
using FrontendMqttCommandResult = FrontendCommandResult;

bool setup_frontend_mqtt_transport(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   IMqttTransport::CommandCallback command_callback,
                                   FrontendMqttConnectedCallback connected_callback,
                                   const char *log_tag);

bool publish_frontend_mqtt_message(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   const char *suffix,
                                   const std::string &payload,
                                   bool retain);

bool publish_frontend_mqtt_status(IMqttTransport *transport,
                                  const EspectreDeviceConfig &config,
                                  bool online,
                                  uint32_t timestamp_ms);

bool publish_frontend_mqtt_command_result(IMqttTransport *transport,
                                          const EspectreDeviceConfig &config,
                                          const EspectreCommand &command,
                                          bool accepted,
                                          const char *message);

bool publish_frontend_mqtt_ota_status(IMqttTransport *transport,
                                      const EspectreDeviceConfig &config,
                                      const EspectreOtaStatus &status,
                                      uint32_t timestamp_ms);

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
                                                       FrontendMqttCommandsCallback commands_callback = {});

}  // namespace espectre
