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
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "espectre_protocol.h"
#include "ota_service.h"
#include "runtime_config_utils.h"

namespace espectre {

struct DeviceConfigCommandResult {
  bool handled{false};
  bool accepted{false};
  bool config_changed{false};
  EspectreDeviceConfig config{};
  std::string message;
};

using DeviceConfigClearHandler = std::function<bool(EspectreDeviceConfig *cleared_config, std::string *message)>;
using DeviceConfigUpdateHandler = std::function<bool(EspectreDeviceConfig *updated_config, std::string *message)>;

using FrontendInfoCallback = std::function<void()>;
using FrontendStatsCallback = std::function<void()>;
using FrontendDeviceLabelCallback = std::function<bool(const std::string &device_label, std::string *message)>;
using FrontendThresholdCallback = std::function<bool(float threshold, std::string *message)>;
using FrontendMotionHitsCallback =
    std::function<bool(uint8_t motion_on_hits, uint8_t motion_off_hits, std::string *message)>;
using FrontendCsiTrafficModeCallback = std::function<bool(CsiTrafficMode mode, std::string *message)>;
using FrontendTrafficGeneratorModeCallback = std::function<bool(RuntimeTrafficMode mode, std::string *message)>;
using FrontendDetectorCallback = std::function<bool(DetectionAlgorithm algorithm, std::string *message)>;
using FrontendRecalibrateCallback = std::function<bool(std::string *message)>;
using FrontendOtaStatusCallback = std::function<void(const EspectreOtaStatus &status)>;
using FrontendCommandsCallback = std::function<void()>;
using FrontendWifiConfigCallback =
    std::function<bool(const EspectreCommand &command, bool clear, std::string *message)>;
using FrontendMqttConfigCallback =
    std::function<bool(const EspectreCommand &command, bool clear, std::string *message)>;
using FrontendSensingControlCallback = std::function<bool(bool enabled, std::string *message)>;

struct FrontendCommandCapabilities {
  bool supports_info{true};
  bool supports_stats{false};
  bool supports_device_config{false};
  bool supports_wifi_config{false};
  bool supports_mqtt_config{false};
  bool supports_sensing_control{false};
  bool supports_threshold{false};
  bool supports_motion_hits{false};
  bool supports_traffic_control{false};
  bool supports_detector{false};
  bool supports_recalibrate{false};
  bool supports_ota{false};
};

struct FrontendCommandResult {
  bool handled{false};
  bool accepted{false};
  EspectreCommand command{};
  std::string message;
};

DeviceConfigCommandResult handle_device_config_command(const std::string &command,
                                                       const EspectreDeviceConfig &current_config,
                                                       DeviceConfigClearHandler clear_handler,
                                                       DeviceConfigUpdateHandler update_handler);

FrontendCommandResult handle_frontend_command(const EspectreCommand &command,
                                              IOtaService *ota_service,
                                              const char *current_version,
                                              const FrontendCommandCapabilities &capabilities,
                                              FrontendInfoCallback info_callback,
                                              FrontendStatsCallback stats_callback,
                                              FrontendDeviceLabelCallback device_label_callback,
                                              FrontendThresholdCallback threshold_callback,
                                              FrontendMotionHitsCallback motion_hits_callback,
                                              FrontendCsiTrafficModeCallback csi_traffic_mode_callback,
                                              FrontendTrafficGeneratorModeCallback traffic_generator_mode_callback,
                                              FrontendDetectorCallback detector_callback,
                                              FrontendRecalibrateCallback recalibrate_callback,
                                              FrontendOtaStatusCallback ota_status_callback,
                                              FrontendCommandsCallback commands_callback = {},
                                              FrontendWifiConfigCallback wifi_config_callback = {},
                                              FrontendMqttConfigCallback mqtt_config_callback = {},
                                              FrontendSensingControlCallback sensing_control_callback = {});

}  // namespace espectre
