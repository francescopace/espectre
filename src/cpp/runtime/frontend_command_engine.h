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

using FrontendReadPayloadCallback = std::function<std::string(const EspectreCommand &command)>;
using FrontendDeviceLabelCallback = std::function<bool(const std::string &device_label, std::string *message)>;
using FrontendThresholdCallback = std::function<bool(float threshold, std::string *message)>;
using FrontendMotionHitsCallback =
    std::function<bool(uint8_t motion_on_hits, uint8_t motion_off_hits, std::string *message)>;
using FrontendCsiTrafficModeCallback = std::function<bool(CsiTrafficMode mode, std::string *message)>;
using FrontendTrafficGeneratorModeCallback = std::function<bool(RuntimeTrafficMode mode, std::string *message)>;
using FrontendDetectorCallback = std::function<bool(DetectionAlgorithm algorithm, std::string *message)>;
using FrontendRecalibrateCallback = std::function<bool(std::string *message)>;
using FrontendWifiBssidCallback =
    std::function<bool(const EspectreCommand &command, std::string *message)>;
using FrontendMqttConfigCallback =
    std::function<bool(const EspectreCommand &command, bool clear, std::string *message)>;
using FrontendSensingControlCallback = std::function<bool(bool enabled, std::string *message)>;
using FrontendRawStreamCallback = std::function<bool(const EspectreCommand &command,
                                                      const struct FrontendCommandContext &context,
                                                      std::string *code,
                                                      std::string *message,
                                                      std::string *data_json)>;

using FrontendCommandCapabilities = EspectreCapabilityProfile;

enum class FrontendCommandChange : uint8_t {
  NONE = 0U,
  STATUS = 1U << 0U,
  INFO = 1U << 1U,
  CONFIG = 1U << 2U,
  OTA_STATUS = 1U << 3U,
};

inline FrontendCommandChange operator|(FrontendCommandChange lhs, FrontendCommandChange rhs) {
  return static_cast<FrontendCommandChange>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

enum class FrontendCommandOrigin : uint8_t {
  DIRECT = 0U,
  MQTT,
  ESPHOME,
  MATTER,
};

struct FrontendCommandContext {
  FrontendCommandOrigin origin{FrontendCommandOrigin::DIRECT};
  /** Opaque request identity used only to complete deferred Direct responses. */
  uint64_t connection_token{0U};
  /** Normalized bearer token supplied by the Direct HTTP transport. */
  std::string authorization;
};

struct FrontendCommandResult {
  bool handled{false};
  bool accepted{false};
  EspectreCommand command{};
  std::string code{"internal_error"};
  std::string message;
  std::string data_json;
  FrontendCommandChange changes{FrontendCommandChange::NONE};
};

DeviceConfigCommandResult handle_device_config_command(const std::string &command,
                                                       const EspectreDeviceConfig &current_config,
                                                       DeviceConfigClearHandler clear_handler,
                                                       DeviceConfigUpdateHandler update_handler);

bool frontend_command_allowed_during_raw_collection(const std::string &command);

class FrontendCommandEngine {
 public:
  FrontendCommandResult execute(const EspectreCommand &command,
                                const FrontendCommandContext &context,
                                IOtaService *ota_service,
                                const char *current_version,
                                const FrontendCommandCapabilities &capabilities,
                                FrontendReadPayloadCallback read_payload_callback,
                                FrontendDeviceLabelCallback device_label_callback = {},
                                FrontendThresholdCallback threshold_callback = {},
                                FrontendMotionHitsCallback motion_hits_callback = {},
                                FrontendCsiTrafficModeCallback csi_traffic_mode_callback = {},
                                FrontendTrafficGeneratorModeCallback traffic_generator_mode_callback = {},
                                FrontendDetectorCallback detector_callback = {},
                                FrontendRecalibrateCallback recalibrate_callback = {},
                                FrontendWifiBssidCallback wifi_bssid_callback = {},
                                FrontendMqttConfigCallback mqtt_config_callback = {},
                                FrontendSensingControlCallback sensing_control_callback = {},
                                FrontendRawStreamCallback raw_stream_callback = {}) const;
};

}  // namespace espectre
