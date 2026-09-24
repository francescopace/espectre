/*
 * ESPectre - Frontend Command Engine
 *
 * Dispatches parsed protocol commands to frontend callbacks.
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
#include "runtime_config_utils.h"

namespace espectre {

/** Map a canonical command parse failure to its stable result code. */
const char *frontend_command_parse_error_code(const std::string &error);

using FrontendReadPayloadCallback = std::function<std::string(const EspectreCommand &command)>;
using FrontendDeviceLabelCallback = std::function<bool(const std::string &device_label, std::string *message)>;
using FrontendThresholdCallback = std::function<bool(float threshold, std::string *message)>;
using FrontendMotionHitsCallback =
    std::function<bool(uint8_t motion_on_hits, uint8_t motion_off_hits, std::string *message)>;
using FrontendTrafficGeneratorModeCallback = std::function<bool(TrafficGeneratorMode mode, std::string *message)>;
using FrontendDetectorCallback = std::function<bool(DetectionAlgorithm algorithm, std::string *message)>;
using FrontendRecalibrateCallback = std::function<bool(std::string *message)>;
using FrontendWifiBssidCallback =
    std::function<bool(const EspectreCommand &command, std::string *message)>;
using FrontendMqttConfigCallback =
    std::function<bool(const EspectreCommand &command, bool clear, std::string *message)>;
using FrontendSensingControlCallback = std::function<bool(bool enabled, std::string *message)>;
/**
 * Check every sensing field of one `update_sensing` before any is applied.
 *
 * Frontends built on RuntimeFrontendController pass
 * `RuntimeFrontendController::validate_control_update()`.
 */
using FrontendSensingPreflightCallback =
    std::function<bool(const RuntimeControlUpdate &update, std::string *message)>;

using FrontendCommandCapabilities = EspectreCapabilityProfile;

enum class FrontendCommandChange : uint8_t {
  NONE = 0U,
  HEALTH = 1U << 0U,
  DEVICE = 1U << 1U,
  SENSING = 1U << 2U,
  WIFI = 1U << 3U,
  MQTT = 1U << 4U,
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
  /** Opaque identity of the originating Direct connection; zero otherwise. */
  uint64_t connection_token{0U};
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

bool frontend_command_allowed_during_raw_collection(
    const std::string &command, const EspectreProtocolExtension *extension = nullptr);

class FrontendCommandEngine {
 public:
  /** Execute a successfully parsed command. Call a protocol parser first;
   * this dispatcher checks capabilities and operational state, not parameters.
   */
  FrontendCommandResult execute(const EspectreCommand &command,
                                const FrontendCommandContext &context,
                                const FrontendCommandCapabilities &capabilities,
                                FrontendReadPayloadCallback read_payload_callback,
                                FrontendDeviceLabelCallback device_label_callback = {},
                                FrontendThresholdCallback threshold_callback = {},
                                FrontendMotionHitsCallback motion_hits_callback = {},
                                FrontendTrafficGeneratorModeCallback traffic_generator_mode_callback = {},
                                FrontendDetectorCallback detector_callback = {},
                                FrontendRecalibrateCallback recalibrate_callback = {},
                                FrontendWifiBssidCallback wifi_bssid_callback = {},
                                FrontendMqttConfigCallback mqtt_config_callback = {},
                                FrontendSensingControlCallback sensing_control_callback = {},
                                FrontendSensingPreflightCallback sensing_preflight_callback = {}) const;
};

}  // namespace espectre
