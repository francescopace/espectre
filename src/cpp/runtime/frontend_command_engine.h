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

/**
 * @name Command callbacks
 * Each callback applies one kind of change and returns false, with a reason
 * in `message`, when it cannot. They run on the task that calls
 * FrontendCommandEngine::execute().
 * @{
 */

/**
 * Build the JSON data for a read command such as `device` or `sensing`.
 * Return an empty string when the data is unavailable.
 */
using FrontendReadPayloadCallback = std::function<std::string(const EspectreCommand &command)>;
/** Apply a new user-facing device label; empty clears it. */
using FrontendDeviceLabelCallback = std::function<bool(const std::string &device_label, std::string *message)>;
/** Apply a validated threshold on the 0..1 scale. */
using FrontendThresholdCallback = std::function<bool(float threshold, std::string *message)>;
/** Apply validated motion hit counts. */
using FrontendMotionHitsCallback =
    std::function<bool(uint8_t motion_on_hits, uint8_t motion_off_hits, std::string *message)>;
/** Switch the traffic generator mode. */
using FrontendTrafficGeneratorModeCallback = std::function<bool(TrafficGeneratorMode mode, std::string *message)>;
/** Switch the active detector. */
using FrontendDetectorCallback = std::function<bool(DetectionAlgorithm algorithm, std::string *message)>;
/** Start a recalibration; false reports the `busy` result code. */
using FrontendRecalibrateCallback = std::function<bool(std::string *message)>;
/**
 * Handle `scan_wifi`, `set_wifi_bssid`, `clear_wifi_bssid`, or
 * `clear_wifi_credentials`, as named by `command.command`.
 */
using FrontendWifiBssidCallback =
    std::function<bool(const EspectreCommand &command, std::string *message)>;
/** Apply `update_mqtt`, or clear the broker settings when `clear` is true. */
using FrontendMqttConfigCallback =
    std::function<bool(const EspectreCommand &command, bool clear, std::string *message)>;
/** Arm or disarm the sensing services. */
using FrontendSensingControlCallback = std::function<bool(bool enabled, std::string *message)>;
/**
 * Check every sensing field of one `update_sensing` before any is applied.
 *
 * Frontends built on RuntimeFrontendController pass
 * `RuntimeFrontendController::validate_control_update()`.
 */
using FrontendSensingPreflightCallback =
    std::function<bool(const RuntimeControlUpdate &update, std::string *message)>;
/** @} */

/** The advertised surface a command must belong to before it is executed. */
using FrontendCommandCapabilities = EspectreCapabilityProfile;

/**
 * Protocol resources a command changed, as a bit set.
 *
 * Transports use it to republish the matching state after an accepted change.
 */
enum class FrontendCommandChange : uint8_t {
  NONE = 0U,
  /**
   * Health output changed. The engine never reports it; frontends pass it to
   * `RuntimeDirectHttpBridge::publish_changes()` to republish health.
   */
  HEALTH = 1U << 0U,
  /** Device label or identity. */
  DEVICE = 1U << 1U,
  /** Sensing configuration or state. */
  SENSING = 1U << 2U,
  /** Wi-Fi station configuration. */
  WIFI = 1U << 3U,
  /** Broker configuration. */
  MQTT = 1U << 4U,
};

/** Combine change flags. */
inline FrontendCommandChange operator|(FrontendCommandChange lhs, FrontendCommandChange rhs) {
  return static_cast<FrontendCommandChange>(static_cast<uint8_t>(lhs) | static_cast<uint8_t>(rhs));
}

/** Transport or frontend surface a command arrived through. */
enum class FrontendCommandOrigin : uint8_t {
  /** Direct HTTP. */
  DIRECT = 0U,
  /**
   * MQTT command topic. Only `update_device`, `update_sensing`,
   * `recalibrate`, and `read_diagnostics` are accepted from it.
   */
  MQTT,
  /** An ESPHome entity or service. */
  ESPHOME,
  /** A Matter attribute or command. */
  MATTER,
};

/** Where a command came from. */
struct FrontendCommandContext {
  FrontendCommandOrigin origin{FrontendCommandOrigin::DIRECT};
  /** Opaque identity of the originating Direct connection; zero otherwise. */
  uint64_t connection_token{0U};
};

/**
 * Outcome of FrontendCommandEngine::execute(), ready for
 * `espectre_command_result_payload()`.
 */
struct FrontendCommandResult {
  /** Always true for results returned by the engine. */
  bool handled{false};
  /** Whether the command was applied or its data returned. */
  bool accepted{false};
  /** The command that was executed. */
  EspectreCommand command{};
  /**
   * Stable result code: `ok`, `unsupported`, `forbidden`, `invalid_params`,
   * `unavailable`, or `busy`.
   */
  std::string code{"internal_error"};
  /** Human-readable outcome. */
  std::string message;
  /** JSON data returned by a read command; empty otherwise. */
  std::string data_json;
  /**
   * Resources the command changed. A rejected `update_sensing` still reports
   * the fields applied before the rejection, so republish whenever this is
   * not `NONE`, whether or not the command was accepted.
   */
  FrontendCommandChange changes{FrontendCommandChange::NONE};
};

/**
 * Whether a command may run while raw CSI collection is active.
 *
 * Reads and configuration commands that do not touch sensing are allowed;
 * extension routes declare their own policy.
 */
bool frontend_command_allowed_during_raw_collection(
    const std::string &command, const EspectreProtocolExtension *extension = nullptr);

/**
 * Dispatches parsed protocol commands to frontend callbacks.
 *
 * One engine serves every transport, so Direct HTTP, MQTT, and frontend
 * surfaces apply the same capability checks and report the same result codes.
 * The engine is stateless: frontends supply the capabilities and a callback
 * for each kind of change they support. A command whose capability is not
 * advertised, or whose callback is empty, is rejected as `unsupported`.
 */
class FrontendCommandEngine {
 public:
  /**
   * Execute a successfully parsed command. Call a protocol parser first;
   * this dispatcher checks capabilities and operational state, not parameters.
   *
   * Read commands return the data from `read_payload_callback`.
   * `update_sensing` first passes all its fields to
   * `sensing_preflight_callback` and rejects the request as `invalid_params`
   * if that fails, so nothing changes. It then applies the fields in order:
   * detector, threshold, motion hits, traffic generator mode, then sensing
   * state. A backend refusal after the preflight stops there; fields applied
   * before it stay applied and are reported in
   * `FrontendCommandResult::changes`. Without a preflight callback, only
   * capabilities are checked up front. Extension
   * commands are not dispatched here; the application handles them.
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
