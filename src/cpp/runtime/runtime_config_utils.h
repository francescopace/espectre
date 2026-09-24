/*
 * ESPectre - Runtime Config Utils
 *
 * Helpers for normalizing and applying runtime configuration.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "runtime_config.h"
#include "runtime_snapshot.h"

namespace espectre {

/** Machine-readable reason a `RuntimeConfig` cannot be applied. */
enum class RuntimeConfigError : uint8_t {
  NONE = 0,
  WIFI_BAND_POLICY,
  DETECTION_ALGORITHM,
  SEGMENTATION_THRESHOLD,
  SEGMENTATION_WINDOW_SIZE_MS,
  CSI_TARGET_PPS,
  TRAFFIC_GENERATOR_MODE,
  CSI_TRAFFIC_UDP_PORT,
  CSI_TRAFFIC_MULTICAST_GROUP,
  EVALUATION_INTERVAL_MS,
  MOTION_HITS,
  LOWPASS_CUTOFF,
  HAMPEL_WINDOW,
  HAMPEL_THRESHOLD,
  TRAFFIC_GENERATOR_TARGET_IP,
  CSI_CAPTURE_PROFILE,
  CSI_CAPTURE_PROFILE_TRAFFIC,
};

/** Whether a threshold is finite and inside the range shared by every detector. */
bool validate_runtime_threshold(float threshold);
/** Whether a threshold is finite and inside the range of one detector. */
bool validate_runtime_threshold_for_algorithm(float threshold, DetectionAlgorithm algorithm);
/** Whether this build target supports the internal traffic source; host builds accept every valid mode. */
bool runtime_traffic_generator_mode_supported(TrafficGeneratorMode mode);
/** Whether a configured CSI profile can be combined with the internal source. */
bool runtime_capture_profile_supports_traffic(CsiCapturePolicy profile, TrafficGeneratorMode mode);

/** Validate the complete configuration before creating runtime state. */
RuntimeConfigError validate_runtime_config(const RuntimeConfig &config);
/** Stable diagnostic label for a configuration error. Never returns `nullptr`. */
const char *runtime_config_error_message(RuntimeConfigError error);

/**
 * A partial change to the live sensing controls.
 *
 * Each `has_*` flag marks a field to change; the others keep their value.
 * Validate the whole change before applying any field, so a request either
 * applies completely or not at all.
 */
struct RuntimeControlUpdate {
  bool has_detection_algorithm{false};
  DetectionAlgorithm detection_algorithm{DetectionAlgorithm::LIGHTWEIGHT};
  bool has_threshold{false};
  float threshold{0.0f};
  bool has_motion_hits{false};
  uint8_t motion_on_hits{0U};
  uint8_t motion_off_hits{0U};
  bool has_traffic_generator_mode{false};
  TrafficGeneratorMode traffic_generator_mode{TrafficGeneratorMode::PING};
};

/**
 * Return `config` with `update` applied, as the controller setters apply it.
 *
 * Fields apply in the order detector, threshold, motion hits, then traffic
 * generator mode. Switching to a different detector also adopts that
 * detector's default threshold unless the update sets one. Pass the result to
 * validate_runtime_config() to check the change before applying it.
 */
RuntimeConfig apply_runtime_control_update(RuntimeConfig config, const RuntimeControlUpdate &update);

/** Resolve the internal traffic destination in network byte order; empty uses the gateway, and invalid IPv4 returns zero. */
uint32_t runtime_traffic_target_addr(const RuntimeConfig &config, uint32_t gateway_addr);

/**
 * @name Wire names
 * Stable names used by the protocol, Kconfig, and logs. The `*_name()`
 * functions never return `nullptr`; the `parse_*()` functions return the
 * default value for an unknown or `nullptr` name.
 * @{
 */
/** Name of a Wi-Fi band policy: `2g`, `5g`, or `auto`. */
const char *wifi_band_policy_name(WifiBandPolicy policy);
/** Name of a traffic generator mode, such as `ping`, `dns`, or `external`. */
const char *traffic_generator_mode_name(TrafficGeneratorMode mode);
/** Name of a detector: `lightweight` or `high_accuracy`. */
const char *detection_algorithm_name(DetectionAlgorithm algorithm);

/** Parse a traffic generator mode name. Defaults to `TrafficGeneratorMode::PING`. */
TrafficGeneratorMode parse_traffic_generator_mode(const char *mode);
/** Parse a detector name. Defaults to `DetectionAlgorithm::LIGHTWEIGHT`. */
DetectionAlgorithm parse_detection_algorithm(const char *algorithm);
/** Parse a Wi-Fi band policy name. Defaults to `WifiBandPolicy::BAND_2G`. */
WifiBandPolicy parse_wifi_band_policy(const char *policy);
/** @} */

}  // namespace espectre
