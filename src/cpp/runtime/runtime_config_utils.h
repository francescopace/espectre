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
  CSI_TRAFFIC_MODE,
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
bool runtime_traffic_mode_supported(RuntimeTrafficMode mode);
/** Whether a configured CSI profile can be combined with the internal source. */
bool runtime_capture_profile_supports_traffic(CsiCapturePolicy profile, RuntimeTrafficMode mode);

/** Validate the complete configuration before creating runtime state. */
RuntimeConfigError validate_runtime_config(const RuntimeConfig &config);
/** Stable diagnostic label for a configuration error. Never returns `nullptr`. */
const char *runtime_config_error_message(RuntimeConfigError error);

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
/** Name of an internal traffic generator packet, such as `ping` or `dns`. */
const char *traffic_mode_name(RuntimeTrafficMode mode);
/** Name of a CSI traffic source: `internal` or `external`. */
const char *csi_traffic_mode_name(CsiTrafficMode mode);
/** Name of a detector: `lightweight` or `high_accuracy`. */
const char *detection_algorithm_name(DetectionAlgorithm algorithm);

/** Parse a traffic generator packet name. Defaults to `RuntimeTrafficMode::PING`. */
RuntimeTrafficMode parse_traffic_mode(const char *mode);
/** Parse a CSI traffic source name. Defaults to `CsiTrafficMode::INTERNAL`. */
CsiTrafficMode parse_csi_traffic_mode(const char *mode);
/** Parse a detector name. Defaults to `DetectionAlgorithm::LIGHTWEIGHT`. */
DetectionAlgorithm parse_detection_algorithm(const char *algorithm);
/** Parse a Wi-Fi band policy name. Defaults to `WifiBandPolicy::BAND_2G`. */
WifiBandPolicy parse_wifi_band_policy(const char *policy);
/** @} */

}  // namespace espectre
