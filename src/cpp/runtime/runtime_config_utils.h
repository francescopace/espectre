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

#include "runtime_interface.h"

namespace espectre {

/** Machine-readable reason a `RuntimeConfig` cannot be applied. */
enum class RuntimeConfigError : uint8_t {
  NONE = 0,
  RUNTIME_PROFILE,
  WIFI_BAND_POLICY,
  DETECTION_ALGORITHM,
  SEGMENTATION_THRESHOLD,
  SEGMENTATION_WINDOW_SIZE_MS,
  CSI_TARGET_PPS,
  TRAFFIC_GENERATOR_MODE,
  CSI_TRAFFIC_MODE,
  CSI_TRAFFIC_UDP_PORT,
  CSI_TRAFFIC_MULTICAST_GROUP,
  PUBLISH_INTERVAL_MS,
  EVALUATION_INTERVAL_MS,
  MOTION_HITS,
  LOWPASS_CUTOFF,
  HAMPEL_WINDOW,
  HAMPEL_THRESHOLD,
};

bool validate_runtime_threshold(float threshold);
bool validate_runtime_threshold_for_algorithm(float threshold, DetectionAlgorithm algorithm);
bool validate_runtime_float(float value, float min_value, float max_value);
bool validate_runtime_uint32(uint32_t value, uint32_t min_value, uint32_t max_value);
bool validate_runtime_uint8(uint8_t value, uint8_t min_value, uint8_t max_value);

/** Validate the complete configuration before creating runtime state. */
RuntimeConfigError validate_runtime_config(const RuntimeConfig &config);
/** Stable diagnostic label for a configuration error. Never returns `nullptr`. */
const char *runtime_config_error_message(RuntimeConfigError error);

const char *runtime_profile_name(RuntimeProfile profile);
const char *wifi_band_policy_name(WifiBandPolicy policy);

const char *traffic_mode_name(RuntimeTrafficMode mode);
const char *csi_traffic_mode_name(CsiTrafficMode mode);
bool csi_traffic_mode_is_sensing_control(CsiTrafficMode mode);
CsiTrafficMode normalize_sensing_csi_traffic_mode(CsiTrafficMode mode);
const char *detection_algorithm_name(DetectionAlgorithm algorithm);
const char *subcarrier_source_name(RuntimeSubcarrierSource source);

RuntimeTrafficMode parse_traffic_mode(const char *mode);
CsiTrafficMode parse_csi_traffic_mode(const char *mode);
DetectionAlgorithm parse_detection_algorithm(const char *algorithm);
WifiBandPolicy parse_wifi_band_policy(const char *policy);

RuntimeConfig make_runtime_sensing_config();

}  // namespace espectre
