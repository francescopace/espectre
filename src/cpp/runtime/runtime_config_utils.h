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
#include "threshold.h"
#include "utils.h"

namespace espectre {

bool validate_runtime_threshold(float threshold);
bool validate_runtime_threshold_for_algorithm(float threshold, DetectionAlgorithm algorithm);
bool validate_runtime_float(float value, float min_value, float max_value);
bool validate_runtime_uint32(uint32_t value, uint32_t min_value, uint32_t max_value);
bool validate_runtime_uint8(uint8_t value, uint8_t min_value, uint8_t max_value);

const char *runtime_profile_name(RuntimeProfile profile);
const char *wifi_band_policy_name(WifiBandPolicy policy);

const char *traffic_mode_name(RuntimeTrafficMode mode);
const char *detection_algorithm_name(DetectionAlgorithm algorithm);
const char *subcarrier_source_name(RuntimeSubcarrierSource source);

RuntimeTrafficMode parse_traffic_mode(const char *mode);
DetectionAlgorithm parse_detection_algorithm(const char *algorithm);
WifiBandPolicy parse_wifi_band_policy(const char *policy);

RuntimeConfig make_runtime_sensing_config();

}  // namespace espectre
