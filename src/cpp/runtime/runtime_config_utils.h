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

const char *threshold_mode_name(ThresholdMode mode);
const char *threshold_mode_display_name(ThresholdMode mode);
const char *traffic_mode_name(RuntimeTrafficMode mode);
const char *detection_algorithm_name(DetectionAlgorithm algorithm);
const char *subcarrier_source_name(RuntimeSubcarrierSource source);

ThresholdMode parse_threshold_mode(const char *mode);
RuntimeTrafficMode parse_traffic_mode(const char *mode);
DetectionAlgorithm parse_detection_algorithm(const char *algorithm);

RuntimeConfig make_runtime_sensing_config();
void set_manual_threshold(RuntimeConfig &config, float threshold);

}  // namespace espectre
