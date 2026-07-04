#pragma once

#include "runtime_interface.h"
#include "utils.h"

namespace esphome {
namespace espectre {

constexpr float RUNTIME_THRESHOLD_MIN = 0.0f;
constexpr float RUNTIME_THRESHOLD_MAX = SEGMENTATION_MAX_THRESHOLD;

bool validate_runtime_threshold(float threshold);

const char *threshold_mode_name(ThresholdMode mode);
const char *threshold_mode_display_name(ThresholdMode mode);
const char *traffic_mode_name(RuntimeTrafficMode mode);
const char *detection_algorithm_name(DetectionAlgorithm algorithm);
const char *subcarrier_source_name(RuntimeSubcarrierSource source);

ThresholdMode parse_threshold_mode(const char *mode);
RuntimeTrafficMode parse_traffic_mode(const char *mode);
DetectionAlgorithm parse_detection_algorithm(const char *algorithm);

void set_manual_threshold(RuntimeConfig &config, float threshold);

}  // namespace espectre
}  // namespace esphome
