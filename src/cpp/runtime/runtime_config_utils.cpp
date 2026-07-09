#include "runtime_config_utils.h"

#include <cmath>
#include <cstring>

namespace esphome {
namespace espectre {

bool validate_runtime_threshold(float threshold) {
  return std::isfinite(threshold) && threshold >= RUNTIME_THRESHOLD_MIN && threshold <= RUNTIME_THRESHOLD_MAX;
}

const char *threshold_mode_name(ThresholdMode mode) {
  switch (mode) {
    case ThresholdMode::MANUAL:
      return "manual";
    case ThresholdMode::MIN:
      return "min";
    case ThresholdMode::AUTO:
    default:
      return "auto";
  }
}

const char *threshold_mode_display_name(ThresholdMode mode) {
  switch (mode) {
    case ThresholdMode::MANUAL:
      return "Manual";
    case ThresholdMode::MIN:
      return "Min (x1.0)";
    case ThresholdMode::AUTO:
    default:
      return "Auto (adaptive)";
  }
}

const char *traffic_mode_name(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? "ping" : "dns";
}

const char *detection_algorithm_name(DetectionAlgorithm algorithm) {
  switch (algorithm) {
    case DetectionAlgorithm::ML:
      return "ml";
    case DetectionAlgorithm::CLASSIC:
    default:
      return "classic";
  }
}

const char *subcarrier_source_name(RuntimeSubcarrierSource source) {
  switch (source) {
    case RuntimeSubcarrierSource::FIXED_DEFAULT:
    default:
      return "fixed";
  }
}

ThresholdMode parse_threshold_mode(const char *mode) {
  if (mode != nullptr && std::strcmp(mode, "min") == 0) {
    return ThresholdMode::MIN;
  }
  return ThresholdMode::AUTO;
}

RuntimeTrafficMode parse_traffic_mode(const char *mode) {
  return (mode != nullptr && std::strcmp(mode, "ping") == 0) ? RuntimeTrafficMode::PING : RuntimeTrafficMode::DNS;
}

DetectionAlgorithm parse_detection_algorithm(const char *algorithm) {
  return (algorithm != nullptr && std::strcmp(algorithm, "ml") == 0)
             ? DetectionAlgorithm::ML
             : DetectionAlgorithm::CLASSIC;
}

void set_manual_threshold(RuntimeConfig &config, float threshold) {
  config.segmentation_threshold = threshold;
  config.threshold_mode = ThresholdMode::MANUAL;
}

}  // namespace espectre
}  // namespace esphome
