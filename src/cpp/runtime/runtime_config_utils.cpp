#include "runtime_config_utils.h"

#include <cmath>

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
      return "Min (P100)";
    case ThresholdMode::AUTO:
    default:
      return "Auto (P95x1.1)";
  }
}

const char *traffic_mode_name(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? "ping" : "dns";
}

const char *gain_lock_mode_name(RuntimeGainLockMode mode) {
  switch (mode) {
    case RuntimeGainLockMode::ENABLED:
      return "enabled";
    case RuntimeGainLockMode::DISABLED:
      return "disabled";
    case RuntimeGainLockMode::AUTO:
    default:
      return "auto";
  }
}

const char *detection_algorithm_name(DetectionAlgorithm algorithm) {
  return algorithm == DetectionAlgorithm::ML ? "ml" : "mvs";
}

const char *subcarrier_source_name(RuntimeSubcarrierSource source) {
  switch (source) {
    case RuntimeSubcarrierSource::FIXED_DEFAULT:
    default:
      return "fixed";
  }
}

void set_manual_threshold(RuntimeConfig &config, float threshold) {
  config.segmentation_threshold = threshold;
  config.threshold_mode = ThresholdMode::MANUAL;
}

}  // namespace espectre
}  // namespace esphome
