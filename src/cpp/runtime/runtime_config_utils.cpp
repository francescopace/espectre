/*
 * ESPectre - Runtime Config Utils
 *
 * Helpers for normalizing and applying runtime configuration.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "runtime_config_utils.h"

#include <cmath>
#include <cstring>

namespace espectre {

bool validate_runtime_threshold(float threshold) {
  return std::isfinite(threshold) && threshold >= RUNTIME_THRESHOLD_MIN && threshold <= RUNTIME_THRESHOLD_MAX;
}

bool validate_runtime_threshold_for_algorithm(float threshold, DetectionAlgorithm algorithm) {
  return std::isfinite(threshold) && threshold >= RUNTIME_THRESHOLD_MIN &&
         threshold <= runtime_threshold_max(algorithm);
}

bool validate_runtime_float(float value, float min_value, float max_value) {
  return std::isfinite(value) && value >= min_value && value <= max_value;
}

bool validate_runtime_uint32(uint32_t value, uint32_t min_value, uint32_t max_value) {
  return value >= min_value && value <= max_value;
}

bool validate_runtime_uint8(uint8_t value, uint8_t min_value, uint8_t max_value) {
  return value >= min_value && value <= max_value;
}

const char *runtime_profile_name(RuntimeProfile profile) {
  return profile == RuntimeProfile::STREAM ? "stream" : "sensing";
}

const char *wifi_band_policy_name(WifiBandPolicy policy) {
  switch (policy) {
    case WifiBandPolicy::BAND_5G:
      return "5g";
    case WifiBandPolicy::AUTO:
      return "auto";
    case WifiBandPolicy::BAND_2G:
    default:
      return "2g";
  }
}

const char *traffic_mode_name(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME
                                          : RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME;
}

const char *detection_algorithm_name(DetectionAlgorithm algorithm) {
  switch (algorithm) {
    case DetectionAlgorithm::ML:
      return RUNTIME_DETECTION_ALGORITHM_ML_NAME;
    case DetectionAlgorithm::CLASSIC:
    default:
      return RUNTIME_DETECTION_ALGORITHM_CLASSIC_NAME;
  }
}

const char *subcarrier_source_name(RuntimeSubcarrierSource source) {
  switch (source) {
    case RuntimeSubcarrierSource::FIXED_DEFAULT:
    default:
      return "fixed";
  }
}

RuntimeTrafficMode parse_traffic_mode(const char *mode) {
  return (mode != nullptr && std::strcmp(mode, RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME) == 0)
             ? RuntimeTrafficMode::PING
             : RuntimeTrafficMode::DNS;
}

DetectionAlgorithm parse_detection_algorithm(const char *algorithm) {
  return (algorithm != nullptr && std::strcmp(algorithm, RUNTIME_DETECTION_ALGORITHM_ML_NAME) == 0)
             ? DetectionAlgorithm::ML
             : DetectionAlgorithm::CLASSIC;
}

WifiBandPolicy parse_wifi_band_policy(const char *policy) {
  if (policy != nullptr && std::strcmp(policy, "5g") == 0) {
    return WifiBandPolicy::BAND_5G;
  }
  if (policy != nullptr && std::strcmp(policy, "auto") == 0) {
    return WifiBandPolicy::AUTO;
  }
  return WifiBandPolicy::BAND_2G;
}

RuntimeConfig make_runtime_sensing_config() { return RuntimeConfig{}; }

}  // namespace espectre
