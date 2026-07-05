/*
 * ESPectre - Adaptive Threshold Calculator
 * 
 * Calculates adaptive threshold from calibration baseline values.
 * Called after calibration to compute the detection threshold.
 * 
 * MVS Formula: threshold = max(cal_values) x factor
 * 
 * Modes:
 * - "auto": max x 1.3 (default, lower false positives on no-gain-lock captures)
 * - "min": max x 1.0 (maximum sensitivity, may have FP)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace esphome {
namespace espectre {

// Multiplier for "auto" mode threshold (reduces false positives)
constexpr float DEFAULT_ADAPTIVE_FACTOR = 1.3f;

/**
 * Threshold mode enumeration
 */
enum class ThresholdMode {
  AUTO,    // max x 1.3 (default)
  MIN,     // max x 1.0 (maximum sensitivity)
  MANUAL   // User-specified fixed value (no adaptive calculation)
};


/**
 * Calculate the maximum value from a vector.
 * 
 * @param values Vector of numeric values
 * @return Maximum value (1.0f if vector is empty)
 */
inline float calculate_max_value(const std::vector<float>& values) {
  if (values.empty()) {
    return 1.0f;
  }
  return *std::max_element(values.begin(), values.end());
}

/**
 * Get threshold multiplier from mode
 * 
 * @param mode Threshold mode (AUTO or MIN)
 * @return multiplier value (1.3 for AUTO, 1.0 for MIN)
 */
inline float get_threshold_factor(ThresholdMode mode) {
  if (mode == ThresholdMode::AUTO) {
    return DEFAULT_ADAPTIVE_FACTOR;
  }
  return 1.0f;  // MIN: no multiplier
}

/**
 * Calculate adaptive threshold from calibration baseline values
 * 
 * MVS: threshold = max(cal_values) x factor for the current production modes
 * 
 * AUTO mode applies a 1.3x multiplier to reduce false positives.
 * MIN mode uses the raw max moving variance for maximum sensitivity.
 * 
 * @param cal_values Vector of moving variance values from calibration
 * @param mode Threshold mode (AUTO or MIN)
 * @param out_threshold Output: calculated adaptive threshold
 * @param out_factor Output: multiplier used
 */
inline void calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    ThresholdMode mode,
    float& out_threshold,
    float& out_factor) {
  out_factor = get_threshold_factor(mode);
  out_threshold = calculate_max_value(cal_values) * out_factor;
}

/**
 * Calculate adaptive threshold with an explicit factor.
 * 
 * @param cal_values Vector of moving variance values from baseline
 * @param factor Multiplier to apply to the max moving variance
 * @return Calculated adaptive threshold
 */
inline float calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    float factor) {
  return calculate_max_value(cal_values) * factor;
}

}  // namespace espectre
}  // namespace esphome
