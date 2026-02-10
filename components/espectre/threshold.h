/*
 * ESPectre - Adaptive Threshold Calculator
 * 
 * Calculates adaptive threshold from calibration baseline values.
 * Called after calibration to compute the detection threshold.
 * 
 * MVS Formula: threshold = percentile(cal_values)
 * 
 * Modes:
 * - "auto": P95 (default, balanced sensitivity/false positives)
 * - "min": P100 (maximum sensitivity, may have FP)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <vector>
#include <algorithm>

namespace esphome {
namespace espectre {

// Default percentile for "auto" mode
constexpr uint8_t DEFAULT_ADAPTIVE_PERCENTILE = 95;

/**
 * Threshold mode enumeration
 */
enum class ThresholdMode {
  AUTO,    // P95 (default)
  MIN,     // P100 (maximum sensitivity)
  MANUAL   // User-specified fixed value (no adaptive calculation)
};


/**
 * Get threshold percentile from mode
 * 
 * @param mode Threshold mode (AUTO or MIN). MANUAL mode bypasses adaptive calculation.
 * @return percentile value (0-100)
 */
inline uint8_t get_threshold_percentile(ThresholdMode mode) {
  if (mode == ThresholdMode::MIN) {
    return 100;
  }
  return DEFAULT_ADAPTIVE_PERCENTILE;  // AUTO
}

/**
 * Calculate percentile value from a vector
 * 
 * Uses linear interpolation between adjacent values.
 * 
 * @param values Vector of numeric values (const ref, copied internally for sorting)
 * @param percentile Percentile to calculate (0-100)
 * @return Percentile value (1.0f if vector is empty)
 */
inline float calculate_percentile(const std::vector<float>& values, uint8_t percentile) {
  if (values.empty()) {
    return 1.0f;
  }
  
  // Copy for sorting (const input)
  std::vector<float> sorted_values(values);
  std::sort(sorted_values.begin(), sorted_values.end());
  
  size_t n = sorted_values.size();
  float p = percentile / 100.0f;
  float k = (n - 1) * p;
  size_t idx = static_cast<size_t>(k);
  
  if (idx >= n - 1) {
    return sorted_values.back();
  }
  
  // Linear interpolation
  float frac = k - idx;
  return sorted_values[idx] * (1.0f - frac) + sorted_values[idx + 1] * frac;
}

/**
 * Calculate adaptive threshold from calibration baseline values
 * 
 * MVS: threshold = percentile(cal_values)
 * 
 * @param cal_values Vector of moving variance values from calibration
 * @param mode Threshold mode (AUTO or MIN)
 * @param out_threshold Output: calculated adaptive threshold
 * @param out_percentile Output: percentile used
 */
inline void calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    ThresholdMode mode,
    float& out_threshold,
    uint8_t& out_percentile) {
  
  out_percentile = get_threshold_percentile(mode);
  out_threshold = calculate_percentile(cal_values, out_percentile);
}

/**
 * Calculate adaptive threshold with explicit percentile
 * 
 * @param cal_values Vector of moving variance values from baseline
 * @param percentile Percentile to use (0-100)
 * @return Calculated adaptive threshold
 */
inline float calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    uint8_t percentile) {
  
  return calculate_percentile(cal_values, percentile);
}

}  // namespace espectre
}  // namespace esphome
