/*
 * ESPectre - Adaptive Threshold Calculator
 * 
 * Calculates adaptive threshold from calibration baseline values.
 * Called after calibration to compute the detection threshold.
 * 
 * MVS Formula: threshold = Pxx(cal_values) * factor
 * 
 * Modes:
 * - "auto": P95 * 1.4 (default, low false positives)
 * - "min": P100 * 1.0 (maximum sensitivity, may have FP)
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

// Default parameters
constexpr uint8_t DEFAULT_ADAPTIVE_PERCENTILE = 95;
constexpr float DEFAULT_ADAPTIVE_FACTOR = 1.4f;

/**
 * Threshold mode enumeration
 */
enum class ThresholdMode {
  AUTO,  // P95 * 1.4 (default)
  MIN    // P100 * 1.0 (maximum sensitivity)
};


/**
 * Get threshold parameters from mode
 * 
 * @param mode Threshold mode (AUTO or MIN)
 * @param out_percentile Output: percentile value (0-100)
 * @param out_factor Output: multiplier factor
 */
inline void get_threshold_params(ThresholdMode mode, uint8_t& out_percentile, float& out_factor) {
  if (mode == ThresholdMode::MIN) {
    out_percentile = 100;
    out_factor = 1.0f;
  } else {  // AUTO (default)
    out_percentile = DEFAULT_ADAPTIVE_PERCENTILE;
    out_factor = DEFAULT_ADAPTIVE_FACTOR;
  }
}

/**
 * Calculate percentile value from a vector
 * 
 * Uses linear interpolation between adjacent values.
 * 
 * @param values Vector of numeric values (will be sorted internally)
 * @param percentile Percentile to calculate (0-100)
 * @return Percentile value (1.0f if vector is empty)
 */
inline float calculate_percentile(std::vector<float> values, uint8_t percentile) {
  if (values.empty()) {
    return 1.0f;
  }
  
  std::sort(values.begin(), values.end());
  
  size_t n = values.size();
  float p = percentile / 100.0f;
  float k = (n - 1) * p;
  size_t idx = static_cast<size_t>(k);
  
  if (idx >= n - 1) {
    return values.back();
  }
  
  // Linear interpolation
  float frac = k - idx;
  return values[idx] * (1.0f - frac) + values[idx + 1] * frac;
}

/**
 * Calculate adaptive threshold from calibration baseline values
 * 
 * MVS: threshold = Pxx(cal_values) * factor
 * 
 * @param cal_values Vector of moving variance values from calibration
 * @param mode Threshold mode (AUTO or MIN)
 * @param out_threshold Output: calculated adaptive threshold
 * @param out_percentile Output: percentile used
 * @param out_factor Output: factor used
 * @param out_pxx Output: raw percentile value (before factor)
 */
inline void calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    ThresholdMode mode,
    float& out_threshold,
    uint8_t& out_percentile,
    float& out_factor,
    float& out_pxx) {
  
  // MVS: threshold = Pxx(cal_values) * factor
  get_threshold_params(mode, out_percentile, out_factor);
  out_pxx = calculate_percentile(cal_values, out_percentile);
  out_threshold = out_pxx * out_factor;
}

/**
 * Calculate adaptive threshold with explicit parameters (MVS only)
 * 
 * @param cal_values Vector of moving variance values from baseline
 * @param percentile Percentile to use (0-100)
 * @param factor Multiplier factor
 * @return Calculated adaptive threshold
 */
inline float calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    uint8_t percentile,
    float factor) {
  
  float pxx = calculate_percentile(cal_values, percentile);
  return pxx * factor;
}

}  // namespace espectre
}  // namespace esphome
