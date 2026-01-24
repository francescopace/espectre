/*
 * ESPectre - Adaptive Threshold Calculator
 * 
 * Calculates adaptive threshold from calibration baseline values.
 * Called after calibration to compute the detection threshold.
 * 
 * MVS Formula: threshold = Pxx(cal_values) * factor
 * PCA Formula: threshold = 1 - min(cal_values) (Espressif approach)
 * 
 * Modes (for MVS):
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
#include "pca_detector.h"  // For PCA_SCALE

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
 * For MVS: threshold = Pxx(cal_values) * factor
 * For PCA: threshold = 1 - min(cal_values) (Espressif correlation-based)
 * 
 * @param cal_values Vector of calibration values (MV for MVS, correlation for PCA)
 * @param mode Threshold mode (AUTO or MIN) - only used for MVS
 * @param is_pca True for PCA algorithm, false for MVS
 * @param out_threshold Output: calculated adaptive threshold
 * @param out_percentile Output: percentile used (0 for PCA)
 * @param out_factor Output: factor used (1.0 for PCA)
 * @param out_pxx Output: raw percentile/min value (before factor)
 */
inline void calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    ThresholdMode mode,
    bool is_pca,
    float& out_threshold,
    uint8_t& out_percentile,
    float& out_factor,
    float& out_pxx) {
  
  if (is_pca) {
    // PCA: threshold = (1 - min(correlation)) * PCA_SCALE
    // cal_values contains correlation values from baseline
    // Scaled by PCA_SCALE (1000) to match MVS threshold range (0.1-10.0)
    if (cal_values.empty()) {
      out_threshold = PCA_DEFAULT_THRESHOLD;  // 10.0 (scaled)
      out_percentile = 0;
      out_factor = 1.0f;
      out_pxx = 0.99f;
      return;
    }
    float min_corr = *std::min_element(cal_values.begin(), cal_values.end());
    out_threshold = (1.0f - min_corr) * PCA_SCALE;
    out_percentile = 0;     // N/A for PCA
    out_factor = 1.0f;      // N/A for PCA
    out_pxx = min_corr;     // Store min_corr for diagnostics
  } else {
    // MVS: threshold = Pxx(cal_values) * factor
    get_threshold_params(mode, out_percentile, out_factor);
    out_pxx = calculate_percentile(cal_values, out_percentile);
    out_threshold = out_pxx * out_factor;
  }
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
