/*
 * ESPectre - ML Feature Extraction
 * 
 * Extracts 12 non-redundant features from CSI turbulence data for ML-based
 * motion detection. Port of micro-espectre/src/features.py to C++.
 * 
 * All features are computed from the turbulence buffer (50 samples),
 * ensuring stable statistical estimates.
 * 
 * Features (in order):
 *  0. turb_mean      - Mean of turbulence buffer
 *  1. turb_std       - Standard deviation
 *  2. turb_max       - Maximum value
 *  3. turb_min       - Minimum value
 *  4. turb_zcr       - Zero-crossing rate around mean
 *  5. turb_skewness  - Fisher's skewness (3rd moment)
 *  6. turb_kurtosis  - Fisher's excess kurtosis (4th moment)
 *  7. turb_entropy   - Shannon entropy
 *  8. turb_autocorr  - Lag-1 autocorrelation
 *  9. turb_mad       - Median absolute deviation
 * 10. turb_slope     - Linear regression slope
 * 11. turb_delta     - Last - first value
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>
#include "utils.h"

namespace esphome {
namespace espectre {

// Number of features extracted
constexpr uint8_t ML_NUM_FEATURES = 12;

// Number of entropy bins
constexpr uint8_t ML_ENTROPY_BINS = 10;

// Maximum buffer size for sorting (MAD calculation)
constexpr uint16_t ML_MAX_SORT_SIZE = 200;

/**
 * Calculate Fisher's skewness (third standardized moment).
 * 
 * @param values Array of values
 * @param count Number of values
 * @param mean Pre-computed mean (must be valid)
 * @param std_dev Pre-computed standard deviation (must be valid)
 * @return Skewness coefficient
 */
inline float calc_skewness(const float* values, uint16_t count, float mean, float std_dev) {
    if (count < 3 || std_dev < 1e-10f) return 0.0f;
    
    float m3 = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        m3 += diff * diff * diff;
    }
    m3 /= count;
    
    return m3 / (std_dev * std_dev * std_dev);
}

/**
 * Calculate Fisher's excess kurtosis (fourth standardized moment - 3).
 * 
 * @param values Array of values
 * @param count Number of values
 * @param mean Pre-computed mean (must be valid)
 * @param std_dev Pre-computed standard deviation (must be valid)
 * @return Excess kurtosis coefficient
 */
inline float calc_kurtosis(const float* values, uint16_t count, float mean, float std_dev) {
    if (count < 4 || std_dev < 1e-10f) return 0.0f;
    
    float m4 = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        float diff2 = diff * diff;
        m4 += diff2 * diff2;
    }
    m4 /= count;
    
    float std4 = std_dev * std_dev * std_dev * std_dev;
    return (m4 / std4) - 3.0f;  // Excess kurtosis
}

/**
 * Calculate Shannon entropy of values.
 * 
 * @param values Array of values
 * @param count Number of values
 * @return Shannon entropy in bits
 */
inline float calc_entropy(const float* values, uint16_t count) {
    if (count < 2) return 0.0f;
    
    // Find min/max
    float min_val = values[0];
    float max_val = values[0];
    for (uint16_t i = 1; i < count; i++) {
        if (values[i] < min_val) min_val = values[i];
        if (values[i] > max_val) max_val = values[i];
    }
    
    float range = max_val - min_val;
    if (range < 1e-10f) return 0.0f;
    
    // Create histogram
    uint16_t bins[ML_ENTROPY_BINS] = {0};
    float bin_width = range / ML_ENTROPY_BINS;
    
    for (uint16_t i = 0; i < count; i++) {
        int bin_idx = static_cast<int>((values[i] - min_val) / bin_width);
        if (bin_idx >= ML_ENTROPY_BINS) bin_idx = ML_ENTROPY_BINS - 1;
        bins[bin_idx]++;
    }
    
    // Calculate entropy
    float entropy = 0.0f;
    float log2 = std::log(2.0f);
    for (uint8_t i = 0; i < ML_ENTROPY_BINS; i++) {
        if (bins[i] > 0) {
            float p = static_cast<float>(bins[i]) / count;
            entropy -= p * std::log(p) / log2;
        }
    }
    
    return entropy;
}

/**
 * Calculate zero-crossing rate around the mean.
 * 
 * Counts the fraction of consecutive samples where the signal crosses
 * the mean value. High ZCR indicates rapid oscillations (motion).
 * 
 * @param values Array of values
 * @param count Number of values
 * @param mean Pre-computed mean
 * @return Zero-crossing rate (0.0 to 1.0)
 */
inline float calc_zero_crossing_rate(const float* values, uint16_t count, float mean) {
    if (count < 2) return 0.0f;
    
    uint16_t crossings = 0;
    bool prev_above = values[0] >= mean;
    
    for (uint16_t i = 1; i < count; i++) {
        bool curr_above = values[i] >= mean;
        if (curr_above != prev_above) {
            crossings++;
        }
        prev_above = curr_above;
    }
    
    return static_cast<float>(crossings) / (count - 1);
}

/**
 * Calculate lag-1 autocorrelation coefficient.
 * 
 * Measures temporal correlation between consecutive values.
 * High autocorrelation indicates smooth signal (idle).
 * 
 * @param values Array of values
 * @param count Number of values
 * @param mean Pre-computed mean
 * @param variance Pre-computed variance
 * @return Autocorrelation coefficient (-1.0 to 1.0)
 */
inline float calc_autocorrelation(const float* values, uint16_t count, float mean, float variance) {
    if (count < 3 || variance < 1e-10f) return 0.0f;
    
    float autocovariance = 0.0f;
    for (uint16_t i = 0; i < count - 1; i++) {
        autocovariance += (values[i] - mean) * (values[i + 1] - mean);
    }
    autocovariance /= (count - 1);
    
    return autocovariance / variance;
}

/**
 * Calculate Median Absolute Deviation (MAD).
 * 
 * Robust measure of variability, less sensitive to outliers than std.
 * Uses insertion sort (efficient for small n, e.g. 50).
 * 
 * @param values Array of values
 * @param count Number of values
 * @return MAD value
 */
inline float calc_mad(const float* values, uint16_t count) {
    if (count < 2 || count > ML_MAX_SORT_SIZE) return 0.0f;
    
    // Copy for sorting (stack allocation, max 200 floats = 800 bytes)
    float sorted[ML_MAX_SORT_SIZE];
    for (uint16_t i = 0; i < count; i++) {
        sorted[i] = values[i];
    }
    
    // Calculate median using utils.h helper
    float median = calculate_median_float(sorted, count);
    
    // Calculate absolute deviations
    float abs_devs[ML_MAX_SORT_SIZE];
    for (uint16_t i = 0; i < count; i++) {
        abs_devs[i] = std::fabs(values[i] - median);
    }
    
    // Return median of absolute deviations
    return calculate_median_float(abs_devs, count);
}

/**
 * Extract all 12 ML features from turbulence buffer.
 * 
 * All features are computed from the turbulence buffer (typically 50 samples),
 * ensuring stable statistical estimates. No amplitude-only features.
 * 
 * @param turb_buffer Turbulence buffer
 * @param turb_count Number of valid values in turbulence buffer
 * @param amplitudes Ignored (kept for API compatibility, can be nullptr)
 * @param amp_count Ignored
 * @param features_out Output array for 12 features (must be pre-allocated)
 */
inline void extract_ml_features(const float* turb_buffer, uint16_t turb_count,
                                const float* amplitudes, uint8_t amp_count,
                                float* features_out) {
    // Suppress unused parameter warnings
    (void)amplitudes;
    (void)amp_count;
    
    // Initialize to zero
    for (uint8_t i = 0; i < ML_NUM_FEATURES; i++) {
        features_out[i] = 0.0f;
    }
    
    if (turb_count < 2) return;
    
    // Calculate turbulence statistics (single pass for sum, min, max)
    float turb_sum = 0.0f;
    float turb_min = turb_buffer[0];
    float turb_max = turb_buffer[0];
    
    for (uint16_t i = 0; i < turb_count; i++) {
        float val = turb_buffer[i];
        turb_sum += val;
        if (val < turb_min) turb_min = val;
        if (val > turb_max) turb_max = val;
    }
    
    float turb_mean = turb_sum / turb_count;
    
    // Calculate variance (second pass)
    float var_sum = 0.0f;
    for (uint16_t i = 0; i < turb_count; i++) {
        float diff = turb_buffer[i] - turb_mean;
        var_sum += diff * diff;
    }
    float turb_var = var_sum / turb_count;
    float turb_std = std::sqrt(turb_var);
    
    // Zero-crossing rate
    float turb_zcr = calc_zero_crossing_rate(turb_buffer, turb_count, turb_mean);
    
    // Skewness (pre-computed mean/std passed to avoid redundant calculation)
    float turb_skewness = calc_skewness(turb_buffer, turb_count, turb_mean, turb_std);
    
    // Kurtosis (pre-computed mean/std passed to avoid redundant calculation)
    float turb_kurtosis = calc_kurtosis(turb_buffer, turb_count, turb_mean, turb_std);
    
    // Shannon entropy
    float turb_entropy = calc_entropy(turb_buffer, turb_count);
    
    // Lag-1 autocorrelation
    float turb_autocorr = calc_autocorrelation(turb_buffer, turb_count, turb_mean, turb_var);
    
    // Median absolute deviation
    float turb_mad = calc_mad(turb_buffer, turb_count);
    
    // Temporal features: slope via linear regression
    float mean_i = (turb_count - 1) / 2.0f;
    float numerator = 0.0f;
    float denominator = 0.0f;
    
    for (uint16_t i = 0; i < turb_count; i++) {
        float diff_i = i - mean_i;
        float diff_x = turb_buffer[i] - turb_mean;
        numerator += diff_i * diff_x;
        denominator += diff_i * diff_i;
    }
    
    float turb_slope = (denominator > 0.0f) ? (numerator / denominator) : 0.0f;
    float turb_delta = turb_buffer[turb_count - 1] - turb_buffer[0];
    
    // Fill output array in correct order
    features_out[0] = turb_mean;       // 0
    features_out[1] = turb_std;        // 1
    features_out[2] = turb_max;        // 2
    features_out[3] = turb_min;        // 3
    features_out[4] = turb_zcr;        // 4
    features_out[5] = turb_skewness;   // 5
    features_out[6] = turb_kurtosis;   // 6
    features_out[7] = turb_entropy;    // 7
    features_out[8] = turb_autocorr;   // 8
    features_out[9] = turb_mad;        // 9
    features_out[10] = turb_slope;     // 10
    features_out[11] = turb_delta;     // 11
}

}  // namespace espectre
}  // namespace esphome
