/*
 * ESPectre - ML Feature Extraction
 * 
 * Extracts 12 features from CSI data for ML-based motion detection.
 * Port of micro-espectre/src/features.py to C++.
 * 
 * Features (in order):
 *  0. turb_mean     - Mean of turbulence buffer
 *  1. turb_std      - Standard deviation
 *  2. turb_max      - Maximum value
 *  3. turb_min      - Minimum value
 *  4. turb_range    - Range (max - min)
 *  5. turb_var      - Variance
 *  6. turb_iqr      - IQR approximation (range * 0.5)
 *  7. turb_entropy  - Shannon entropy
 *  8. amp_skewness  - Amplitude skewness
 *  9. amp_kurtosis  - Amplitude kurtosis
 * 10. turb_slope    - Linear regression slope
 * 11. turb_delta    - Last - first value
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cmath>

namespace esphome {
namespace espectre {

// Number of features extracted
constexpr uint8_t ML_NUM_FEATURES = 12;

// Number of entropy bins
constexpr uint8_t ML_ENTROPY_BINS = 10;

/**
 * Calculate Fisher's skewness (third standardized moment).
 * 
 * @param values Array of values
 * @param count Number of values
 * @return Skewness coefficient
 */
inline float calc_skewness(const float* values, uint8_t count) {
    if (count < 3) return 0.0f;
    
    // Calculate mean
    float mean = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        mean += values[i];
    }
    mean /= count;
    
    // Calculate variance
    float variance = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        variance += diff * diff;
    }
    variance /= count;
    
    float std_dev = std::sqrt(variance);
    if (std_dev < 1e-10f) return 0.0f;
    
    // Third central moment
    float m3 = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
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
 * @return Excess kurtosis coefficient
 */
inline float calc_kurtosis(const float* values, uint8_t count) {
    if (count < 4) return 0.0f;
    
    // Calculate mean
    float mean = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        mean += values[i];
    }
    mean /= count;
    
    // Calculate variance
    float variance = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        variance += diff * diff;
    }
    variance /= count;
    
    float std_dev = std::sqrt(variance);
    if (std_dev < 1e-10f) return 0.0f;
    
    // Fourth central moment
    float m4 = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
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
 * Extract all 12 ML features from turbulence buffer and amplitudes.
 * 
 * @param turb_buffer Turbulence buffer (circular, but we use it linearly)
 * @param turb_count Number of valid values in turbulence buffer
 * @param amplitudes Current packet amplitudes (12 subcarriers)
 * @param amp_count Number of amplitude values
 * @param features_out Output array for 12 features (must be pre-allocated)
 */
inline void extract_ml_features(const float* turb_buffer, uint16_t turb_count,
                                const float* amplitudes, uint8_t amp_count,
                                float* features_out) {
    // Initialize to zero
    for (uint8_t i = 0; i < ML_NUM_FEATURES; i++) {
        features_out[i] = 0.0f;
    }
    
    if (turb_count < 2) return;
    
    // Calculate turbulence statistics
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
    float turb_range = turb_max - turb_min;
    
    // Calculate variance
    float var_sum = 0.0f;
    for (uint16_t i = 0; i < turb_count; i++) {
        float diff = turb_buffer[i] - turb_mean;
        var_sum += diff * diff;
    }
    float turb_var = var_sum / turb_count;
    float turb_std = std::sqrt(turb_var);
    
    // IQR approximation
    float turb_iqr = turb_range * 0.5f;
    
    // Entropy
    float turb_entropy = calc_entropy(turb_buffer, turb_count);
    
    // Amplitude features
    float amp_skewness = 0.0f;
    float amp_kurtosis = 0.0f;
    if (amplitudes != nullptr && amp_count >= 3) {
        amp_skewness = calc_skewness(amplitudes, amp_count);
        amp_kurtosis = calc_kurtosis(amplitudes, amp_count);
    }
    
    // Temporal features: slope via linear regression
    // slope = Σ((i - mean_i)(x - mean_x)) / Σ(i - mean_i)²
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
    features_out[0] = turb_mean;
    features_out[1] = turb_std;
    features_out[2] = turb_max;
    features_out[3] = turb_min;
    features_out[4] = turb_range;
    features_out[5] = turb_var;
    features_out[6] = turb_iqr;
    features_out[7] = turb_entropy;
    features_out[8] = amp_skewness;
    features_out[9] = amp_kurtosis;
    features_out[10] = turb_slope;
    features_out[11] = turb_delta;
}

}  // namespace espectre
}  // namespace esphome
