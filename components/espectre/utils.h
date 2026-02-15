/*
 * ESPectre - Utility Functions
 * 
 * Shared utility functions used across multiple modules.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

// =============================================================================
// Basic Statistical Functions
// =============================================================================

/**
 * Calculate mean of an array
 * 
 * @param values Array of float values
 * @param n Number of values
 * @return Mean (0.0 if n == 0)
 */
inline float calculate_mean(const float* values, size_t n) {
    if (n == 0 || !values) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += values[i];
    }
    return sum / n;
}

/**
 * Calculate median of a float array (sorts array in-place)
 * 
 * @param arr Array of float values (will be sorted)
 * @param size Number of values
 * @return Median value (0.0 if size == 0)
 */
inline float calculate_median_float(float* arr, size_t size) {
    if (size == 0 || !arr) return 0.0f;
    std::sort(arr, arr + size);
    if (size % 2 == 0) {
        return (arr[size / 2 - 1] + arr[size / 2]) / 2.0f;
    }
    return arr[size / 2];
}

/**
 * Calculate median of a uint8 array (sorts array in-place)
 * 
 * @param arr Array of uint8 values (will be sorted)
 * @param size Number of values
 * @return Median value (0 if size == 0)
 */
inline uint8_t calculate_median_u8(uint8_t* arr, size_t size) {
    if (size == 0 || !arr) return 0;
    std::sort(arr, arr + size);
    if (size % 2 == 0) {
        return (arr[size / 2 - 1] + arr[size / 2]) / 2;
    }
    return arr[size / 2];
}

/**
 * Calculate median of an int8 array (sorts array in-place)
 * 
 * @param arr Array of int8 values (will be sorted)
 * @param size Number of values
 * @return Median value (0 if size == 0)
 */
inline int8_t calculate_median_i8(int8_t* arr, size_t size) {
    if (size == 0 || !arr) return 0;
    std::sort(arr, arr + size);
    if (size % 2 == 0) {
        return (arr[size / 2 - 1] + arr[size / 2]) / 2;
    }
    return arr[size / 2];
}

/**
 * Apply CV normalization to standard deviation
 * 
 * CV (Coefficient of Variation) = std / mean
 * Makes turbulence gain-invariant when AGC is not locked.
 * 
 * @param std_dev Standard deviation
 * @param mean Mean value
 * @param use_cv true = return std/mean, false = return raw std
 * @return CV-normalized or raw std
 */
inline float apply_cv_normalization(float std_dev, float mean, bool use_cv) {
    if (!use_cv) return std_dev;
    return (mean > 0.0f) ? std_dev / mean : 0.0f;
}

/**
 * Calculate turbulence from variance with optional CV normalization
 * 
 * Combines variance → std → optional CV normalization in one call.
 * 
 * @param variance Pre-calculated variance
 * @param values Array used for mean calculation (if CV enabled)
 * @param count Number of values
 * @param use_cv true = CV normalization, false = raw std
 * @return Turbulence value
 */
inline float calculate_turbulence_from_variance(float variance, 
                                                 const float* values, 
                                                 size_t count,
                                                 bool use_cv) {
    float std_dev = std::sqrt(variance);
    if (!use_cv) return std_dev;
    float mean = calculate_mean(values, count);
    return apply_cv_normalization(std_dev, mean, true);
}

// =============================================================================
// HT20 Constants (64 subcarriers - do not change)
// =============================================================================
constexpr uint16_t HT20_NUM_SUBCARRIERS = 64;      // HT20: 64 subcarriers
constexpr uint16_t HT20_CSI_LEN = 128;             // 64 SC × 2 bytes (I/Q pairs)
constexpr uint8_t HT20_GUARD_BAND_LOW = 11;        // First valid subcarrier
constexpr uint8_t HT20_GUARD_BAND_HIGH = 52;       // Last valid subcarrier
constexpr uint8_t HT20_DC_SUBCARRIER = 32;         // DC null subcarrier
constexpr uint8_t HT20_SELECTED_BAND_SIZE = 12;    // Selected subcarriers for motion detection

// =============================================================================
// Segmentation Constants
// =============================================================================

// Note: Window size constants are defined in base_detector.h:

constexpr float SEGMENTATION_DEFAULT_THRESHOLD = 1.0f;
// Min threshold lowered to support CV normalization (std/mean produces smaller values)
constexpr float SEGMENTATION_MIN_THRESHOLD = 1e-9f;
constexpr float SEGMENTATION_MAX_THRESHOLD = 10.0f;

/**
 * Calculate variance using two-pass algorithm (numerically stable)
 * 
 * Two-pass algorithm: variance = sum((x - mean)^2) / n
 * More stable than single-pass E[X²] - E[X]² for float32 arithmetic.
 * 
 * @param values Array of float values
 * @param n Number of values
 * @return Variance (0.0 if n == 0)
 */
inline float calculate_variance_two_pass(const float *values, size_t n) {
    if (n == 0 || !values) {
        return 0.0f;
    }
    
    // First pass: calculate mean
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) {
        mean += values[i];
    }
    mean /= n;
    
    // Second pass: calculate variance
    float variance = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = values[i] - mean;
        variance += diff * diff;
    }
    variance /= n;
    
    return variance;
}

/**
 * Calculate magnitude (amplitude) from I/Q components
 * 
 * @param i In-phase component
 * @param q Quadrature component
 * @return Magnitude = sqrt(I² + Q²)
 */
inline float calculate_magnitude(int8_t i, int8_t q) {
    float fi = static_cast<float>(i);
    float fq = static_cast<float>(q);
    return std::sqrt(fi * fi + fq * fq);
}

/**
 * Calculate spatial turbulence from pre-calculated magnitudes
 * 
 * Spatial turbulence is the standard deviation of magnitudes across
 * selected subcarriers. It measures the spatial variability of the
 * Wi-Fi channel - higher values indicate motion/disturbance.
 * 
 * Two modes:
 *   CV normalization (std/mean): gain-invariant, used when gain is NOT locked
 *   Raw std: better sensitivity for contiguous bands, used when gain IS locked
 * 
 * @param magnitudes Array of magnitude values (one per subcarrier)
 * @param subcarriers Array of selected subcarrier indices
 * @param num_subcarriers Number of selected subcarriers (max 12)
 * @param max_subcarrier Maximum valid subcarrier index (default: 64 for HT20)
 * @param use_cv_normalization true = std/mean, false = raw std (default: true)
 * @return Turbulence value
 */
inline float calculate_spatial_turbulence(const float* magnitudes,
                                          const uint8_t* subcarriers,
                                          uint8_t num_subcarriers,
                                          uint16_t max_subcarrier = 64,
                                          bool use_cv_normalization = true) {
    if (num_subcarriers == 0 || !magnitudes || !subcarriers) {
        return 0.0f;
    }
    
    // Collect valid magnitudes (max 12 subcarriers for band selection)
    float valid_mags[12];
    uint8_t valid_count = 0;
    
    for (uint8_t i = 0; i < num_subcarriers && valid_count < 12; i++) {
        if (subcarriers[i] < max_subcarrier) {
            valid_mags[valid_count++] = magnitudes[subcarriers[i]];
        }
    }
    
    if (valid_count == 0) {
        return 0.0f;
    }
    
    float variance = calculate_variance_two_pass(valid_mags, valid_count);
    return calculate_turbulence_from_variance(variance, valid_mags, valid_count, use_cv_normalization);
}

/**
 * Calculate spatial turbulence directly from raw CSI data (I/Q pairs)
 * 
 * This is a convenience wrapper that calculates magnitudes internally
 * before computing spatial turbulence.
 * 
 * HT20 only: 64 subcarriers, 128 bytes CSI data.
 * 
 * @param csi_data Raw CSI data (interleaved I/Q pairs)
 * @param csi_len Length of CSI data in bytes (expected: 128 for HT20)
 * @param subcarriers Array of selected subcarrier indices
 * @param num_subcarriers Number of selected subcarriers (max 12)
 * @param use_cv_normalization true = std/mean, false = raw std (default: true)
 * @return Turbulence value
 */
inline float calculate_spatial_turbulence_from_csi(const int8_t* csi_data,
                                                   size_t csi_len,
                                                   const uint8_t* subcarriers,
                                                   uint8_t num_subcarriers,
                                                   bool use_cv_normalization = true) {
    if (!csi_data || csi_len < 2 || num_subcarriers == 0 || !subcarriers) {
        return 0.0f;
    }
    
    // HT20: 64 subcarriers max
    int total_subcarriers = static_cast<int>(csi_len / 2);
    
    // Calculate magnitudes only for selected subcarriers (max 12)
    float amplitudes[12];
    int valid_count = 0;
    
    for (int i = 0; i < num_subcarriers && i < 12; i++) {
        int sc_idx = subcarriers[i];
        
        if (sc_idx >= total_subcarriers) {
            continue;
        }
        
        // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        float Q = static_cast<float>(csi_data[sc_idx * 2]);      // Imaginary first
        float I = static_cast<float>(csi_data[sc_idx * 2 + 1]);  // Real second
        amplitudes[valid_count++] = std::sqrt(I * I + Q * Q);
    }
    
    if (valid_count == 0) {
        return 0.0f;
    }
    
    float variance = calculate_variance_two_pass(amplitudes, valid_count);
    return calculate_turbulence_from_variance(variance, amplitudes, valid_count, use_cv_normalization);
}

/**
 * Compare two float values for qsort
 * 
 * @param a Pointer to first float
 * @param b Pointer to second float
 * @return -1 if a < b, 0 if a == b, 1 if a > b
 */
inline int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/**
 * Compare two int8_t values for qsort
 * 
 * @param a Pointer to first int8_t
 * @param b Pointer to second int8_t
 * @return Difference between values
 */
inline int compare_int8(const void *a, const void *b) {
    return (*(const int8_t*)a - *(const int8_t*)b);
}

/**
 * Compare absolute values of two floats for qsort
 * 
 * @param a Pointer to first float
 * @param b Pointer to second float
 * @return -1 if |a| < |b|, 0 if |a| == |b|, 1 if |a| > |b|
 */
inline int compare_float_abs(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < 0) fa = -fa;
    if (fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

/**
 * Create and log a progress bar with optional metrics
 * 
 * @param tag Log tag (e.g., TAG)
 * @param progress Progress value (0.0 to 1.0+)
 * @param width Bar width in characters (default: 20)
 * @param threshold_pos Optional threshold marker position (-1 = no threshold marker)
 * @param format Optional format string for additional text after the bar (can be NULL)
 * @param ... Variable arguments for format string
 * 
 * Examples:
 *   log_progress_bar(TAG, 0.8f, 20, 15, "%d%% | mvmt:%.4f thr:%.4f", percent, mv, thr);
 *   log_progress_bar(TAG, progress, 20, -1, "%d%% (%d/%d)", percent, current, total);
 */
inline void log_progress_bar(const char* tag, float progress, int width = 20, 
                             int threshold_pos = -1, const char* format = nullptr, ...) {
  // Create progress bar
  int filled = (int)(progress * (threshold_pos > 0 ? threshold_pos : width));
  filled = (filled < 0) ? 0 : (filled > width ? width : filled);
  
  char bar[24];  // '[' + 20 chars + '|' + ']' + '\0'
  int idx = 0;
  bar[idx++] = '[';
  
  for (int i = 0; i < width; i++) {
    if (threshold_pos >= 0 && i == threshold_pos) {
      bar[idx++] = '|';
    } else if (i < filled) {
      bar[idx++] = '#';
    } else {
      bar[idx++] = '-';
    }
  }
  
  bar[idx++] = ']';
  bar[idx] = '\0';
  
  // Log with optional formatted text
  if (format != nullptr) {
    char text[256];
    va_list args;
    va_start(args, format);
    vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    ESP_LOGI(tag, "%s %s", bar, text);
  } else {
    ESP_LOGI(tag, "%s", bar);
  }
}

}  // namespace espectre
}  // namespace esphome
