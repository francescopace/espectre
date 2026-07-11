/*
 * ESPectre - Utility Functions
 *
 * Shared statistical helpers (mean, median, variance, turbulence) used
 * across multiple modules. CSI layout constants and payload helpers live
 * in csi_format.h.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cmath>
#include <algorithm>

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
 * Apply gain-invariant normalization to standard deviation
 * 
 * CV (Coefficient of Variation) = std / mean
 * Makes turbulence gain-invariant when AGC is not locked.
 * 
 * @param std_dev Standard deviation
 * @param mean Mean value
 * @return Normalized turbulence
 */
inline float apply_cv_normalization(float std_dev, float mean) {
    return (mean > 0.0f) ? std_dev / mean : 0.0f;
}

/**
 * Calculate turbulence from variance with gain-invariant normalization
 * 
 * Combines variance → std → normalization in one call.
 * 
 * @param variance Pre-calculated variance
 * @param values Array used for mean calculation
 * @param count Number of values
 * @return Turbulence value
 */
inline float calculate_turbulence_from_variance(float variance, 
                                                 const float* values, 
                                                 size_t count) {
    float std_dev = std::sqrt(variance);
    float mean = calculate_mean(values, count);
    return apply_cv_normalization(std_dev, mean);
}

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
 * Write the mean-normalized amplitude profile into `out`
 *
 * Shared numeric core for the L1-Delta detector; mirrors the Python
 * `features.normalize_amplitude_profile_into` helper.
 *
 * @param amplitudes Input amplitude values
 * @param count Number of input values
 * @param out Output buffer (at least `count` elements)
 * @return Number of values written (0 when the profile is invalid)
 */
inline uint8_t normalize_amplitude_profile(const float* amplitudes,
                                           uint8_t count,
                                           float* out) {
    if (!amplitudes || !out || count < 2) {
        return 0;
    }
    float total = 0.0f;
    for (uint8_t i = 0; i < count; i++) {
        total += amplitudes[i];
    }
    if (total <= 0.0f) {
        return 0;
    }
    float mean = total / count;
    for (uint8_t i = 0; i < count; i++) {
        out[i] = amplitudes[i] / mean;
    }
    return count;
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

}  // namespace espectre
