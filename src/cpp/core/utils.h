/*
 * ESPectre - Utility Functions
 *
 * Shared statistical helpers (mean, median, variance, turbulence) used
 * across multiple modules. CSI layout constants and payload helpers live
 * in csi_format.h.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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

struct MeanVariance {
    float mean{0.0f};
    float variance{0.0f};
};

/**
 * Calculate mean and variance in one two-pass sweep (numerically stable)
 *
 * Two-pass algorithm: variance = sum((x - mean)^2) / n
 * More stable than single-pass E[X²] - E[X]² for float32 arithmetic.
 *
 * The single definition matters beyond DRY. Both detectors feed these values
 * into a threshold comparison, and the C++/Python parity gate requires the two
 * runtimes to decide identically, so the accumulation order has to be fixed in
 * exactly one place. Three hand-rolled copies of these loops used to exist.
 *
 * @param values Array of float values
 * @param n Number of values
 * @return Mean and variance (both 0.0 if n == 0)
 */
inline MeanVariance calculate_mean_variance_two_pass(const float *values, size_t n) {
    MeanVariance result;
    if (n == 0 || !values) {
        return result;
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

    result.mean = mean;
    result.variance = variance;
    return result;
}

/**
 * Calculate variance using the two-pass algorithm
 *
 * @param values Array of float values
 * @param n Number of values
 * @return Variance (0.0 if n == 0)
 */
inline float calculate_variance_two_pass(const float *values, size_t n) {
    return calculate_mean_variance_two_pass(values, n).variance;
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
 * Shared numeric core for the C++ L1-delta tracker. The MicroPython tracker
 * performs the same normalization directly in its packet loop.
 *
 * @param amplitudes Input amplitude values
 * @param count Number of input values
 * @param mean Precomputed arithmetic mean of the input values
 * @param out Output buffer (at least `count` elements)
 * @return Number of values written (0 when the profile is invalid)
 */
inline uint8_t normalize_amplitude_profile(const float* amplitudes,
                                           uint8_t count,
                                           float mean,
                                           float* out) {
    if (!amplitudes || !out || count < 2) {
        return 0;
    }
    if (mean <= 0.0f) {
        return 0;
    }
    for (uint8_t i = 0; i < count; i++) {
        out[i] = amplitudes[i] / mean;
    }
    return count;
}

inline uint8_t normalize_amplitude_profile(const float* amplitudes,
                                           uint8_t count,
                                           float* out) {
    if (!amplitudes || !out || count < 2) {
        return 0;
    }
    return normalize_amplitude_profile(
        amplitudes, count, calculate_mean(amplitudes, count), out);
}

}  // namespace espectre
