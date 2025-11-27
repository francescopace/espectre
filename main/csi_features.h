/*
 * ESPectre - CSI Feature Calculation Module
 * 
 * Provides individual feature calculation functions for CSI data analysis.
 * Features are organized in three categories:
 * - Statistical: variance, skewness, kurtosis, entropy, IQR
 * - Spatial: variance, correlation, gradient (across subcarriers)
 * - Temporal: delta mean, delta variance (between consecutive packets)
 * 
 * Note: The csi_features_t struct is defined in csi_processor.h
 * Note: Feature extraction orchestration (csi_extract_features) is in csi_processor.c
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef CSI_FEATURES_H
#define CSI_FEATURES_H

#include <stdint.h>
#include <stddef.h>
#include "csi_processor.h"  // For csi_features_t definition

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

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
float calculate_variance_two_pass(const float *values, size_t n);

// ============================================================================
// STATISTICAL FEATURES
// ============================================================================

/**
 * Calculate variance from int8_t CSI data
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Variance value
 */
float csi_calculate_variance(const int8_t *data, size_t len);

/**
 * Calculate skewness from turbulence buffer
 * Measures asymmetry of the distribution
 * 
 * @param buffer Turbulence buffer (float array)
 * @param count Number of valid values in buffer
 * @return Skewness value, or 0.0 if count < 3
 */
float csi_calculate_skewness(const float *buffer, uint16_t count);

/**
 * Calculate kurtosis from turbulence buffer
 * Measures "tailedness" of the distribution
 * 
 * @param buffer Turbulence buffer (float array)
 * @param count Number of valid values in buffer
 * @return Kurtosis value (excess kurtosis, normal distribution = 0), or 0.0 if count < 4
 */
float csi_calculate_kurtosis(const float *buffer, uint16_t count);

/**
 * Calculate Shannon entropy
 * Measures randomness/information content
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Entropy value in bits
 */
float csi_calculate_entropy(const int8_t *data, size_t len);

/**
 * Calculate Interquartile Range (IQR)
 * Robust measure of statistical dispersion
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return IQR value (Q3 - Q1)
 */
float csi_calculate_iqr(const int8_t *data, size_t len);

// ============================================================================
// SPATIAL FEATURES
// ============================================================================

/**
 * Calculate spatial variance
 * Variance across antenna/subcarrier space
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Spatial variance
 */
float csi_calculate_spatial_variance(const int8_t *data, size_t len);

/**
 * Calculate spatial correlation
 * Correlation between adjacent samples
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Correlation coefficient (-1 to 1)
 */
float csi_calculate_spatial_correlation(const int8_t *data, size_t len);

/**
 * Calculate spatial gradient
 * Average absolute difference between adjacent samples
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Average gradient magnitude
 */
float csi_calculate_spatial_gradient(const int8_t *data, size_t len);

// ============================================================================
// TEMPORAL FEATURES
// ============================================================================

/**
 * Calculate temporal delta mean
 * Average absolute difference between current and previous packet
 * 
 * @param current_data Current CSI packet
 * @param previous_data Previous CSI packet
 * @param len Length of data arrays
 * @return Average absolute difference
 */
float csi_calculate_temporal_delta_mean(const int8_t *current_data,
                                        const int8_t *previous_data,
                                        size_t len);

/**
 * Calculate temporal delta variance
 * Variance of differences between current and previous packet
 * 
 * @param current_data Current CSI packet
 * @param previous_data Previous CSI packet
 * @param len Length of data arrays
 * @return Variance of differences
 */
float csi_calculate_temporal_delta_variance(const int8_t *current_data,
                                            const int8_t *previous_data,
                                            size_t len);

/**
 * Reset temporal feature buffer
 * Call this when you want to clear the history of previous packets
 */
void csi_reset_temporal_buffer(void);

#endif // CSI_FEATURES_H
