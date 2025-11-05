/*
 * ESPectre - CSI Processing Module
 * 
 * Extracts 8 mathematical features from Channel State Information (CSI) data:
 * - Time domain: variance, skewness, kurtosis, entropy, IQR
 * - Spatial: variance, correlation, gradient
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef CSI_PROCESSOR_H
#define CSI_PROCESSOR_H

#include <stdint.h>
#include <stddef.h>

// Maximum CSI data length for ESP32-S3
#define CSI_MAX_LENGTH 384

// CSI features extracted from raw data
typedef struct {
    // Time domain features (5)
    float variance;
    float skewness;
    float kurtosis;
    float entropy;
    float iqr;  // Interquartile range
    
    // Spatial features (3)
    float spatial_variance;
    float spatial_correlation;
    float spatial_gradient;
} csi_features_t;

/**
 * Calculate variance from int8_t CSI data
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Variance value
 */
float csi_calculate_variance(const int8_t *data, size_t len);

/**
 * Extract all 8 features from CSI data
 * 
 * @param csi_data Raw CSI data (int8_t array)
 * @param csi_len Length of CSI data
 * @param features Output structure for extracted features
 */
void csi_extract_features(const int8_t *csi_data, 
                          size_t csi_len,
                          csi_features_t *features);

/**
 * Calculate skewness (third standardized moment)
 * Measures asymmetry of the distribution
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Skewness value
 */
float csi_calculate_skewness(const int8_t *data, size_t len);

/**
 * Calculate kurtosis (fourth standardized moment)
 * Measures "tailedness" of the distribution
 * 
 * @param data CSI data array
 * @param len Length of data array
 * @return Kurtosis value (excess kurtosis, normal distribution = 0)
 */
float csi_calculate_kurtosis(const int8_t *data, size_t len);

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

#endif // CSI_PROCESSOR_H
