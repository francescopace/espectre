/*
 * ESPectre - CSI Processing Module
 * 
 * Extracts 10 mathematical features from Channel State Information (CSI) data:
 * - Statistical (5): variance, skewness (amplitude-based), kurtosis, entropy, IQR
 * - Spatial (3): variance, correlation, gradient (across subcarriers within packet)
 * - Temporal (2): delta_mean, delta_variance (changes between consecutive packets)
 * 
 * NOTE: Skewness is calculated on a moving window of amplitude values (not raw bytes)
 *       for better separation between baseline and movement (82.3% accuracy vs 72.6%)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef CSI_PROCESSOR_H
#define CSI_PROCESSOR_H

#include "sdkconfig.h"

#include <stdint.h>
#include <stddef.h>

// Maximum CSI data length (ESP32-S3: 256 bytes, ESP32-C6: 128 bytes, buffer sized for largest)
#define CSI_MAX_LENGTH 384

// CSI features extracted from raw data
typedef struct {
    // Statistical features (5)
    float variance;
    float skewness;
    float kurtosis;
    float entropy;
    float iqr;  // Interquartile range
    
    // Spatial features (3)
    float spatial_variance;
    float spatial_correlation;
    float spatial_gradient;
    
    // Temporal features (2) - changes between consecutive packets
    float temporal_delta_mean;      // Average absolute difference from previous packet
    float temporal_delta_variance;  // Variance of differences from previous packet
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
 * Extract features from CSI data
 * 
 * @param csi_data Raw CSI data (int8_t array)
 * @param csi_len Length of CSI data
 * @param features Output structure for extracted features
 * @param selected_features Array of feature indices to calculate
 * @param num_features Number of features to calculate
 */
void csi_extract_features(const int8_t *csi_data,
                         size_t csi_len,
                         csi_features_t *features,
                         const uint8_t *selected_features,
                         uint8_t num_features);

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

/**
 * Reset amplitude skewness buffer
 * Call this to clear the amplitude history used for skewness calculation
 */
void csi_reset_amplitude_skewness_buffer(void);

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

/**
 * Calculate spatial turbulence (std of subcarrier amplitudes)
 * Used for Moving Variance Segmentation (MVS)
 * 
 * @param csi_data Raw CSI data (I/Q pairs for subcarriers)
 * @param csi_len Length of CSI data
 * @param selected_subcarriers Array of subcarrier indices to use
 * @param num_subcarriers Number of selected subcarriers
 * @return Spatial turbulence (standard deviation of amplitudes)
 */
float csi_calculate_spatial_turbulence(const int8_t *csi_data, size_t csi_len,
                                       const uint8_t *selected_subcarriers,
                                       uint8_t num_subcarriers);

/**
 * Set the subcarrier selection for feature extraction
 * This updates the internal configuration used by all CSI processing functions
 * 
 * @param selected_subcarriers Array of subcarrier indices (0-63)
 * @param num_subcarriers Number of selected subcarriers (1-64)
 */
void csi_set_subcarrier_selection(const uint8_t *selected_subcarriers,
                                   uint8_t num_subcarriers);

/**
 * Get current subcarrier selection
 * 
 * @param selected_subcarriers Output array for subcarrier indices
 * @param num_subcarriers Output for number of selected subcarriers
 */
void csi_get_subcarrier_selection(uint8_t *selected_subcarriers,
                                   uint8_t *num_subcarriers);

#endif // CSI_PROCESSOR_H
