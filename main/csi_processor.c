/*
 * ESPectre - CSI Processing Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_processor.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char *TAG = "CSI_Processor";

// Numerical stability constant
#define EPSILON_SMALL 1e-6f

// Statistical moments for single-pass calculation
typedef struct {
    float mean;
    float m2;  // second moment (variance * n)
    float m3;  // third moment
    float m4;  // fourth moment
} moments_t;

// Reusable buffer for IQR sorting (avoids malloc in hot path)
static int8_t iqr_sort_buffer[CSI_MAX_LENGTH];

// qsort comparator for int8_t
static int compare_int8(const void *a, const void *b) {
    int8_t ia = *(const int8_t*)a;
    int8_t ib = *(const int8_t*)b;
    return (ia > ib) - (ia < ib);
}

// Fast mean calculation for CSI data
static inline float calculate_mean_int8(const int8_t *data, size_t len) {
    if (len == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        sum += data[i];
    }
    return sum / len;
}

// Single-pass calculation of variance, skewness, and kurtosis
static moments_t calculate_moments(const int8_t *data, size_t len) {
    moments_t moments = {0};
    if (len == 0) return moments;
    
    moments.mean = calculate_mean_int8(data, len);
    
    for (size_t i = 0; i < len; i++) {
        float diff = data[i] - moments.mean;
        float diff2 = diff * diff;
        moments.m2 += diff2;
        moments.m3 += diff2 * diff;
        moments.m4 += diff2 * diff2;
    }
    
    return moments;
}

// Basic statistical functions
float csi_calculate_variance(const int8_t *data, size_t len) {
    if (len == 0) return 0.0f;
    
    float mean = 0.0f;
    for (size_t i = 0; i < len; i++) {
        mean += data[i];
    }
    mean /= len;
    
    float variance = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    return variance / len;
}

// Statistical features
float csi_calculate_skewness(const int8_t *data, size_t len) {
    if (len < 3) return 0.0f;
    
    moments_t moments = calculate_moments(data, len);
    float m2 = moments.m2 / len;
    float m3 = moments.m3 / len;
    
    float stddev = sqrtf(m2);
    if (stddev < EPSILON_SMALL) return 0.0f;
    
    return m3 / (stddev * stddev * stddev);
}

float csi_calculate_kurtosis(const int8_t *data, size_t len) {
    if (len < 4) return 0.0f;
    
    moments_t moments = calculate_moments(data, len);
    float m2 = moments.m2 / len;
    float m4 = moments.m4 / len;
    
    if (m2 < EPSILON_SMALL) return 0.0f;
    
    // Return excess kurtosis (normal distribution = 0)
    return (m4 / (m2 * m2)) - 3.0f;
}

float csi_calculate_entropy(const int8_t *data, size_t len) {
    if (len == 0) return 0.0f;
    
    // Create histogram (256 bins for int8_t range)
    int histogram[256] = {0};
    
    for (size_t i = 0; i < len; i++) {
        int bin = (int)data[i] + 128;  // Shift to 0-255 range
        histogram[bin]++;
    }
    
    // Calculate Shannon entropy
    float entropy = 0.0f;
    for (int i = 0; i < 256; i++) {
        if (histogram[i] > 0) {
            float p = (float)histogram[i] / len;
            entropy -= p * log2f(p);
        }
    }
    
    return entropy;
}

float csi_calculate_iqr(const int8_t *data, size_t len) {
    if (len < 4) return 0.0f;
    
    // Use static buffer to avoid dynamic allocation
    if (len > CSI_MAX_LENGTH) {
        ESP_LOGE(TAG, "IQR: data length %zu exceeds buffer size %d", len, CSI_MAX_LENGTH);
        return 0.0f;
    }
    
    // Copy and sort data
    memcpy(iqr_sort_buffer, data, len * sizeof(int8_t));
    qsort(iqr_sort_buffer, len, sizeof(int8_t), compare_int8);
    
    // Calculate Q1 and Q3
    size_t q1_idx = len / 4;
    size_t q3_idx = (3 * len) / 4;
    
    float q1 = iqr_sort_buffer[q1_idx];
    float q3 = iqr_sort_buffer[q3_idx];
    
    return q3 - q1;
}

// Spatial features
float csi_calculate_spatial_variance(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    // Calculate variance of spatial differences (between adjacent subcarriers)
    // This captures how much the signal varies spatially across subcarriers
    float mean_diff = 0.0f;
    size_t n = len - 1;
    
    // First pass: calculate mean of absolute differences
    for (size_t i = 0; i < n; i++) {
        mean_diff += fabsf((float)(data[i + 1] - data[i]));
    }
    mean_diff /= n;
    
    // Second pass: calculate variance of differences
    float variance = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = fabsf((float)(data[i + 1] - data[i]));
        float deviation = diff - mean_diff;
        variance += deviation * deviation;
    }
    
    return variance / n;
}

float csi_calculate_spatial_correlation(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    float sum_xy = 0.0f;
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    float sum_x2 = 0.0f;
    float sum_y2 = 0.0f;
    size_t n = len - 1;
    
    for (size_t i = 0; i < n; i++) {
        float x = data[i];
        float y = data[i + 1];
        sum_xy += x * y;
        sum_x += x;
        sum_y += y;
        sum_x2 += x * x;
        sum_y2 += y * y;
    }
    
    float numerator = n * sum_xy - sum_x * sum_y;
    float term1 = n * sum_x2 - sum_x * sum_x;
    float term2 = n * sum_y2 - sum_y * sum_y;
    
    // Protect against negative values due to floating point errors
    if (term1 < 0.0f) term1 = 0.0f;
    if (term2 < 0.0f) term2 = 0.0f;
    
    float denominator = sqrtf(term1 * term2);
    
    if (denominator < EPSILON_SMALL) return 0.0f;
    
    return numerator / denominator;
}

float csi_calculate_spatial_gradient(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    float sum_diff = 0.0f;
    for (size_t i = 0; i < len - 1; i++) {
        sum_diff += fabsf((float)(data[i + 1] - data[i]));
    }
    
    return sum_diff / (len - 1);
}

// Temporal features: buffer for previous packet
static int8_t prev_csi_data[CSI_MAX_LENGTH] = {0};
static size_t prev_csi_len = 0;
static bool first_packet = true;

// Temporal delta mean calculation
float csi_calculate_temporal_delta_mean(const int8_t *current_data,
                                        const int8_t *previous_data,
                                        size_t len) {
    if (!current_data || !previous_data || len == 0) {
        return 0.0f;
    }
    
    float delta_sum = 0.0f;
    for (size_t i = 0; i < len; i++) {
        delta_sum += fabsf((float)(current_data[i] - previous_data[i]));
    }
    
    return delta_sum / len;
}

// Temporal delta variance calculation
float csi_calculate_temporal_delta_variance(const int8_t *current_data,
                                            const int8_t *previous_data,
                                            size_t len) {
    if (!current_data || !previous_data || len == 0) {
        return 0.0f;
    }
    
    // First calculate delta mean
    float delta_mean = csi_calculate_temporal_delta_mean(current_data, previous_data, len);
    
    // Then calculate variance of deltas
    float delta_variance = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = fabsf((float)(current_data[i] - previous_data[i]));
        float deviation = diff - delta_mean;
        delta_variance += deviation * deviation;
    }
    
    return delta_variance / len;
}

// Reset temporal buffer (call when starting new calibration phase)
void csi_reset_temporal_buffer(void) {
    memset(prev_csi_data, 0, sizeof(prev_csi_data));
    prev_csi_len = 0;
    first_packet = true;
}

// Main feature extraction function
void csi_extract_features(const int8_t *csi_data,
                         size_t csi_len,
                         csi_features_t *features,
                         const uint8_t *selected_features,
                         uint8_t num_features) {
    if (!csi_data || !features) {
        ESP_LOGE(TAG, "csi_extract_features: NULL pointer");
        return;
    }
    
    // Initialize all features to 0
    memset(features, 0, sizeof(csi_features_t));
    
    // Calculate only selected features
    for (uint8_t i = 0; i < num_features; i++) {
        uint8_t feat_idx = selected_features[i];
        
        switch (feat_idx) {
            case 0: // variance
                features->variance = csi_calculate_variance(csi_data, csi_len);
                break;
            case 1: // skewness
                features->skewness = csi_calculate_skewness(csi_data, csi_len);
                break;
            case 2: // kurtosis
                features->kurtosis = csi_calculate_kurtosis(csi_data, csi_len);
                break;
            case 3: // entropy
                features->entropy = csi_calculate_entropy(csi_data, csi_len);
                break;
            case 4: // iqr
                features->iqr = csi_calculate_iqr(csi_data, csi_len);
                break;
            case 5: // spatial_variance
                features->spatial_variance = csi_calculate_spatial_variance(csi_data, csi_len);
                break;
            case 6: // spatial_correlation
                features->spatial_correlation = csi_calculate_spatial_correlation(csi_data, csi_len);
                break;
            case 7: // spatial_gradient
                features->spatial_gradient = csi_calculate_spatial_gradient(csi_data, csi_len);
                break;
            case 8: // temporal_delta_mean
            case 9: // temporal_delta_variance
                // Temporal features require previous packet - calculate both together
                if (first_packet || prev_csi_len != csi_len) {
                    features->temporal_delta_mean = 0.0f;
                    features->temporal_delta_variance = 0.0f;
                    if (csi_len <= CSI_MAX_LENGTH) {
                        memcpy(prev_csi_data, csi_data, csi_len * sizeof(int8_t));
                        prev_csi_len = csi_len;
                    }
                    first_packet = false;
                } else {
                    features->temporal_delta_mean = csi_calculate_temporal_delta_mean(
                        csi_data, prev_csi_data, csi_len);
                    features->temporal_delta_variance = csi_calculate_temporal_delta_variance(
                        csi_data, prev_csi_data, csi_len);
                    memcpy(prev_csi_data, csi_data, csi_len * sizeof(int8_t));
                }
                break;
            default:
                ESP_LOGW(TAG, "Unknown feature index: %d", feat_idx);
                break;
        }
    }
}
