/*
 * ESPectre - CSI Feature Calculation Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_features.h"
#include "espectre.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "esp_log.h"

static const char *TAG = "CSI_Features";

// Numerical stability constant
#define EPSILON_SMALL 1e-6f

// ============================================================================
// STATIC BUFFERS AND HELPERS
// ============================================================================

// Reusable buffer for IQR sorting (avoids malloc in hot path)
static int8_t iqr_sort_buffer[CSI_MAX_LENGTH];

// Temporal features: unified buffer
static int8_t prev_csi_data[CSI_MAX_LENGTH] = {0};
static size_t prev_csi_len = 0;
static bool first_packet = true;

// qsort comparator for int8_t
static int compare_int8(const void *a, const void *b) {
    int8_t ia = *(const int8_t*)a;
    int8_t ib = *(const int8_t*)b;
    return (ia > ib) - (ia < ib);
}

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Two-pass variance calculation (numerically stable)
// variance = sum((x - mean)^2) / n
float calculate_variance_two_pass(const float *values, size_t n) {
    if (n == 0) return 0.0f;
    
    // First pass: calculate mean
    float sum = 0.0f;
    for (size_t i = 0; i < n; i++) {
        sum += values[i];
    }
    float mean = sum / n;
    
    // Second pass: calculate variance
    float variance = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = values[i] - mean;
        variance += diff * diff;
    }
    
    return variance / n;
}

// ============================================================================
// STATISTICAL FEATURES
// ============================================================================

// Calculate variance from int8_t CSI data
float csi_calculate_variance(const int8_t *data, size_t len) {
    if (len == 0) return 0.0f;
    
    // Calculate mean
    float mean = 0.0f;
    for (size_t i = 0; i < len; i++) {
        mean += data[i];
    }
    mean /= len;
    
    // Calculate variance
    float variance = 0.0f;
    for (size_t i = 0; i < len; i++) {
        float diff = data[i] - mean;
        variance += diff * diff;
    }
    return variance / len;
}

// Calculate skewness from turbulence buffer
// Skewness = E[(X - μ)³] / σ³
float csi_calculate_skewness(const float *buffer, uint16_t count) {
    if (!buffer || count < 3) {
        return 0.0f;
    }
    
    // Calculate mean
    float mean = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        mean += buffer[i];
    }
    mean /= count;
    
    // Calculate second and third moments
    float m2 = 0.0f;
    float m3 = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float diff = buffer[i] - mean;
        float diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }
    
    m2 /= count;
    m3 /= count;
    
    // Calculate skewness
    float stddev = sqrtf(m2);
    if (stddev < EPSILON_SMALL) {
        return 0.0f;
    }
    
    return m3 / (stddev * stddev * stddev);
}

// Calculate kurtosis from turbulence buffer
// Excess Kurtosis = E[(X - μ)⁴] / σ⁴ - 3
float csi_calculate_kurtosis(const float *buffer, uint16_t count) {
    if (!buffer || count < 4) {
        return 0.0f;
    }
    
    // Calculate mean
    float mean = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        mean += buffer[i];
    }
    mean /= count;
    
    // Calculate second and fourth moments
    float m2 = 0.0f;
    float m4 = 0.0f;
    for (uint16_t i = 0; i < count; i++) {
        float diff = buffer[i] - mean;
        float diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }
    
    m2 /= count;
    m4 /= count;
    
    if (m2 < EPSILON_SMALL) {
        return 0.0f;
    }
    
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

// ============================================================================
// SPATIAL FEATURES
// ============================================================================

float csi_calculate_spatial_variance(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    // Calculate variance of spatial differences (between adjacent subcarriers)
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

// ============================================================================
// TEMPORAL FEATURES
// ============================================================================

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

void csi_reset_temporal_buffer(void) {
    memset(prev_csi_data, 0, sizeof(prev_csi_data));
    prev_csi_len = 0;
    first_packet = true;
}
