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

// Time domain features
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

// Main feature extraction function
void csi_extract_features(const int8_t *csi_data, 
                          size_t csi_len,
                          csi_features_t *features) {
    if (!csi_data || !features) {
        ESP_LOGE(TAG, "csi_extract_features: NULL pointer");
        return;
    }
    
    // Time domain features
    features->variance = csi_calculate_variance(csi_data, csi_len);
    features->skewness = csi_calculate_skewness(csi_data, csi_len);
    features->kurtosis = csi_calculate_kurtosis(csi_data, csi_len);
    features->entropy = csi_calculate_entropy(csi_data, csi_len);
    features->iqr = csi_calculate_iqr(csi_data, csi_len);
    
    // Spatial features
    features->spatial_variance = csi_calculate_spatial_variance(csi_data, csi_len);
    features->spatial_correlation = csi_calculate_spatial_correlation(csi_data, csi_len);
    features->spatial_gradient = csi_calculate_spatial_gradient(csi_data, csi_len);
}
