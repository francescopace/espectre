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
#include <stdbool.h>
#include "esp_log.h"

static const char *TAG = "CSI_Processor";

// Numerical stability constant
#define EPSILON_SMALL 1e-6f

// Reusable buffer for IQR sorting (avoids malloc in hot path)
static int8_t iqr_sort_buffer[CSI_MAX_LENGTH];

// Amplitude buffer for skewness/kurtosis calculation (moving window approach)
// Both features share the same buffer for efficiency
#define AMPLITUDE_MOMENTS_WINDOW 20

// Subcarrier selection for amplitude calculation
#define ENABLE_SUBCARRIER_FILTERING 1

#if CONFIG_IDF_TARGET_ESP32C6
    // ESP32-C6: Effective subcarriers are -28~-1, 1~28 (56 total in 64 positions)
    // Subcarrier order: -32~-1, 0~32 (DIFFERENT from ESP32-S3!)
    // Guard band at positions -32~-29 and 29~32 (null values)
    // Temporary range (to be optimized with PCA analysis on ESP32-C6 data)
    // Current range selects central subcarriers for stability
    #define SUBCARRIER_START 10   // Approx. maps to subcarriers -18 to -1
    #define SUBCARRIER_END 38     // Approx. maps to subcarriers 1 to 18
    #define NUM_USEFUL_SUBCARRIERS (SUBCARRIER_END - SUBCARRIER_START)
#else
    // ESP32-S3: 64 subcarriers (order: 0~31, -32~-1)
    // Based on PCA analysis on real ESP32-S3 data: optimal range is [47-58]
    // This captures the most informative subcarriers for motion detection
    #define SUBCARRIER_START 47
    #define SUBCARRIER_END 59     // Inclusive of SC58
    #define NUM_USEFUL_SUBCARRIERS (SUBCARRIER_END - SUBCARRIER_START)
#endif
static float amplitude_moments_buffer[AMPLITUDE_MOMENTS_WINDOW] = {0};
static int amp_moments_index = 0;
static int amp_moments_count = 0;

// Cached moments for kurtosis (calculated during skewness computation)
static float cached_m2 = 0.0f;
static float cached_m4 = 0.0f;
static bool moments_valid = false;

// qsort comparator for int8_t
static int compare_int8(const void *a, const void *b) {
    int8_t ia = *(const int8_t*)a;
    int8_t ib = *(const int8_t*)b;
    return (ia > ib) - (ia < ib);
}

// Helper: Filter CSI data to selected subcarriers only
// This pre-processes the data once, making all feature calculations agnostic
// Returns the filtered data length (2 * num_selected_subcarriers)
static size_t csi_filter_subcarriers(const int8_t *input_data, size_t input_len,
                                     int8_t *output_data, size_t max_output_len) {
    int num_subcarriers = input_len / 2;  // Each subcarrier has I and Q
    
    // Determine range of useful subcarriers
    int start_idx = (num_subcarriers > SUBCARRIER_END) ? SUBCARRIER_START : 0;
    int end_idx = (num_subcarriers > SUBCARRIER_END) ? SUBCARRIER_END : num_subcarriers;
    int useful_count = end_idx - start_idx;
    
    if (useful_count <= 0) {
        // Fallback: use all subcarriers if selection doesn't make sense
        if (input_len <= max_output_len) {
            memcpy(output_data, input_data, input_len);
            return input_len;
        }
        return 0;
    }
    
    // Check output buffer size
    size_t output_len = useful_count * 2;  // I,Q pairs
    if (output_len > max_output_len) {
        ESP_LOGE(TAG, "Output buffer too small: need %zu, have %zu", output_len, max_output_len);
        return 0;
    }
    
    // Copy selected subcarriers (I,Q pairs)
    for (int i = 0; i < useful_count; i++) {
        int src_idx = (start_idx + i) * 2;
        int dst_idx = i * 2;
        output_data[dst_idx] = input_data[src_idx];         // I
        output_data[dst_idx + 1] = input_data[src_idx + 1]; // Q
    }
    
    return output_len;
}

// Basic statistical functions
// Now operates on filtered data (agnostic to subcarrier selection)
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

// Statistical features - AMPLITUDE SKEWNESS (moving window approach)
// Now operates on filtered data (agnostic to subcarrier selection)
float csi_calculate_skewness(const int8_t *data, size_t len) {
    if (len < 2) return 0.0f;
    
    // Step 1: Calculate average amplitude from filtered data
    float avg_amplitude = 0.0f;
    int num_subcarriers = len / 2;  // Each subcarrier has I and Q
    
    for (int i = 0; i < num_subcarriers; i++) {
        float I = (float)data[2 * i];
        float Q = (float)data[2 * i + 1];
        avg_amplitude += sqrtf(I * I + Q * Q);
    }
    avg_amplitude /= num_subcarriers;
    
    // Step 2: Add to circular buffer
    amplitude_moments_buffer[amp_moments_index] = avg_amplitude;
    amp_moments_index = (amp_moments_index + 1) % AMPLITUDE_MOMENTS_WINDOW;
    if (amp_moments_count < AMPLITUDE_MOMENTS_WINDOW) {
        amp_moments_count++;
    }
    
    // Step 3: Calculate skewness and cache moments for kurtosis
    if (amp_moments_count < 3) {
        moments_valid = false;
        return 0.0f;
    }
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < amp_moments_count; i++) {
        mean += amplitude_moments_buffer[i];
    }
    mean /= amp_moments_count;
    
    // Calculate second, third, and fourth moments (for both skewness and kurtosis)
    float m2 = 0.0f;
    float m3 = 0.0f;
    float m4 = 0.0f;
    for (int i = 0; i < amp_moments_count; i++) {
        float diff = amplitude_moments_buffer[i] - mean;
        float diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
        m4 += diff2 * diff2;  // Fourth moment for kurtosis
    }
    
    m2 /= amp_moments_count;
    m3 /= amp_moments_count;
    m4 /= amp_moments_count;
    
    // Cache moments for kurtosis
    cached_m2 = m2;
    cached_m4 = m4;
    moments_valid = true;
    
    // Calculate skewness
    float stddev = sqrtf(m2);
    if (stddev < EPSILON_SMALL) return 0.0f;
    
    return m3 / (stddev * stddev * stddev);
}

// AMPLITUDE KURTOSIS (moving window approach)
// Now operates on filtered data (agnostic to subcarrier selection)
float csi_calculate_kurtosis(const int8_t *data, size_t len) {
    // If moments are valid (skewness was called first in this cycle), use cached values
    // This optimization avoids recalculating the same moments twice
    if (moments_valid && cached_m2 > EPSILON_SMALL) {
        // Return excess kurtosis using cached moments
        return (cached_m4 / (cached_m2 * cached_m2)) - 3.0f;
    }
    
    // If skewness wasn't called first, calculate amplitude kurtosis independently
    if (len < 2) return 0.0f;
    
    // Step 1: Calculate average amplitude from filtered data
    float avg_amplitude = 0.0f;
    int num_subcarriers = len / 2;  // Each subcarrier has I and Q
    
    for (int i = 0; i < num_subcarriers; i++) {
        float I = (float)data[2 * i];
        float Q = (float)data[2 * i + 1];
        avg_amplitude += sqrtf(I * I + Q * Q);
    }
    avg_amplitude /= num_subcarriers;
    
    // Step 2: Add to circular buffer (shared with skewness)
    amplitude_moments_buffer[amp_moments_index] = avg_amplitude;
    amp_moments_index = (amp_moments_index + 1) % AMPLITUDE_MOMENTS_WINDOW;
    if (amp_moments_count < AMPLITUDE_MOMENTS_WINDOW) {
        amp_moments_count++;
    }
    
    // Step 3: Calculate kurtosis from amplitude buffer
    if (amp_moments_count < 4) {
        return 0.0f;
    }
    
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < amp_moments_count; i++) {
        mean += amplitude_moments_buffer[i];
    }
    mean /= amp_moments_count;
    
    // Calculate second and fourth moments
    float m2 = 0.0f;
    float m4 = 0.0f;
    for (int i = 0; i < amp_moments_count; i++) {
        float diff = amplitude_moments_buffer[i] - mean;
        float diff2 = diff * diff;
        m2 += diff2;
        m4 += diff2 * diff2;
    }
    
    m2 /= amp_moments_count;
    m4 /= amp_moments_count;
    
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

// Temporal features: unified buffer (simplified - no raw/filtered separation)
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

// Reset amplitude moments buffer (call when starting new calibration phase)
// Resets buffer used by both skewness and kurtosis
void csi_reset_amplitude_skewness_buffer(void) {
    memset(amplitude_moments_buffer, 0, sizeof(amplitude_moments_buffer));
    amp_moments_index = 0;
    amp_moments_count = 0;
    moments_valid = false;
}

// Calculate spatial turbulence (std of subcarrier amplitudes)
// Used for Moving Variance Segmentation (MVS)
// Replicates: calculate_spatial_turbulence() from Python code
// 
// NOTE: Uses SELECTIVE subcarrier filtering (47-58) to match Python implementation
//       which uses SELECTED_SUBCARRIERS = list(range(47, 59))
float csi_calculate_spatial_turbulence(const int8_t *csi_data, size_t csi_len) {
    if (!csi_data || csi_len < 2) {
        return 0.0f;
    }
    
    // Use SELECTIVE subcarrier filtering (47-58) to match Python
    // Python: SELECTED_SUBCARRIERS = list(range(47, 59))  # 12 subcarriers
    int num_subcarriers = csi_len / 2;  // Each subcarrier has I and Q
    
    // Determine range of useful subcarriers (same as Python SELECTIVE mode)
    int start_idx = (num_subcarriers > SUBCARRIER_END) ? SUBCARRIER_START : 0;
    int end_idx = (num_subcarriers > SUBCARRIER_END) ? SUBCARRIER_END : num_subcarriers;
    int useful_count = end_idx - start_idx;
    
    if (useful_count <= 0) {
        // Fallback: use all subcarriers
        start_idx = 0;
        end_idx = num_subcarriers;
        useful_count = num_subcarriers;
    }
    
    // Calculate amplitudes for selected subcarriers
    float sum = 0.0f;
    float sum_sq = 0.0f;
    
    for (int i = start_idx; i < end_idx; i++) {
        float I = (float)csi_data[i * 2];
        float Q = (float)csi_data[i * 2 + 1];
        float amplitude = sqrtf(I * I + Q * Q);
        
        sum += amplitude;
        sum_sq += amplitude * amplitude;
    }
    
    // Calculate standard deviation
    float mean = sum / useful_count;
    float variance = (sum_sq / useful_count) - (mean * mean);
    
    // Protect against negative variance due to floating point errors
    if (variance < 0.0f) variance = 0.0f;
    
    return sqrtf(variance);
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
    
    // Flag to track if temporal features have been calculated in this call
    bool temporal_calculated = false;
    
#if ENABLE_SUBCARRIER_FILTERING
    // PRE-FILTER: Apply subcarrier selection ONCE for all features
    static int8_t filtered_data[CSI_MAX_LENGTH];
    size_t filtered_len = csi_filter_subcarriers(csi_data, csi_len, filtered_data, CSI_MAX_LENGTH);
    
    if (filtered_len == 0) {
        ESP_LOGE(TAG, "Failed to filter subcarriers");
        return;
    }
    
    // Use filtered data
    const int8_t *data_to_use = filtered_data;
    size_t len_to_use = filtered_len;
#else
    // No filtering: use original data directly
    const int8_t *data_to_use = csi_data;
    size_t len_to_use = csi_len;
#endif
    
    // Calculate only selected features
    for (uint8_t i = 0; i < num_features; i++) {
        uint8_t feat_idx = selected_features[i];
        
        switch (feat_idx) {
            case 0: // variance
                features->variance = csi_calculate_variance(data_to_use, len_to_use);
                break;
            case 1: // skewness
                features->skewness = csi_calculate_skewness(data_to_use, len_to_use);
                break;
            case 2: // kurtosis
                features->kurtosis = csi_calculate_kurtosis(data_to_use, len_to_use);
                break;
            case 3: // entropy
                features->entropy = csi_calculate_entropy(data_to_use, len_to_use);
                break;
            case 4: // iqr
                features->iqr = csi_calculate_iqr(data_to_use, len_to_use);
                break;
            case 5: // spatial_variance
                features->spatial_variance = csi_calculate_spatial_variance(data_to_use, len_to_use);
                break;
            case 6: // spatial_correlation
                features->spatial_correlation = csi_calculate_spatial_correlation(data_to_use, len_to_use);
                break;
            case 7: // spatial_gradient
                features->spatial_gradient = csi_calculate_spatial_gradient(data_to_use, len_to_use);
                break;
            case 8: // temporal_delta_mean
            case 9: // temporal_delta_variance
                // Calculate temporal features only once per packet (skip if already calculated)
                if (temporal_calculated) {
                    break;  // Already calculated when we encountered the other temporal feature index
                }
                
                // Mark as calculated to prevent double calculation
                temporal_calculated = true;
                
                // Temporal features require previous packet - calculate both together
                // Handle first packet: initialize buffer, skip temporal calculation
                if (first_packet) {
                    if (len_to_use <= CSI_MAX_LENGTH) {
                        memcpy(prev_csi_data, data_to_use, len_to_use * sizeof(int8_t));
                        prev_csi_len = len_to_use;
                    }
                    first_packet = false;
                    // Leave temporal features at 0.0 for first packet (already set by memset)
                } else if (prev_csi_len == len_to_use) {
                    // Calculate temporal features from second packet onwards
                    features->temporal_delta_mean = csi_calculate_temporal_delta_mean(
                        data_to_use, prev_csi_data, len_to_use);
                    features->temporal_delta_variance = csi_calculate_temporal_delta_variance(
                        data_to_use, prev_csi_data, len_to_use);
                    // Update buffer for next packet
                    memcpy(prev_csi_data, data_to_use, len_to_use * sizeof(int8_t));
                }
                break;
            default:
                ESP_LOGW(TAG, "Unknown feature index: %d", feat_idx);
                break;
        }
    }
}
