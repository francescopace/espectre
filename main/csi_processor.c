/*
 * ESPectre - CSI Processing Module Implementation
 * 
 * Combines CSI feature extraction with Moving Variance Segmentation (MVS).
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
#include "espectre.h"
#include "validation.h"

static const char *TAG = "CSI_Processor";

// Reusable buffer for IQR sorting (avoids malloc in hot path)
static int8_t iqr_sort_buffer[CSI_MAX_LENGTH];

// ============================================================================
// SUBCARRIER SELECTION - Configurable at runtime
// ============================================================================

// Runtime subcarrier selection (configurable via MQTT/NVS)
static uint8_t g_selected_subcarriers[64];
static uint8_t g_num_selected_subcarriers = 0;

// qsort comparator for int8_t
static int compare_int8(const void *a, const void *b) {
    int8_t ia = *(const int8_t*)a;
    int8_t ib = *(const int8_t*)b;
    return (ia > ib) - (ia < ib);
}

// ============================================================================
// VARIANCE CALCULATION
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
// SUBCARRIER FILTERING
// ============================================================================

// Helper: Filter CSI data to selected subcarriers only
// Uses runtime-configurable subcarrier list
// Returns the filtered data length (2 * num_selected_subcarriers)
static size_t csi_filter_subcarriers(const int8_t *input_data, size_t input_len,
                                     int8_t *output_data, size_t max_output_len) {
    if (g_num_selected_subcarriers == 0) {
        ESP_LOGE(TAG, "No subcarriers selected");
        return 0;
    }
    
    int num_subcarriers = input_len / 2;  // Each subcarrier has I and Q
    int useful_count = g_num_selected_subcarriers;
    
    // Check output buffer size
    size_t output_len = useful_count * 2;  // I,Q pairs
    if (output_len > max_output_len) {
        ESP_LOGE(TAG, "Output buffer too small: need %zu, have %zu", output_len, max_output_len);
        return 0;
    }
    
    // Copy selected subcarriers (I,Q pairs)
    for (int i = 0; i < useful_count; i++) {
        int sc_idx = g_selected_subcarriers[i];
        
        // Validate subcarrier index
        if (sc_idx >= num_subcarriers) {
            ESP_LOGE(TAG, "Subcarrier index %d out of range (max %d)", sc_idx, num_subcarriers - 1);
            return 0;
        }
        
        int src_idx = sc_idx * 2;
        int dst_idx = i * 2;
        output_data[dst_idx] = input_data[src_idx];         // I
        output_data[dst_idx + 1] = input_data[src_idx + 1]; // Q
    }
    
    return output_len;
}

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

// Initialize CSI processor context
void csi_processor_init(csi_processor_context_t *ctx) {
    if (!ctx) {
        ESP_LOGE(TAG, "csi_processor_init: NULL context");
        return;
    }
    
    memset(ctx, 0, sizeof(csi_processor_context_t));
    
    // Initialize with platform-specific defaults
    ctx->window_size = SEGMENTATION_DEFAULT_WINDOW_SIZE;
    ctx->threshold = SEGMENTATION_DEFAULT_THRESHOLD;
    ctx->state = CSI_STATE_IDLE;
    
    ESP_LOGI(TAG, "CSI processor initialized (window=%d, threshold=%.2f)",
             ctx->window_size, ctx->threshold);
}

// Reset CSI processor context (state machine only, preserve buffer)
void csi_processor_reset(csi_processor_context_t *ctx) {
    if (!ctx) return;
    
    // Reset state machine ONLY (preserve buffer and parameters)
    ctx->state = CSI_STATE_IDLE;
    ctx->packet_index = 0;
    ctx->total_packets_processed = 0;
    
    // PRESERVE these to avoid "cold start" problem:
    // - ctx->turbulence_buffer (circular buffer with last values)
    // - ctx->buffer_index (current position in circular buffer)
    // - ctx->buffer_count (should stay at window_size after warm-up)
    // - ctx->current_moving_variance (will be recalculated on next packet)
    // - ctx->threshold (configured threshold)
    // - ctx->window_size (configured parameter)
    
    ESP_LOGD(TAG, "CSI processor reset (buffer and parameters preserved)");
}

// ============================================================================
// PARAMETER SETTERS
// ============================================================================

// Set window size
bool csi_processor_set_window_size(csi_processor_context_t *ctx, uint16_t window_size) {
    if (!ctx) {
        ESP_LOGE(TAG, "csi_processor_set_window_size: NULL context");
        return false;
    }
    
    // Validate window size using centralized validation
    if (!validate_segmentation_window_size(window_size)) {
        ESP_LOGE(TAG, "Invalid window size: %d", window_size);
        return false;
    }
    
    // If changing window size, reset buffer to avoid inconsistencies
    if (window_size != ctx->window_size) {
        ESP_LOGI(TAG, "Window size changed from %d to %d - resetting buffer", 
                 ctx->window_size, window_size);
        ctx->buffer_index = 0;
        ctx->buffer_count = 0;
        ctx->current_moving_variance = 0.0f;
    }
    
    ctx->window_size = window_size;
    ESP_LOGI(TAG, "Window size updated: %d", window_size);
    return true;
}

// Set threshold
bool csi_processor_set_threshold(csi_processor_context_t *ctx, float threshold) {
    if (!ctx) {
        ESP_LOGE(TAG, "csi_processor_set_threshold: NULL context");
        return false;
    }
    
    // Validate threshold using centralized validation
    if (!validate_segmentation_threshold(threshold)) {
        ESP_LOGE(TAG, "Invalid threshold: %.2f", threshold);
        return false;
    }
    
    ctx->threshold = threshold;
    ESP_LOGI(TAG, "Threshold updated: %.2f", threshold);
    return true;
}

// ============================================================================
// PARAMETER GETTERS
// ============================================================================

uint16_t csi_processor_get_window_size(const csi_processor_context_t *ctx) {
    return ctx ? ctx->window_size : 0;
}

float csi_processor_get_threshold(const csi_processor_context_t *ctx) {
    return ctx ? ctx->threshold : 0.0f;
}

csi_motion_state_t csi_processor_get_state(const csi_processor_context_t *ctx) {
    return ctx ? ctx->state : CSI_STATE_IDLE;
}

float csi_processor_get_moving_variance(const csi_processor_context_t *ctx) {
    return ctx ? ctx->current_moving_variance : 0.0f;
}

float csi_processor_get_last_turbulence(const csi_processor_context_t *ctx) {
    if (!ctx || ctx->buffer_count == 0) {
        return 0.0f;
    }
    
    // Get the most recently added value (buffer_index - 1)
    int16_t last_idx = (int16_t)ctx->buffer_index - 1;
    if (last_idx < 0) {
        last_idx = ctx->window_size - 1;
    }
    
    return ctx->turbulence_buffer[last_idx];
}

uint32_t csi_processor_get_total_packets(const csi_processor_context_t *ctx) {
    return ctx ? ctx->total_packets_processed : 0;
}

const float* csi_processor_get_turbulence_buffer(const csi_processor_context_t *ctx, 
                                                  uint16_t *count) {
    if (!ctx) {
        if (count) *count = 0;
        return NULL;
    }
    
    if (count) {
        *count = ctx->buffer_count;
    }
    
    return ctx->turbulence_buffer;
}

// ============================================================================
// STATISTICAL FEATURE FUNCTIONS
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
// SPATIAL FEATURE FUNCTIONS
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
// TEMPORAL FEATURE FUNCTIONS
// ============================================================================

// Temporal features: unified buffer
static int8_t prev_csi_data[CSI_MAX_LENGTH] = {0};
static size_t prev_csi_len = 0;
static bool first_packet = true;

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

// ============================================================================
// TURBULENCE CALCULATION (for MVS)
// ============================================================================

// Calculate spatial turbulence (std of subcarrier amplitudes)
// Used for Moving Variance Segmentation (MVS)
static float calculate_spatial_turbulence(const int8_t *csi_data, size_t csi_len,
                                          const uint8_t *selected_subcarriers,
                                          uint8_t num_subcarriers) {
    if (!csi_data || csi_len < 2) {
        return 0.0f;
    }
    
    if (num_subcarriers == 0) {
        ESP_LOGE(TAG, "No subcarriers provided");
        return 0.0f;
    }
    
    int total_subcarriers = csi_len / 2;  // Each subcarrier has I and Q
    
    // Temporary buffer for amplitudes (max 64 subcarriers)
    float amplitudes[64];
    int valid_count = 0;
    
    // Calculate amplitudes for selected subcarriers
    for (int i = 0; i < num_subcarriers && i < 64; i++) {
        int sc_idx = selected_subcarriers[i];
        
        // Validate subcarrier index
        if (sc_idx >= total_subcarriers) {
            ESP_LOGW(TAG, "Subcarrier %d out of range, skipping", sc_idx);
            continue;
        }
        
        float I = (float)csi_data[sc_idx * 2];
        float Q = (float)csi_data[sc_idx * 2 + 1];
        amplitudes[valid_count++] = sqrtf(I * I + Q * Q);
    }
    
    if (valid_count == 0) {
        return 0.0f;
    }
    
    // Use two-pass variance for numerical stability
    float variance = calculate_variance_two_pass(amplitudes, valid_count);
    
    return sqrtf(variance);
}

// ============================================================================
// MOVING VARIANCE CALCULATION
// ============================================================================

// Calculate moving variance from turbulence buffer
static float calculate_moving_variance(const csi_processor_context_t *ctx) {
    // Return 0 if buffer not full yet
    if (ctx->buffer_count < ctx->window_size) {
        return 0.0f;
    }
    
    // Use centralized two-pass variance calculation
    return calculate_variance_two_pass(ctx->turbulence_buffer, ctx->window_size);
}

// Add turbulence value to buffer and update state
static void add_turbulence_and_update_state(csi_processor_context_t *ctx, float turbulence) {
    // Add to circular buffer
    ctx->turbulence_buffer[ctx->buffer_index] = turbulence;
    ctx->buffer_index = (ctx->buffer_index + 1) % ctx->window_size;
    if (ctx->buffer_count < ctx->window_size) {
        ctx->buffer_count++;
    }
    
    // Calculate moving variance
    ctx->current_moving_variance = calculate_moving_variance(ctx);
    
    // State machine for motion detection (simplified)
    if (ctx->state == CSI_STATE_IDLE) {
        // IDLE state: looking for motion start
        if (ctx->current_moving_variance > ctx->threshold) {
            // Motion detected - transition to MOTION state
            ctx->state = CSI_STATE_MOTION;
            ESP_LOGD(TAG, "Motion started at packet %lu", (unsigned long)ctx->packet_index);
        }
    } else {
        // MOTION state: check for motion end
        if (ctx->current_moving_variance < ctx->threshold) {
            // Motion ended - return to IDLE state
            ctx->state = CSI_STATE_IDLE;
            ESP_LOGD(TAG, "Motion ended at packet %lu", (unsigned long)ctx->packet_index);
        }
    }
    
    ctx->packet_index++;
    ctx->total_packets_processed++;
}

// ============================================================================
// MAIN PROCESSING FUNCTION
// ============================================================================

// Process a CSI packet: calculate turbulence, update motion detection, extract features
void csi_process_packet(csi_processor_context_t *ctx,
                        const int8_t *csi_data,
                        size_t csi_len,
                        const uint8_t *selected_subcarriers,
                        uint8_t num_subcarriers,
                        csi_features_t *features,
                        const uint8_t *selected_features,
                        uint8_t num_features) {
    if (!ctx || !csi_data) {
        ESP_LOGE(TAG, "csi_process_packet: NULL pointer");
        return;
    }
    
    // Step 1: Calculate spatial turbulence
    float turbulence = calculate_spatial_turbulence(csi_data, csi_len,
                                                    selected_subcarriers,
                                                    num_subcarriers);
    
    // Step 2: Add turbulence to buffer and update motion detection state
    add_turbulence_and_update_state(ctx, turbulence);
    
    // Step 3: Extract features if requested
    if (features && selected_features && num_features > 0) {
        csi_extract_features(csi_data, csi_len,
                            ctx->turbulence_buffer, ctx->buffer_count,
                            features, selected_features, num_features);
    }
}

// ============================================================================
// FEATURE EXTRACTION
// ============================================================================

// Main feature extraction function
void csi_extract_features(const int8_t *csi_data,
                         size_t csi_len,
                         const float *turbulence_buffer,
                         uint16_t turbulence_count,
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
            case 1: // skewness (from turbulence buffer)
                features->skewness = csi_calculate_skewness(turbulence_buffer, turbulence_count);
                break;
            case 2: // kurtosis (from turbulence buffer)
                features->kurtosis = csi_calculate_kurtosis(turbulence_buffer, turbulence_count);
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

// ============================================================================
// SUBCARRIER SELECTION
// ============================================================================

// Set subcarrier selection for feature extraction
void csi_set_subcarrier_selection(const uint8_t *selected_subcarriers,
                                   uint8_t num_subcarriers) {
    if (!selected_subcarriers || num_subcarriers == 0 || num_subcarriers > 64) {
        ESP_LOGE(TAG, "Invalid subcarrier selection parameters");
        return;
    }
    
    memcpy(g_selected_subcarriers, selected_subcarriers, num_subcarriers * sizeof(uint8_t));
    g_num_selected_subcarriers = num_subcarriers;
    
    ESP_LOGI(TAG, "Subcarrier selection updated: %d subcarriers", num_subcarriers);
}

// Get current subcarrier selection
void csi_get_subcarrier_selection(uint8_t *selected_subcarriers,
                                   uint8_t *num_subcarriers) {
    if (!selected_subcarriers || !num_subcarriers) {
        ESP_LOGE(TAG, "Invalid output parameters");
        return;
    }
    
    memcpy(selected_subcarriers, g_selected_subcarriers, 
           g_num_selected_subcarriers * sizeof(uint8_t));
    *num_subcarriers = g_num_selected_subcarriers;
}
