/*
 * ESPectre - CSI Processing Module Implementation
 * 
 * Combines CSI feature extraction with Moving Variance Segmentation (MVS).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_processor.h"
#include "csi_features.h"
#include "filters.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include "esp_log.h"
#include "espectre.h"
#include "validation.h"

static const char *TAG = "CSI_Processor";

// ============================================================================
// SUBCARRIER SELECTION - Configurable at runtime
// ============================================================================

// Runtime subcarrier selection (configurable via MQTT/NVS)
static uint8_t g_selected_subcarriers[64];
static uint8_t g_num_selected_subcarriers = 0;

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
// HAMPEL FILTER FOR TURBULENCE
// ============================================================================

// Apply Hampel filter to turbulence value
// Uses the existing hampel_filter() function from filters.c with a circular buffer
static float apply_hampel_to_turbulence(csi_processor_context_t *ctx, float turbulence) {
#if ENABLE_HAMPEL_TURBULENCE_FILTER
    // Add value to Hampel circular buffer
    ctx->hampel_buffer[ctx->hampel_index] = turbulence;
    ctx->hampel_index = (ctx->hampel_index + 1) % HAMPEL_TURBULENCE_WINDOW;
    if (ctx->hampel_count < HAMPEL_TURBULENCE_WINDOW) {
        ctx->hampel_count++;
    }
    
    // Need at least 3 values for meaningful Hampel filtering
    if (ctx->hampel_count < 3) {
        return turbulence;
    }
    
    // Call existing hampel_filter() function from filters.c
    return hampel_filter(ctx->hampel_buffer, ctx->hampel_count, 
                        turbulence, HAMPEL_TURBULENCE_THRESHOLD);
#else
    // Hampel filter disabled - return raw value
    return turbulence;
#endif
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
    // Apply Hampel filter to remove outliers before adding to MVS buffer
    float filtered_turbulence = apply_hampel_to_turbulence(ctx, turbulence);
    
    // Add filtered value to circular buffer
    ctx->turbulence_buffer[ctx->buffer_index] = filtered_turbulence;
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
// FEATURE EXTRACTION ORCHESTRATION
// ============================================================================

// Main feature extraction function (orchestrator)
// This function decides which features to calculate and calls the individual
// feature calculation functions from csi_features.c
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
    
    // Use original data directly
    const int8_t *data_to_use = csi_data;
    size_t len_to_use = csi_len;
    
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
                // Calculate temporal features only once per packet
                if (temporal_calculated) {
                    break;
                }
                
                temporal_calculated = true;
                
                // Temporal features use internal state in csi_features.c
                // Just call the functions - they manage their own previous packet buffer
                features->temporal_delta_mean = csi_calculate_temporal_delta_mean(
                    data_to_use, data_to_use, len_to_use);  // Note: csi_features.c manages prev buffer
                features->temporal_delta_variance = csi_calculate_temporal_delta_variance(
                    data_to_use, data_to_use, len_to_use);
                break;
            default:
                ESP_LOGW(TAG, "Unknown feature index: %d", feat_idx);
                break;
        }
    }
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
