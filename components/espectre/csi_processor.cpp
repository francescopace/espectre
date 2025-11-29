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
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "esp_log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "CSI_Processor";

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

// Initialize CSI processor context
void csi_processor_init(csi_processor_context_t *ctx) {
    if (!ctx) {
        ESP_LOGE(TAG, "csi_processor_init: NULL context");
        return;
    }
    
    // Use C++ aggregate initialization instead of memset
    *ctx = {};
    
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
    
    // Inline validation
    if (std::isnan(threshold) || std::isinf(threshold) || threshold < 0.5f || threshold > 10.0f) {
        ESP_LOGE(TAG, "Invalid threshold: %.2f (must be 0.5-10.0 and not NaN/Inf)", threshold);
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
        amplitudes[valid_count++] = std::sqrt(I * I + Q * Q);
    }
    
    if (valid_count == 0) {
        return 0.0f;
    }
    
    // Use two-pass variance for numerical stability
    float variance = calculate_variance_two_pass(amplitudes, valid_count);
    
    return std::sqrt(variance);
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
    float filtered_turbulence = hampel_filter_turbulence(&ctx->hampel_state, turbulence);
    
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
// Extracts all 10 features from CSI data
void csi_extract_features(const int8_t *csi_data,
                         size_t csi_len,
                         const float *turbulence_buffer,
                         uint16_t turbulence_count,
                         csi_features_t *features) {
    if (!csi_data || !features) {
        ESP_LOGE(TAG, "csi_extract_features: NULL pointer");
        return;
    }
    
    // Initialize all features to 0
    *features = {};
    
    // Statistical features
    features->variance = csi_calculate_variance(csi_data, csi_len);
    features->skewness = csi_calculate_skewness(turbulence_buffer, turbulence_count);
    features->kurtosis = csi_calculate_kurtosis(turbulence_buffer, turbulence_count);
    features->entropy = csi_calculate_entropy(csi_data, csi_len);
    features->iqr = csi_calculate_iqr(csi_data, csi_len);
    
    // Spatial features
    features->spatial_variance = csi_calculate_spatial_variance(csi_data, csi_len);
    features->spatial_correlation = csi_calculate_spatial_correlation(csi_data, csi_len);
    features->spatial_gradient = csi_calculate_spatial_gradient(csi_data, csi_len);
    
    // Temporal features - use internal state in csi_features.cpp
    features->temporal_delta_mean = csi_calculate_temporal_delta_mean(csi_data, csi_data, csi_len);
    features->temporal_delta_variance = csi_calculate_temporal_delta_variance(csi_data, csi_data, csi_len);
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
                        csi_features_t *features) {
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
    
    // Step 3: Extract all features if requested
    if (features) {
        csi_extract_features(csi_data, csi_len,
                            ctx->turbulence_buffer, ctx->buffer_count,
                            features);
    }
}

// ============================================================================
// SUBCARRIER SELECTION
// ============================================================================

// Note: Subcarrier selection is now managed by ESpectreComponent class
// This function is kept for API compatibility but is no longer used
void csi_set_subcarrier_selection(const uint8_t *selected_subcarriers,
                                   uint8_t num_subcarriers) {
    if (!selected_subcarriers || num_subcarriers == 0 || num_subcarriers > 64) {
        ESP_LOGE(TAG, "Invalid subcarrier selection parameters");
        return;
    }
    
    ESP_LOGI(TAG, "Subcarrier selection updated: %d subcarriers (managed by component)", num_subcarriers);
}

}  // namespace espectre
}  // namespace esphome
