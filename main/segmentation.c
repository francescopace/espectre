/*
 * ESPectre - Moving Variance Segmentation (MVS) Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "segmentation.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char *TAG = "Segmentation";

// Numerical stability constant
#define EPSILON_SMALL 1e-6f

// Initialize segmentation context
void segmentation_init(segmentation_context_t *ctx) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_init: NULL context");
        return;
    }
    
    memset(ctx, 0, sizeof(segmentation_context_t));
    
    // Initialize with platform-specific defaults
    ctx->k_factor = SEGMENTATION_DEFAULT_K_FACTOR;
    ctx->window_size = SEGMENTATION_DEFAULT_WINDOW_SIZE;
    ctx->min_length = SEGMENTATION_DEFAULT_MIN_LENGTH;
    ctx->max_length = SEGMENTATION_DEFAULT_MAX_LENGTH;
    ctx->adaptive_threshold = SEGMENTATION_DEFAULT_THRESHOLD;
    ctx->state = SEG_STATE_IDLE;
    
    ESP_LOGI(TAG, "Segmentation initialized (window=%d, K=%.1f, min=%d, max=%d, threshold=%.2f)",
             ctx->window_size, ctx->k_factor, ctx->min_length, ctx->max_length, ctx->adaptive_threshold);
}

// Set K factor with validation
bool segmentation_set_k_factor(segmentation_context_t *ctx, float k_factor) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_set_k_factor: NULL context");
        return false;
    }
    
    if (k_factor < SEGMENTATION_K_FACTOR_MIN || k_factor > SEGMENTATION_K_FACTOR_MAX) {
        ESP_LOGE(TAG, "Invalid K factor: %.2f (must be %.1f-%.1f)", 
                 k_factor, SEGMENTATION_K_FACTOR_MIN, SEGMENTATION_K_FACTOR_MAX);
        return false;
    }
    
    ctx->k_factor = k_factor;
    ESP_LOGI(TAG, "K factor updated: %.2f", k_factor);
    return true;
}

// Set window size with validation
bool segmentation_set_window_size(segmentation_context_t *ctx, uint16_t window_size) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_set_window_size: NULL context");
        return false;
    }
    
    if (window_size < SEGMENTATION_WINDOW_SIZE_MIN || window_size > SEGMENTATION_MAX_WINDOW_SIZE) {
        ESP_LOGE(TAG, "Invalid window size: %d (must be %d-%d)", 
                 window_size, SEGMENTATION_WINDOW_SIZE_MIN, SEGMENTATION_MAX_WINDOW_SIZE);
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

// Set minimum segment length with validation
bool segmentation_set_min_length(segmentation_context_t *ctx, uint16_t min_length) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_set_min_length: NULL context");
        return false;
    }
    
    if (min_length < SEGMENTATION_MIN_LENGTH_MIN || min_length > SEGMENTATION_MIN_LENGTH_MAX) {
        ESP_LOGE(TAG, "Invalid min length: %d (must be %d-%d)", 
                 min_length, SEGMENTATION_MIN_LENGTH_MIN, SEGMENTATION_MIN_LENGTH_MAX);
        return false;
    }
    
    ctx->min_length = min_length;
    ESP_LOGI(TAG, "Min segment length updated: %d", min_length);
    return true;
}

// Set maximum segment length with validation
bool segmentation_set_max_length(segmentation_context_t *ctx, uint16_t max_length) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_set_max_length: NULL context");
        return false;
    }
    
    // 0 means no limit
    if (max_length != 0 && (max_length < SEGMENTATION_MAX_LENGTH_MIN || max_length > SEGMENTATION_MAX_LENGTH_MAX)) {
        ESP_LOGE(TAG, "Invalid max length: %d (must be 0 or %d-%d)", 
                 max_length, SEGMENTATION_MAX_LENGTH_MIN, SEGMENTATION_MAX_LENGTH_MAX);
        return false;
    }
    
    ctx->max_length = max_length;
    ESP_LOGI(TAG, "Max segment length updated: %d%s", max_length, max_length == 0 ? " (no limit)" : "");
    return true;
}

// Set threshold directly
bool segmentation_set_threshold(segmentation_context_t *ctx, float threshold) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_set_threshold: NULL context");
        return false;
    }
    
    if (threshold <= 0.0f || threshold > 10.0f) {
        ESP_LOGE(TAG, "Invalid threshold: %.2f (must be 0.0-10.0)", threshold);
        return false;
    }
    
    ctx->adaptive_threshold = threshold;
    ESP_LOGI(TAG, "Threshold updated: %.2f", threshold);
    return true;
}

// Getters
float segmentation_get_k_factor(const segmentation_context_t *ctx) {
    return ctx ? ctx->k_factor : 0.0f;
}

uint16_t segmentation_get_window_size(const segmentation_context_t *ctx) {
    return ctx ? ctx->window_size : 0;
}

uint16_t segmentation_get_min_length(const segmentation_context_t *ctx) {
    return ctx ? ctx->min_length : 0;
}

uint16_t segmentation_get_max_length(const segmentation_context_t *ctx) {
    return ctx ? ctx->max_length : 0;
}

// Calculate moving variance from turbulence buffer
static float calculate_moving_variance(const segmentation_context_t *ctx) {
    // Return 0 if buffer not full yet
    if (ctx->buffer_count < ctx->window_size) {
        return 0.0f;
    }
    
    // Calculate mean of the window
    float mean = 0.0f;
    for (uint16_t i = 0; i < ctx->window_size; i++) {
        mean += ctx->turbulence_buffer[i];
    }
    mean /= ctx->window_size;
    
    // Calculate variance of the window
    float variance = 0.0f;
    for (uint16_t i = 0; i < ctx->window_size; i++) {
        float diff = ctx->turbulence_buffer[i] - mean;
        variance += diff * diff;
    }
    variance /= ctx->window_size;
    
    return variance;
}

// Add turbulence value and update segmentation
bool segmentation_add_turbulence(segmentation_context_t *ctx, float turbulence) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_add_turbulence: NULL context");
        return false;
    }
    
    bool segment_completed = false;
    
    // Add to circular buffer
    ctx->turbulence_buffer[ctx->buffer_index] = turbulence;
    ctx->buffer_index = (ctx->buffer_index + 1) % ctx->window_size;
    if (ctx->buffer_count < ctx->window_size) {
        ctx->buffer_count++;
    }
    
    // Calculate moving variance
    ctx->current_moving_variance = calculate_moving_variance(ctx);
    
    // State machine for segmentation
    if (ctx->state == SEG_STATE_IDLE) {
        // IDLE state: looking for motion start
        if (ctx->current_moving_variance > ctx->adaptive_threshold) {
            // Motion detected - transition to MOTION state
            ctx->state = SEG_STATE_MOTION;
            ctx->motion_start_index = ctx->packet_index;
            ctx->motion_length = 1;
            
            ESP_LOGD(TAG, "Motion started at packet %lu", (unsigned long)ctx->motion_start_index);
        }
    } else {
        // MOTION state: accumulating motion data
        ctx->motion_length++;
        
        // Check for motion end or max length reached
        bool motion_ended = (ctx->current_moving_variance < ctx->adaptive_threshold);
        bool max_length_reached = (ctx->max_length > 0 && ctx->motion_length >= ctx->max_length);
        
        if (motion_ended || max_length_reached) {
            // Validate segment length
            if (ctx->motion_length >= ctx->min_length) {
                // Valid segment completed
                segment_completed = true;
                
                ESP_LOGD(TAG, "Motion segment completed: start=%lu, length=%d (%.2fs)",
                         (unsigned long)ctx->motion_start_index,
                         ctx->motion_length,
                         ctx->motion_length / 20.0f);  // Assuming 20 pps
            } else {
                ESP_LOGD(TAG, "Segment too short (%d < %d) - discarded",
                         ctx->motion_length, ctx->min_length);
            }
            
            // Return to IDLE state
            ctx->state = SEG_STATE_IDLE;
            ctx->motion_length = 0;
        }
    }
    
    ctx->packet_index++;
    ctx->total_packets_processed++;
    
    return segment_completed;
}

// Get current state
segmentation_state_t segmentation_get_state(const segmentation_context_t *ctx) {
    return ctx ? ctx->state : SEG_STATE_IDLE;
}

// Reset segmentation context
void segmentation_reset(segmentation_context_t *ctx) {
    if (!ctx) return;
    
    // Reset state machine ONLY (preserve buffer and parameters)
    ctx->state = SEG_STATE_IDLE;
    ctx->motion_start_index = 0;
    ctx->motion_length = 0;
    ctx->packet_index = 0;
    ctx->total_packets_processed = 0;
    
    // PRESERVE these to avoid "cold start" problem:
    // - ctx->turbulence_buffer (circular buffer with last values)
    // - ctx->buffer_index (current position in circular buffer)
    // - ctx->buffer_count (should stay at window_size after warm-up)
    // - ctx->current_moving_variance (will be recalculated on next packet)
    // - ctx->adaptive_threshold (configured threshold)
    // - ctx->k_factor, window_size, min_length, max_length (configured parameters)
    
    ESP_LOGD(TAG, "Segmentation reset (buffer and parameters preserved)");
}

// Get threshold
float segmentation_get_threshold(const segmentation_context_t *ctx) {
    return ctx ? ctx->adaptive_threshold : 0.0f;
}

// Get current moving variance
float segmentation_get_moving_variance(const segmentation_context_t *ctx) {
    return ctx ? ctx->current_moving_variance : 0.0f;
}

// Get last turbulence value
float segmentation_get_last_turbulence(const segmentation_context_t *ctx) {
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

// Get total packets processed
uint32_t segmentation_get_total_packets(const segmentation_context_t *ctx) {
    return ctx ? ctx->total_packets_processed : 0;
}
