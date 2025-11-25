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
    ctx->window_size = SEGMENTATION_DEFAULT_WINDOW_SIZE;
    ctx->threshold = SEGMENTATION_DEFAULT_THRESHOLD;
    ctx->state = SEG_STATE_IDLE;
    
    ESP_LOGI(TAG, "Segmentation initialized (window=%d, threshold=%.2f)",
             ctx->window_size, ctx->threshold);
}

// Set window size (no validation - caller must validate)
bool segmentation_set_window_size(segmentation_context_t *ctx, uint16_t window_size) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_set_window_size: NULL context");
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


// Set threshold (no validation - caller must validate)
bool segmentation_set_threshold(segmentation_context_t *ctx, float threshold) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_set_threshold: NULL context");
        return false;
    }
    
    ctx->threshold = threshold;
    ESP_LOGI(TAG, "Threshold updated: %.2f", threshold);
    return true;
}

// Getters
uint16_t segmentation_get_window_size(const segmentation_context_t *ctx) {
    return ctx ? ctx->window_size : 0;
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
void segmentation_add_turbulence(segmentation_context_t *ctx, float turbulence) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_add_turbulence: NULL context");
        return;
    }
    
    // Add to circular buffer
    ctx->turbulence_buffer[ctx->buffer_index] = turbulence;
    ctx->buffer_index = (ctx->buffer_index + 1) % ctx->window_size;
    if (ctx->buffer_count < ctx->window_size) {
        ctx->buffer_count++;
    }
    
    // Calculate moving variance
    ctx->current_moving_variance = calculate_moving_variance(ctx);
    
    // State machine for segmentation (simplified)
    if (ctx->state == SEG_STATE_IDLE) {
        // IDLE state: looking for motion start
        if (ctx->current_moving_variance > ctx->threshold) {
            // Motion detected - transition to MOTION state
            ctx->state = SEG_STATE_MOTION;
            ESP_LOGD(TAG, "Motion started at packet %lu", (unsigned long)ctx->packet_index);
        }
    } else {
        // MOTION state: check for motion end
        if (ctx->current_moving_variance < ctx->threshold) {
            // Motion ended - return to IDLE state
            ctx->state = SEG_STATE_IDLE;
            ESP_LOGD(TAG, "Motion ended at packet %lu", (unsigned long)ctx->packet_index);
        }
    }
    
    ctx->packet_index++;
    ctx->total_packets_processed++;
}

// Get current state
segmentation_state_t segmentation_get_state(const segmentation_context_t *ctx) {
    return ctx ? ctx->state : SEG_STATE_IDLE;
}

// Reset segmentation context (unit test only)
void segmentation_reset(segmentation_context_t *ctx) {
    if (!ctx) return;
    
    // Reset state machine ONLY (preserve buffer and parameters)
    ctx->state = SEG_STATE_IDLE;
    ctx->packet_index = 0;
    ctx->total_packets_processed = 0;
    
    // PRESERVE these to avoid "cold start" problem:
    // - ctx->turbulence_buffer (circular buffer with last values)
    // - ctx->buffer_index (current position in circular buffer)
    // - ctx->buffer_count (should stay at window_size after warm-up)
    // - ctx->current_moving_variance (will be recalculated on next packet)
    // - ctx->threshold (configured threshold)
    // - ctx->window_size (configured parameter)
    
    ESP_LOGD(TAG, "Segmentation reset (buffer and parameters preserved)");
}

// Get threshold
float segmentation_get_threshold(const segmentation_context_t *ctx) {
    return ctx ? ctx->threshold : 0.0f;
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
