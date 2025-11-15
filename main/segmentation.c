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
    ctx->state = SEG_STATE_IDLE;
    ctx->calibrating = false;
    ctx->calibration_variances = NULL;
    
    // Set default threshold (can be overridden by calibration)
    ctx->adaptive_threshold = SEGMENTATION_DEFAULT_THRESHOLD;
    ctx->threshold_calibrated = true;  // Enable segmentation with default threshold
    
    ESP_LOGI(TAG, "Segmentation initialized (window=%d, K=%.1f, min=%d, max=%d)",
             SEGMENTATION_WINDOW_SIZE, SEGMENTATION_K_FACTOR,
             SEGMENTATION_MIN_LENGTH, SEGMENTATION_MAX_LENGTH);
    ESP_LOGI(TAG, "Using default threshold: %.2f (from empirical testing)", 
             SEGMENTATION_DEFAULT_THRESHOLD);
}

// Start baseline calibration
bool segmentation_start_calibration(segmentation_context_t *ctx, uint32_t num_samples) {
    if (!ctx) {
        ESP_LOGE(TAG, "segmentation_start_calibration: NULL context");
        return false;
    }
    
    if (num_samples < SEGMENTATION_CALIBRATION_MIN_SAMPLES) {
        ESP_LOGE(TAG, "Insufficient calibration samples: %lu (min: %d)",
                 (unsigned long)num_samples, SEGMENTATION_CALIBRATION_MIN_SAMPLES);
        return false;
    }
    
    // Allocate calibration buffer
    ctx->calibration_variances = (float*)malloc(num_samples * sizeof(float));
    if (!ctx->calibration_variances) {
        ESP_LOGE(TAG, "Failed to allocate calibration buffer");
        return false;
    }
    
    ctx->calibration_target = num_samples;
    ctx->calibration_count = 0;
    ctx->calibrating = true;
    ctx->threshold_calibrated = false;
    
    // Reset buffers
    memset(ctx->turbulence_buffer, 0, sizeof(ctx->turbulence_buffer));
    ctx->buffer_index = 0;
    ctx->buffer_count = 0;
    ctx->packet_index = 0;
    
    ESP_LOGI(TAG, "Calibration started (target: %lu samples)", (unsigned long)num_samples);
    
    return true;
}

// Calculate moving variance from turbulence buffer
static float calculate_moving_variance(const segmentation_context_t *ctx) {
    // Return 0 if buffer not full yet
    if (ctx->buffer_count < SEGMENTATION_WINDOW_SIZE) {
        return 0.0f;
    }
    
    // Calculate mean of the FULL window (all SEGMENTATION_WINDOW_SIZE samples)
    float mean = 0.0f;
    for (uint16_t i = 0; i < SEGMENTATION_WINDOW_SIZE; i++) {
        mean += ctx->turbulence_buffer[i];
    }
    mean /= SEGMENTATION_WINDOW_SIZE;
    
    // Calculate variance of the FULL window
    float variance = 0.0f;
    for (uint16_t i = 0; i < SEGMENTATION_WINDOW_SIZE; i++) {
        float diff = ctx->turbulence_buffer[i] - mean;
        variance += diff * diff;
    }
    variance /= SEGMENTATION_WINDOW_SIZE;
    
    return variance;
}

// Finalize calibration and calculate adaptive threshold
bool segmentation_finalize_calibration(segmentation_context_t *ctx) {
    if (!ctx || !ctx->calibrating) {
        ESP_LOGE(TAG, "segmentation_finalize_calibration: invalid state");
        return false;
    }
    
    if (ctx->calibration_count < SEGMENTATION_CALIBRATION_MIN_SAMPLES) {
        ESP_LOGE(TAG, "Insufficient calibration data: %lu samples",
                 (unsigned long)ctx->calibration_count);
        free(ctx->calibration_variances);
        ctx->calibration_variances = NULL;
        ctx->calibrating = false;
        return false;
    }
    
    // Calculate mean and std of variance values
    float mean = 0.0f;
    for (uint32_t i = 0; i < ctx->calibration_count; i++) {
        mean += ctx->calibration_variances[i];
    }
    mean /= ctx->calibration_count;
    
    float variance = 0.0f;
    for (uint32_t i = 0; i < ctx->calibration_count; i++) {
        float diff = ctx->calibration_variances[i] - mean;
        variance += diff * diff;
    }
    variance /= ctx->calibration_count;
    float std = sqrtf(variance);
    
    // Calculate adaptive threshold: mean + K * std
    ctx->baseline_mean_variance = mean;
    ctx->baseline_std_variance = std;
    ctx->adaptive_threshold = mean + SEGMENTATION_K_FACTOR * std;
    ctx->threshold_calibrated = true;
    
    ESP_LOGI(TAG, "Calibration complete:");
    ESP_LOGI(TAG, "  Samples: %lu", (unsigned long)ctx->calibration_count);
    ESP_LOGI(TAG, "  Mean variance: %.4f", mean);
    ESP_LOGI(TAG, "  Std variance: %.4f", std);
    ESP_LOGI(TAG, "  Adaptive threshold: %.4f (mean + %.1f*std)",
             ctx->adaptive_threshold, SEGMENTATION_K_FACTOR);
    
    // Free calibration buffer
    free(ctx->calibration_variances);
    ctx->calibration_variances = NULL;
    ctx->calibrating = false;
    
    return true;
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
    ctx->buffer_index = (ctx->buffer_index + 1) % SEGMENTATION_WINDOW_SIZE;
    if (ctx->buffer_count < SEGMENTATION_WINDOW_SIZE) {
        ctx->buffer_count++;
    }
    
    // Calculate moving variance
    ctx->current_moving_variance = calculate_moving_variance(ctx);
    
    // During calibration: collect variance values
    if (ctx->calibrating) {
        // Only collect after window is full
        if (ctx->buffer_count >= SEGMENTATION_WINDOW_SIZE) {
            if (ctx->calibration_count < ctx->calibration_target) {
                ctx->calibration_variances[ctx->calibration_count] = ctx->current_moving_variance;
                ctx->calibration_count++;
            }
        }
        ctx->packet_index++;
        return false;  // No segmentation during calibration
    }
    
    // Normal operation: segmentation with state machine
    if (!ctx->threshold_calibrated) {
        ESP_LOGW(TAG, "Threshold not calibrated - skipping segmentation");
        ctx->packet_index++;
        return false;
    }
    
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
        bool max_length_reached = (ctx->motion_length >= SEGMENTATION_MAX_LENGTH);
        
        if (motion_ended || max_length_reached) {
            // Validate segment length
            if (ctx->motion_length >= SEGMENTATION_MIN_LENGTH) {
                // Valid segment - add to list if space available
                if (ctx->num_segments < SEGMENTATION_MAX_SEGMENTS) {
                    segment_t *seg = &ctx->segments[ctx->num_segments];
                    seg->start_index = ctx->motion_start_index;
                    seg->length = ctx->motion_length;
                    seg->active = true;
                    
                    // Calculate segment statistics
                    float sum = 0.0f;
                    float max_val = 0.0f;
                    uint16_t count = 0;
                    
                    // Get turbulence values from segment
                    for (uint16_t i = 0; i < ctx->motion_length && i < SEGMENTATION_WINDOW_SIZE; i++) {
                        // Calculate buffer index for this packet
                        int16_t buf_idx = (int16_t)ctx->buffer_index - 1 - i;
                        if (buf_idx < 0) buf_idx += SEGMENTATION_WINDOW_SIZE;
                        
                        float val = ctx->turbulence_buffer[buf_idx];
                        sum += val;
                        if (val > max_val) max_val = val;
                        count++;
                    }
                    
                    seg->avg_turbulence = (count > 0) ? (sum / count) : 0.0f;
                    seg->max_turbulence = max_val;
                    
                    ctx->num_segments++;
                    ctx->total_segments_detected++;
                    segment_completed = true;
                    
                    ESP_LOGD(TAG, "Segment #%d: start=%lu, length=%d (%.2fs), avg=%.2f, max=%.2f",
                             ctx->num_segments,
                             (unsigned long)seg->start_index,
                             seg->length,
                             seg->length / 20.0f,  // Assuming 20 pps
                             seg->avg_turbulence,
                             seg->max_turbulence);
                } else {
                    ESP_LOGW(TAG, "Segment buffer full - discarding segment");
                }
            } else {
                ESP_LOGD(TAG, "Segment too short (%d < %d) - discarded",
                         ctx->motion_length, SEGMENTATION_MIN_LENGTH);
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

// Get number of segments
uint8_t segmentation_get_num_segments(const segmentation_context_t *ctx) {
    return ctx ? ctx->num_segments : 0;
}

// Get segment by index
const segment_t* segmentation_get_segment(const segmentation_context_t *ctx, uint8_t index) {
    if (!ctx || index >= ctx->num_segments) {
        return NULL;
    }
    return &ctx->segments[index];
}

// Clear all segments
void segmentation_clear_segments(segmentation_context_t *ctx) {
    if (!ctx) return;
    
    memset(ctx->segments, 0, sizeof(ctx->segments));
    ctx->num_segments = 0;
}

// Reset segmentation context
void segmentation_reset(segmentation_context_t *ctx) {
    if (!ctx) return;
    
    // Free calibration buffer if allocated
    if (ctx->calibration_variances) {
        free(ctx->calibration_variances);
        ctx->calibration_variances = NULL;
    }
    
    // Reset all state
    memset(ctx->turbulence_buffer, 0, sizeof(ctx->turbulence_buffer));
    ctx->buffer_index = 0;
    ctx->buffer_count = 0;
    ctx->current_moving_variance = 0.0f;
    ctx->state = SEG_STATE_IDLE;
    ctx->motion_start_index = 0;
    ctx->motion_length = 0;
    ctx->packet_index = 0;
    
    segmentation_clear_segments(ctx);
    
    ctx->total_segments_detected = 0;
    ctx->total_packets_processed = 0;
    
    // Keep calibration results (threshold, mean, std)
    // Don't reset: threshold_calibrated, adaptive_threshold, baseline_mean_variance, baseline_std_variance
    
    ESP_LOGI(TAG, "Segmentation reset (threshold preserved)");
}

// Get calibration status
bool segmentation_is_calibrated(const segmentation_context_t *ctx) {
    return ctx ? ctx->threshold_calibrated : false;
}

// Get adaptive threshold
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
        last_idx = SEGMENTATION_WINDOW_SIZE - 1;
    }
    
    return ctx->turbulence_buffer[last_idx];
}

// Get count of active segments
uint8_t segmentation_get_active_segments_count(const segmentation_context_t *ctx) {
    if (!ctx) {
        return 0;
    }
    
    uint8_t count = 0;
    for (uint8_t i = 0; i < ctx->num_segments; i++) {
        if (ctx->segments[i].active) {
            count++;
        }
    }
    
    return count;
}

// Get last completed segment
const segment_t* segmentation_get_last_completed_segment(const segmentation_context_t *ctx) {
    if (!ctx || ctx->num_segments == 0) {
        return NULL;
    }
    
    // Return the most recently added segment (last in array)
    return &ctx->segments[ctx->num_segments - 1];
}

// Get total packets processed
uint32_t segmentation_get_total_packets(const segmentation_context_t *ctx) {
    return ctx ? ctx->total_packets_processed : 0;
}
