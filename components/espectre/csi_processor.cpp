/*
 * ESPectre - CSI Processing Module Implementation
 * 
 * Implements Moving Variance Segmentation (MVS) for motion detection.
 * Includes Hampel filter for turbulence outlier removal.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_processor.h"
#include "utils.h"
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <new>  // For std::nothrow
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "CSI_Processor";

// ============================================================================
// HAMPEL FILTER IMPLEMENTATION
// ============================================================================

/**
 * Insertion sort for small arrays (faster than qsort for N < 15)
 * 
 * Uses in-place sorting with no function call overhead.
 * For Hampel filter with window_size 3-11, this is significantly faster
 * than qsort due to cache locality and no recursion.
 */
static void insertion_sort_float(float *arr, size_t n) {
    for (size_t i = 1; i < n; i++) {
        float key = arr[i];
        size_t j = i;
        while (j > 0 && arr[j - 1] > key) {
            arr[j] = arr[j - 1];
            j--;
        }
        arr[j] = key;
    }
}

void hampel_turbulence_init(hampel_turbulence_state_t *state, uint8_t window_size, float threshold, bool enabled) {
    if (!state) {
        ESP_LOGE(TAG, "hampel_turbulence_init: NULL state pointer");
        return;
    }
    
    // Validate window size
    if (window_size < HAMPEL_TURBULENCE_WINDOW_MIN || window_size > HAMPEL_TURBULENCE_WINDOW_MAX) {
        ESP_LOGW(TAG, "Invalid Hampel window size %d, using default %d", 
                 window_size, HAMPEL_TURBULENCE_WINDOW_DEFAULT);
        window_size = HAMPEL_TURBULENCE_WINDOW_DEFAULT;
    }
    
    std::memset(state->buffer, 0, sizeof(state->buffer));
    std::memset(state->sorted_buffer, 0, sizeof(state->sorted_buffer));
    std::memset(state->deviations, 0, sizeof(state->deviations));
    state->window_size = window_size;
    state->index = 0;
    state->count = 0;
    state->threshold = threshold;
    state->enabled = enabled;
}

float hampel_filter(const float *window, size_t window_size, 
                   float current_value, float threshold) {
    if (!window || window_size < 3) {
        return current_value;
    }
    
    // Create sorted copy for median calculation
    float *sorted = (float*)std::malloc(window_size * sizeof(float));
    if (!sorted) {
        ESP_LOGE(TAG, "hampel_filter: Failed to allocate memory");
        return current_value;
    }
    
    std::memcpy(sorted, window, window_size * sizeof(float));
    std::qsort(sorted, window_size, sizeof(float), compare_float);
    
    // Calculate median
    float median = (window_size % 2 == 1) ? 
                   sorted[window_size / 2] : 
                   (sorted[window_size / 2 - 1] + sorted[window_size / 2]) / 2.0f;
    
    // Calculate absolute deviations
    float *abs_deviations = (float*)std::malloc(window_size * sizeof(float));
    if (!abs_deviations) {
        ESP_LOGE(TAG, "hampel_filter: Failed to allocate memory for deviations");
        std::free(sorted);
        return current_value;
    }
    
    for (size_t i = 0; i < window_size; i++) {
        abs_deviations[i] = std::abs(window[i] - median);
    }
    std::qsort(abs_deviations, window_size, sizeof(float), compare_float);
    
    // Calculate MAD (Median Absolute Deviation)
    float mad = (window_size % 2 == 1) ? 
                abs_deviations[window_size / 2] : 
                (abs_deviations[window_size / 2 - 1] + abs_deviations[window_size / 2]) / 2.0f;
    
    std::free(abs_deviations);
    std::free(sorted);
    
    // Scale MAD to approximate standard deviation
    float mad_scaled = MAD_SCALE_FACTOR * mad;
    
    float deviation = std::abs(current_value - median);
    float threshold_value = threshold * mad_scaled;
    
    // Replace outliers with median
    if (deviation > threshold_value) {
        return median;
    }
    
    return current_value;
}

/**
 * Optimized Hampel filter using pre-allocated buffers
 * 
 * This version uses static buffers in the state struct instead of
 * malloc/free per call. Uses insertion sort instead of qsort for
 * better performance with small window sizes (3-11 elements).
 * 
 * Performance improvement: ~20-30μs saved per packet by eliminating
 * 2x malloc + 2x free + qsort overhead.
 */
float hampel_filter_turbulence(hampel_turbulence_state_t *state, float turbulence) {
    if (!state) {
        ESP_LOGE(TAG, "hampel_filter_turbulence: NULL state pointer");
        return turbulence;
    }
    
    // If filter disabled, return raw value
    if (!state->enabled) {
        return turbulence;
    }
    
    // Add value to circular buffer FIRST (matches original behavior)
    state->buffer[state->index] = turbulence;
    state->index = (state->index + 1) % state->window_size;
    if (state->count < state->window_size) {
        state->count++;
    }
    
    // Need at least 3 values for meaningful Hampel filtering
    if (state->count < 3) {
        return turbulence;
    }
    
    size_t n = state->count;
    
    // Copy buffer to sorted buffer (uses pre-allocated memory)
    std::memcpy(state->sorted_buffer, state->buffer, n * sizeof(float));
    
    // Insertion sort (faster than qsort for small N)
    insertion_sort_float(state->sorted_buffer, n);
    
    // Calculate median
    float median = (n % 2 == 1) ? 
                   state->sorted_buffer[n / 2] : 
                   (state->sorted_buffer[n / 2 - 1] + state->sorted_buffer[n / 2]) / 2.0f;
    
    // Calculate absolute deviations (uses pre-allocated memory)
    for (size_t i = 0; i < n; i++) {
        state->deviations[i] = std::abs(state->buffer[i] - median);
    }
    
    // Sort deviations for MAD calculation
    insertion_sort_float(state->deviations, n);
    
    // Calculate MAD (Median Absolute Deviation)
    float mad = (n % 2 == 1) ? 
                state->deviations[n / 2] : 
                (state->deviations[n / 2 - 1] + state->deviations[n / 2]) / 2.0f;
    
    // Scale MAD to approximate standard deviation
    float mad_scaled = MAD_SCALE_FACTOR * mad;
    
    // Check if current value is an outlier
    float deviation = std::abs(turbulence - median);
    float threshold_value = state->threshold * mad_scaled;
    
    if (deviation > threshold_value) {
        return median;
    }
    
    return turbulence;
}

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

bool csi_processor_init(csi_processor_context_t *ctx, 
                        uint16_t window_size, float threshold) {
    if (!ctx) {
        ESP_LOGE(TAG, "csi_processor_init: NULL context");
        return false;
    }
    
    // Validate window size
    if (window_size < SEGMENTATION_MIN_WINDOW_SIZE || window_size > SEGMENTATION_MAX_WINDOW_SIZE) {
        ESP_LOGE(TAG, "csi_processor_init: Invalid window size: %d (must be %d-%d)", 
                 window_size, SEGMENTATION_MIN_WINDOW_SIZE, SEGMENTATION_MAX_WINDOW_SIZE);
        return false;
    }
    
    // Validate threshold
    if (std::isnan(threshold) || std::isinf(threshold) || threshold < 0.5f || threshold > 10.0f) {
        ESP_LOGE(TAG, "csi_processor_init: Invalid threshold: %.2f (must be 0.5-10.0)", threshold);
        return false;
    }
    
    // Use C++ aggregate initialization
    *ctx = {};
    
    // Allocate turbulence buffer
    ctx->turbulence_buffer = new (std::nothrow) float[window_size];
    if (!ctx->turbulence_buffer) {
        ESP_LOGE(TAG, "csi_processor_init: Failed to allocate turbulence buffer (%d elements, %zu bytes)",
                 window_size, window_size * sizeof(float));
        return false;
    }
    
    // Set parameters
    ctx->window_size = window_size;
    ctx->threshold = threshold;
    ctx->state = CSI_STATE_IDLE;
    
    // Initialize Hampel filter with defaults (will be reconfigured by csi_manager)
    hampel_turbulence_init(&ctx->hampel_state, 
                        HAMPEL_TURBULENCE_WINDOW_DEFAULT,
                        HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, 
                        true);
    
    ESP_LOGI(TAG, "CSI processor initialized (window=%d, threshold=%.2f, buffer=%zu bytes)",
             ctx->window_size, ctx->threshold, window_size * sizeof(float));
    
    return true;
}

// used in test code only
void csi_processor_reset(csi_processor_context_t *ctx) {
    if (!ctx) return;
    
    // Reset state machine ONLY (preserve buffer and parameters)
    ctx->state = CSI_STATE_IDLE;
    ctx->packet_index = 0;
    ctx->total_packets_processed = 0;
}

void csi_processor_cleanup(csi_processor_context_t *ctx) {
    if (!ctx) return;
    
    // Deallocate turbulence buffer
    if (ctx->turbulence_buffer) {
        delete[] ctx->turbulence_buffer;
        ctx->turbulence_buffer = nullptr;
    }
    
    // Reset context to safe state
    *ctx = {};
}

// ============================================================================
// PARAMETER SETTERS
// ============================================================================

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
    
    // Get the most recently added value
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
// MOVING VARIANCE CALCULATION
// ============================================================================

static float calculate_moving_variance(const csi_processor_context_t *ctx) {
    // Return 0 if buffer not full yet
    if (ctx->buffer_count < ctx->window_size) {
        return 0.0f;
    }
    
    return calculate_variance_two_pass(ctx->turbulence_buffer, ctx->window_size);
}

static void add_turbulence_and_update_state(csi_processor_context_t *ctx, float turbulence) {
    // Apply Hampel filter to remove outliers
    float filtered_turbulence = hampel_filter_turbulence(&ctx->hampel_state, turbulence);
    
    // Add to circular buffer
    ctx->turbulence_buffer[ctx->buffer_index] = filtered_turbulence;
    ctx->buffer_index = (ctx->buffer_index + 1) % ctx->window_size;
    if (ctx->buffer_count < ctx->window_size) {
        ctx->buffer_count++;
    }
    
    // Calculate moving variance
    ctx->current_moving_variance = calculate_moving_variance(ctx);
    
    // State machine
    if (ctx->state == CSI_STATE_IDLE) {
        if (ctx->current_moving_variance > ctx->threshold) {
            ctx->state = CSI_STATE_MOTION;
            ESP_LOGV(TAG, "Motion started at packet %lu", (unsigned long)ctx->packet_index);
        }
    } else {
        if (ctx->current_moving_variance < ctx->threshold) {
            ctx->state = CSI_STATE_IDLE;
            ESP_LOGV(TAG, "Motion ended at packet %lu", (unsigned long)ctx->packet_index);
        }
    }
    
    ctx->packet_index++;
    ctx->total_packets_processed++;
}

// ============================================================================
// MAIN PROCESSING FUNCTION
// ============================================================================

void csi_process_packet(csi_processor_context_t *ctx,
                        const int8_t *csi_data,
                        size_t csi_len,
                        const uint8_t *selected_subcarriers,
                        uint8_t num_subcarriers) {
    if (!ctx || !csi_data) {
        ESP_LOGE(TAG, "csi_process_packet: NULL pointer");
        return;
    }
    
    // Calculate spatial turbulence
    float turbulence = calculate_spatial_turbulence_from_csi(csi_data, csi_len,
                                                             selected_subcarriers,
                                                             num_subcarriers);
    
    // Add turbulence to buffer and update motion detection state
    add_turbulence_and_update_state(ctx, turbulence);
}

// ============================================================================
// SUBCARRIER SELECTION
// ============================================================================

void csi_set_subcarrier_selection(const uint8_t *selected_subcarriers,
                                   uint8_t num_subcarriers) {
    if (!selected_subcarriers || num_subcarriers == 0 || num_subcarriers > 64) {
        ESP_LOGE(TAG, "Invalid subcarrier selection parameters");
        return;
    }
    
    ESP_LOGI(TAG, "Subcarrier selection updated: %d subcarriers", num_subcarriers);
}

}  // namespace espectre
}  // namespace esphome
