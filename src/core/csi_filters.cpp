/*
 * ESPectre - CSI Filters Implementation
 * 
 * Low-pass and Hampel filter implementations for signal processing.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "filters.h"
#include "utils.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "CSI_Filters";

// ============================================================================
// LOW-PASS FILTER IMPLEMENTATION
// ============================================================================

void lowpass_filter_init(lowpass_filter_state_t *state, float cutoff_hz, float sample_rate_hz, bool enabled) {
    if (!state) {
        ESP_LOGE(TAG, "lowpass_filter_init: NULL state pointer");
        return;
    }
    
    // Clamp cutoff to valid range
    if (cutoff_hz < LOWPASS_CUTOFF_MIN) cutoff_hz = LOWPASS_CUTOFF_MIN;
    if (cutoff_hz > LOWPASS_CUTOFF_MAX) cutoff_hz = LOWPASS_CUTOFF_MAX;
    
    state->cutoff_hz = cutoff_hz;
    state->enabled = enabled;
    state->initialized = false;
    state->x_prev = 0.0f;
    state->y_prev = 0.0f;
    
    // Calculate filter coefficients using bilinear transform
    float wc = tanf(M_PI * cutoff_hz / sample_rate_hz);
    float k = 1.0f + wc;
    
    state->b0 = wc / k;
    state->a1 = (wc - 1.0f) / k;
    
    ESP_LOGD(TAG, "LowPass filter initialized: cutoff=%.1f Hz, enabled=%d", cutoff_hz, enabled);
}

float lowpass_filter_apply(lowpass_filter_state_t *state, float value) {
    if (!state || !state->enabled) {
        return value;
    }
    
    if (!state->initialized) {
        state->x_prev = value;
        state->y_prev = value;
        state->initialized = true;
        return value;
    }
    
    float y = state->b0 * value + state->b0 * state->x_prev - state->a1 * state->y_prev;
    state->x_prev = value;
    state->y_prev = y;
    
    return y;
}

void lowpass_filter_reset(lowpass_filter_state_t *state) {
    if (!state) return;
    state->x_prev = 0.0f;
    state->y_prev = 0.0f;
    state->initialized = false;
}

// ============================================================================
// HAMPEL FILTER IMPLEMENTATION
// ============================================================================

void hampel_turbulence_init(hampel_turbulence_state_t *state, uint8_t window_size, float threshold, bool enabled) {
    if (!state) {
        ESP_LOGE(TAG, "hampel_turbulence_init: NULL state pointer");
        return;
    }
    
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
    
    // Stack allocation - window_size is bounded (3-11 max)
    float sorted[HAMPEL_TURBULENCE_WINDOW_MAX];
    float abs_deviations[HAMPEL_TURBULENCE_WINDOW_MAX];
    
    // Clamp to max supported window size
    if (window_size > HAMPEL_TURBULENCE_WINDOW_MAX) {
        window_size = HAMPEL_TURBULENCE_WINDOW_MAX;
    }
    
    std::memcpy(sorted, window, window_size * sizeof(float));
    float median = calculate_median_float(sorted, window_size);
    
    for (size_t i = 0; i < window_size; i++) {
        abs_deviations[i] = std::abs(window[i] - median);
    }
    float mad = calculate_median_float(abs_deviations, window_size);
    
    float mad_scaled = MAD_SCALE_FACTOR * mad;
    float deviation = std::abs(current_value - median);
    
    if (deviation > threshold * mad_scaled) {
        return median;
    }
    
    return current_value;
}

float hampel_filter_turbulence(hampel_turbulence_state_t *state, float turbulence) {
    if (!state || !state->enabled) {
        return turbulence;
    }
    
    state->buffer[state->index] = turbulence;
    state->index = (state->index + 1) % state->window_size;
    if (state->count < state->window_size) {
        state->count++;
    }
    
    if (state->count < 3) {
        return turbulence;
    }
    
    size_t n = state->count;
    std::memcpy(state->sorted_buffer, state->buffer, n * sizeof(float));
    float median = calculate_median_float(state->sorted_buffer, n);
    
    for (size_t i = 0; i < n; i++) {
        state->deviations[i] = std::abs(state->buffer[i] - median);
    }
    float mad = calculate_median_float(state->deviations, n);
    
    float deviation = std::abs(turbulence - median);
    
    if (deviation > state->threshold * MAD_SCALE_FACTOR * mad) {
        return median;
    }
    
    return turbulence;
}

}  // namespace espectre
}  // namespace esphome
