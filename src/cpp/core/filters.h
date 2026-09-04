/*
 * ESPectre - Signal Filters
 *
 * Low-pass and Hampel filters used by the shared signal-processing
 * pipeline.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <cstddef>

#include "filter_config.h"

namespace espectre {

// =============================================================================
// Low-pass Filter (1st order Butterworth IIR)
// =============================================================================
struct lowpass_filter_state_t {
    float b0;           // Numerator coefficient
    float a1;           // Denominator coefficient (negated)
    float x_prev;       // Previous input
    float y_prev;       // Previous output
    float cutoff_hz;    // Cutoff frequency
    bool enabled;       // Whether filter is enabled
    bool initialized;   // Whether filter has been initialized with first sample
};

void lowpass_filter_init(lowpass_filter_state_t *state, float cutoff_hz, 
                         float sample_rate_hz, bool enabled);
float lowpass_filter_apply(lowpass_filter_state_t *state, float value);
void lowpass_filter_reset(lowpass_filter_state_t *state);

// =============================================================================
// Hampel Filter (MAD-based outlier removal)
// =============================================================================
constexpr float MAD_SCALE_FACTOR = 1.4826f;        // Median Absolute Deviation scale factor
struct hampel_turbulence_state_t {
    float buffer[HAMPEL_TURBULENCE_WINDOW_MAX];       // Circular buffer for values
    uint8_t window_size;  // Actual window size (3-11)
    uint8_t index;
    uint8_t count;
    float threshold;      // Configurable threshold (MAD multiplier)
    bool enabled;         // Whether filter is enabled
};

// Alias for cleaner naming
using hampel_filter_state_t = hampel_turbulence_state_t;

void hampel_turbulence_init(hampel_turbulence_state_t *state, uint8_t window_size,
                            float threshold, bool enabled);

/**
 * Stateless Hampel decision over a caller-owned window.
 *
 * Shared numeric core: hampel_filter_turbulence() pushes into its ring and then
 * defers here, so the outlier rule exists in exactly one place. The scratch is
 * two stack arrays bounded by HAMPEL_TURBULENCE_WINDOW_MAX (11 floats each),
 * which is small enough for the CSI callback stack.
 *
 * @return median when current_value is an outlier, otherwise current_value
 */
float hampel_filter(const float *window, size_t window_size,
                    float current_value, float threshold);
float hampel_filter_turbulence(hampel_turbulence_state_t *state, float turbulence);

}  // namespace espectre
