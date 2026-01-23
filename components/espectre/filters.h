/*
 * ESPectre - Signal Filters
 * 
 * Low-pass and Hampel filter types and function declarations.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cstddef>

namespace esphome {
namespace espectre {

// =============================================================================
// Low-pass Filter (1st order Butterworth IIR)
// =============================================================================
constexpr float LOWPASS_CUTOFF_DEFAULT = 11.0f;    // Cutoff frequency in Hz
constexpr float LOWPASS_CUTOFF_MIN = 5.0f;
constexpr float LOWPASS_CUTOFF_MAX = 20.0f;
constexpr float LOWPASS_SAMPLE_RATE = 100.0f;      // Assumed sample rate in Hz

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
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MIN = 3;
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MAX = 11;
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_DEFAULT = 7;
constexpr float HAMPEL_TURBULENCE_THRESHOLD_DEFAULT = 4.0f;

struct hampel_turbulence_state_t {
    float buffer[HAMPEL_TURBULENCE_WINDOW_MAX];       // Circular buffer for values
    float sorted_buffer[HAMPEL_TURBULENCE_WINDOW_MAX]; // Pre-allocated for sorting
    float deviations[HAMPEL_TURBULENCE_WINDOW_MAX];    // Pre-allocated for MAD calc
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
float hampel_filter(const float *window, size_t window_size, 
                    float current_value, float threshold);
float hampel_filter_turbulence(hampel_turbulence_state_t *state, float turbulence);

}  // namespace espectre
}  // namespace esphome
