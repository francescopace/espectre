/*
 * ESPectre - CSI Processing Module
 * 
 * Implements Moving Variance Segmentation (MVS) for real-time motion detection
 * using Wi-Fi Channel State Information (CSI).
 * 
 * Motion detection algorithm (MVS):
 * 1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
 * 2. Apply optional Hampel filter to remove outliers
 * 3. Compute moving variance on turbulence signal
 * 4. Apply configurable threshold for motion segmentation
 * 5. Update state machine (IDLE ↔ MOTION)
 * 
 * Key features:
 * - Zero configuration (works with automatic subcarrier selection)
 * - Real-time processing
 * - Configurable sensitivity (threshold and window size)
 * - Optional Hampel filter for noisy environments
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "sdkconfig.h"
#include <cstdint>
#include <cstddef>

namespace esphome {
namespace espectre {

// Segmentation constants
constexpr uint16_t SEGMENTATION_DEFAULT_WINDOW_SIZE = 50;
constexpr uint16_t SEGMENTATION_MIN_WINDOW_SIZE = 10;  // Minimum buffer size
constexpr uint16_t SEGMENTATION_MAX_WINDOW_SIZE = 200;  // Maximum buffer size
constexpr float SEGMENTATION_DEFAULT_THRESHOLD = 1.0f;

// Low-pass filter constants (1st order Butterworth)
constexpr float LOWPASS_CUTOFF_DEFAULT = 11.0f;    // Cutoff frequency in Hz
constexpr float LOWPASS_CUTOFF_MIN = 5.0f;
constexpr float LOWPASS_CUTOFF_MAX = 20.0f;
constexpr float LOWPASS_SAMPLE_RATE = 100.0f;       // Assumed sample rate in Hz

// Hampel filter constants
constexpr float MAD_SCALE_FACTOR = 1.4826f; // Median Absolute Deviation scale factor
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MIN = 3;
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MAX = 11;
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_DEFAULT = 7;
constexpr float HAMPEL_TURBULENCE_THRESHOLD_DEFAULT = 4.0f;

// Low-pass filter state (1st order Butterworth IIR)
struct lowpass_filter_state_t {
    float b0;           // Numerator coefficient
    float a1;           // Denominator coefficient (negated)
    float x_prev;       // Previous input
    float y_prev;       // Previous output
    float cutoff_hz;    // Cutoff frequency
    bool enabled;       // Whether filter is enabled
    bool initialized;   // Whether filter has been initialized with first sample
};

// Hampel turbulence filter state (for MVS preprocessing)
struct hampel_turbulence_state_t {
    float buffer[HAMPEL_TURBULENCE_WINDOW_MAX];       // Circular buffer for values
    float sorted_buffer[HAMPEL_TURBULENCE_WINDOW_MAX]; // Pre-allocated for sorting
    float deviations[HAMPEL_TURBULENCE_WINDOW_MAX];    // Pre-allocated for MAD calc
    uint8_t window_size;  // Actual window size (3-11)
    uint8_t index;
    uint8_t count;
    float threshold;  // Configurable threshold (MAD multiplier)
    bool enabled;     // Whether filter is enabled
};

// ============================================================================
// MOTION DETECTION STATE
// ============================================================================

// Motion detection state
enum csi_motion_state_t {
    CSI_STATE_IDLE,           // No motion detected
    CSI_STATE_MOTION          // Motion in progress
};

// ============================================================================
// CSI PROCESSOR CONTEXT
// ============================================================================

// Unified CSI processor context (combines feature extraction + motion detection)
// Fields ordered by size to minimize padding (pointers/32-bit first, then 16-bit, then 8-bit)
struct csi_processor_context_t {
    // 32-bit aligned fields first (pointers, floats, uint32_t)
    float *turbulence_buffer;           // Turbulence circular buffer (allocated by owner)
    float current_moving_variance;      // Moving variance state
    float threshold;                    // Motion detection threshold value
    float normalization_scale;          // Amplitude normalization factor (default: 1.0)
    uint32_t packet_index;              // Global packet counter
    uint32_t total_packets_processed;   // Statistics
    csi_motion_state_t state;           // State machine (enum = 4 bytes)
    
    // 16-bit fields grouped together
    uint16_t buffer_index;              // Circular buffer write index
    uint16_t buffer_count;              // Number of values in buffer
    uint16_t window_size;               // Moving variance window size (packets)
    
    // Large structs last (already internally aligned)
    lowpass_filter_state_t lowpass_state;    // Low-pass filter state for noise reduction
    hampel_turbulence_state_t hampel_state;  // Hampel filter state for preprocessing
};

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/**
 * Initialize CSI processor context
 * 
 * Allocates and initializes the turbulence buffer internally.
 * 
 * @param ctx CSI processor context to initialize
 * @param window_size Moving variance window size (10-200 packets)
 * @param threshold Motion detection threshold (0.5-10.0)
 * @return true on success, false on allocation failure
 */
bool csi_processor_init(csi_processor_context_t *ctx, 
                        uint16_t window_size, float threshold);

/**
 * Process a CSI packet: calculate turbulence, update motion detection
 * 
 * This is the main entry point for CSI processing. It:
 * 1. Calculates spatial turbulence from the packet
 * 2. Updates the turbulence buffer and moving variance
 * 3. Updates motion detection state
 * 
 * @param ctx CSI processor context
 * @param csi_data Raw CSI data (int8_t array)
 * @param csi_len Length of CSI data
 * @param selected_subcarriers Array of subcarrier indices for turbulence calculation
 * @param num_subcarriers Number of selected subcarriers
 */
void csi_process_packet(csi_processor_context_t *ctx,
                        const int8_t *csi_data,
                        size_t csi_len,
                        const uint8_t *selected_subcarriers,
                        uint8_t num_subcarriers);

/**
 * Reset CSI processor context (clear state machine only)
 * 
 * Resets the state machine (IDLE/MOTION state, packet counters) but preserves:
 * - Turbulence buffer (keeps buffer "warm" to avoid cold start)
 * - Buffer index and count
 * - Configured parameters and threshold
 * 
 * @param ctx CSI processor context
 */
void csi_processor_reset(csi_processor_context_t *ctx);

/**
 * Clear turbulence buffer and reset to cold start
 * 
 * Call this after calibration to avoid stale data causing
 * high initial movement values. Clears:
 * - Turbulence buffer (all zeros)
 * - Buffer index and count
 * - Moving variance
 * - Hampel filter state
 * 
 * @param ctx CSI processor context
 */
void csi_processor_clear_buffer(csi_processor_context_t *ctx);

/**
 * Cleanup CSI processor context (deallocate buffer)
 * 
 * Deallocates the turbulence buffer allocated by csi_processor_init().
 * Call this when the context is no longer needed.
 * 
 * @param ctx CSI processor context
 */
void csi_processor_cleanup(csi_processor_context_t *ctx);

// ============================================================================
// PARAMETER SETTERS
// ============================================================================

/**
 * Set motion detection threshold
 * 
 * @param ctx CSI processor context
 * @param threshold New threshold value (must be positive)
 * @return true if value is valid and was set
 */
bool csi_processor_set_threshold(csi_processor_context_t *ctx, float threshold);

/**
 * Set normalization scale factor
 * 
 * This factor is applied to turbulence values to normalize CSI amplitudes
 * across different ESP32 variants (S3, C6, etc.). Calculated during
 * calibration as TARGET_MEAN / avg_mean.
 * 
 * @param ctx CSI processor context
 * @param scale Normalization scale factor (default: 1.0)
 */
void csi_processor_set_normalization_scale(csi_processor_context_t *ctx, float scale);

/**
 * Get current normalization scale factor
 * 
 * @param ctx CSI processor context
 * @return Current normalization scale (1.0 if not set)
 */
float csi_processor_get_normalization_scale(const csi_processor_context_t *ctx);

/**
 * Enable or disable low-pass filter
 * 
 * @param ctx CSI processor context
 * @param enabled Whether to enable the filter
 */
void csi_processor_set_lowpass_enabled(csi_processor_context_t *ctx, bool enabled);

/**
 * Set low-pass filter cutoff frequency
 * 
 * @param ctx CSI processor context
 * @param cutoff_hz Cutoff frequency in Hz (5.0-20.0)
 */
void csi_processor_set_lowpass_cutoff(csi_processor_context_t *ctx, float cutoff_hz);

/**
 * Get low-pass filter enabled state
 * 
 * @param ctx CSI processor context
 * @return true if filter is enabled
 */
bool csi_processor_get_lowpass_enabled(const csi_processor_context_t *ctx);

/**
 * Get low-pass filter cutoff frequency
 * 
 * @param ctx CSI processor context
 * @return Current cutoff frequency in Hz
 */
float csi_processor_get_lowpass_cutoff(const csi_processor_context_t *ctx);

// ============================================================================
// PARAMETER GETTERS
// ============================================================================

/**
 * Get current window size
 * 
 * @param ctx CSI processor context
 * @return Current window size
 */
uint16_t csi_processor_get_window_size(const csi_processor_context_t *ctx);

/**
 * Get current threshold
 * 
 * @param ctx CSI processor context
 * @return Current threshold value
 */
float csi_processor_get_threshold(const csi_processor_context_t *ctx);

/**
 * Get current motion detection state
 * 
 * @param ctx CSI processor context
 * @return Current state (IDLE or MOTION)
 */
csi_motion_state_t csi_processor_get_state(const csi_processor_context_t *ctx);

/**
 * Get current moving variance
 * 
 * @param ctx CSI processor context
 * @return Current moving variance value
 */
float csi_processor_get_moving_variance(const csi_processor_context_t *ctx);

/**
 * Update state machine with lazy variance calculation
 * 
 * Call this at publish time to calculate the moving variance and update
 * the motion detection state. This implements lazy evaluation - variance
 * is only calculated when needed, saving ~99% CPU compared to per-packet
 * calculation.
 * 
 * @param ctx CSI processor context
 */
void csi_processor_update_state(csi_processor_context_t *ctx);

/**
 * Get last turbulence value added
 * 
 * @param ctx CSI processor context
 * @return Last turbulence value
 */
float csi_processor_get_last_turbulence(const csi_processor_context_t *ctx);

/**
 * Get total packets processed
 * 
 * @param ctx CSI processor context
 * @return Total packets processed
 */
uint32_t csi_processor_get_total_packets(const csi_processor_context_t *ctx);


// ============================================================================
// SUBCARRIER SELECTION
// ============================================================================

/**
 * Set the subcarrier selection for feature extraction
 * This updates the internal configuration used by all CSI processing functions
 * 
 * @param selected_subcarriers Array of subcarrier indices (0-63)
 * @param num_subcarriers Number of selected subcarriers (1-64)
 */
void csi_set_subcarrier_selection(const uint8_t *selected_subcarriers,
                                   uint8_t num_subcarriers);

// ============================================================================
// LOW-PASS FILTER (Noise Reduction)
// ============================================================================

/**
 * Initialize low-pass filter state
 * 
 * Uses 1st order Butterworth IIR filter with bilinear transform.
 * 
 * @param state Low-pass filter state
 * @param cutoff_hz Cutoff frequency in Hz (5.0-20.0, default: 11.0)
 * @param sample_rate_hz Sample rate in Hz (default: 100.0)
 * @param enabled Whether filter is enabled
 */
void lowpass_filter_init(lowpass_filter_state_t *state, float cutoff_hz, float sample_rate_hz, bool enabled);

/**
 * Apply low-pass filter to a single value
 * 
 * @param state Low-pass filter state
 * @param value Input value to filter
 * @return Filtered value (or original if filter disabled)
 */
float lowpass_filter_apply(lowpass_filter_state_t *state, float value);

/**
 * Reset low-pass filter state
 * 
 * @param state Low-pass filter state
 */
void lowpass_filter_reset(lowpass_filter_state_t *state);

// ============================================================================
// HAMPEL FILTER (Turbulence Outlier Removal)
// ============================================================================

/**
 * Initialize Hampel turbulence filter state
 * 
 * @param state Hampel filter state
 * @param window_size Window size for median calculation (3-11)
 * @param threshold MAD multiplier threshold
 * @param enabled Whether filter is enabled
 */
void hampel_turbulence_init(hampel_turbulence_state_t *state, uint8_t window_size, float threshold, bool enabled);

/**
 * Apply Hampel filter for outlier detection
 * 
 * @param window Array of values
 * @param window_size Size of window
 * @param current_value Current value to filter
 * @param threshold MAD multiplier threshold
 * @return Filtered value
 */
float hampel_filter(const float *window, size_t window_size, 
                   float current_value, float threshold);

/**
 * Apply Hampel filter to turbulence value (for MVS preprocessing)
 * 
 * @param state Hampel filter state
 * @param turbulence Current turbulence value
 * @return Filtered turbulence value (or original if filter disabled)
 */
float hampel_filter_turbulence(hampel_turbulence_state_t *state, float turbulence);

}  // namespace espectre
}  // namespace esphome
