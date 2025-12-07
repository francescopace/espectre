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

// Hampel filter constants
constexpr float MAD_SCALE_FACTOR = 1.4826f; // Median Absolute Deviation scale factor
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MIN = 3;
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MAX = 11;
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_DEFAULT = 7;
constexpr float HAMPEL_TURBULENCE_THRESHOLD_DEFAULT = 4.0f;

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
struct csi_processor_context_t {
    // Turbulence circular buffer (pointer to buffer allocated by owner)
    // Buffer size must be at least window_size elements
    float *turbulence_buffer;
    uint16_t buffer_index;
    uint16_t buffer_count;
    
    // Hampel filter state for turbulence preprocessing
    hampel_turbulence_state_t hampel_state;
    
    // Moving variance state
    float current_moving_variance;
    
    // Configurable parameters
    uint16_t window_size;        // Moving variance window size (packets)
    float threshold;             // Motion detection threshold value
    
    // State machine
    csi_motion_state_t state;
    uint32_t packet_index;         // Global packet counter
    
    // Statistics
    uint32_t total_packets_processed;
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
