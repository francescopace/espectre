/*
 * ESPectre - Unified CSI Processing Module
 * 
 * Combines CSI feature extraction with Moving Variance Segmentation (MVS) for motion detection.
 * 
 * Features extracted from CSI data:
 * - Statistical (5): variance, skewness (amplitude-based), kurtosis, entropy, IQR
 * - Spatial (3): variance, correlation, gradient (across subcarriers within packet)
 * - Temporal (2): delta_mean, delta_variance (changes between consecutive packets)
 * 
 * Motion detection algorithm (MVS):
 * 1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
 * 2. Compute moving variance on turbulence signal
 * 3. Apply configurable threshold
 * 4. Segment motion using state machine
 * 
 * NOTE: Skewness/kurtosis are calculated on the turbulence buffer (moving window)
 *       for better separation between baseline and movement.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef CSI_PROCESSOR_H
#define CSI_PROCESSOR_H

#include "sdkconfig.h"

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "espectre.h"

// Numerical stability constant
#define EPSILON_SMALL               1e-6f

// ============================================================================
// MOTION DETECTION STATE
// ============================================================================

// Motion detection state
typedef enum {
    CSI_STATE_IDLE,           // No motion detected
    CSI_STATE_MOTION          // Motion in progress
} csi_motion_state_t;

// ============================================================================
// CSI PROCESSOR CONTEXT
// ============================================================================

// Unified CSI processor context (combines feature extraction + motion detection)
typedef struct {
    // Turbulence circular buffer (allocated at max size to support runtime window_size changes)
    // Only the first window_size elements are used (window_size can be 10-200)
    float turbulence_buffer[SEGMENTATION_WINDOW_SIZE_MAX];
    uint16_t buffer_index;
    uint16_t buffer_count;
    
    // Hampel filter buffer for turbulence preprocessing
    // Used to remove outliers before adding to turbulence_buffer
    float hampel_buffer[HAMPEL_TURBULENCE_WINDOW];
    uint8_t hampel_index;
    uint8_t hampel_count;
    
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
    
} csi_processor_context_t;

// ============================================================================
// CSI FEATURES
// ============================================================================

// CSI features extracted from raw data
typedef struct {
    // Statistical features (5)
    float variance;
    float skewness;
    float kurtosis;
    float entropy;
    float iqr;  // Interquartile range
    
    // Spatial features (3)
    float spatial_variance;
    float spatial_correlation;
    float spatial_gradient;
    
    // Temporal features (2) - changes between consecutive packets
    float temporal_delta_mean;      // Average absolute difference from previous packet
    float temporal_delta_variance;  // Variance of differences from previous packet
} csi_features_t;

// ============================================================================
// CONTEXT MANAGEMENT
// ============================================================================

/**
 * Initialize CSI processor context with default parameters
 * 
 * @param ctx CSI processor context to initialize
 */
void csi_processor_init(csi_processor_context_t *ctx);

/**
 * Process a CSI packet: calculate turbulence, update motion detection, extract features
 * 
 * This is the main entry point for CSI processing. It:
 * 1. Calculates spatial turbulence from the packet
 * 2. Updates the turbulence buffer and moving variance
 * 3. Updates motion detection state
 * 4. Optionally extracts features if requested
 * 
 * @param ctx CSI processor context
 * @param csi_data Raw CSI data (int8_t array)
 * @param csi_len Length of CSI data
 * @param selected_subcarriers Array of subcarrier indices for turbulence calculation
 * @param num_subcarriers Number of selected subcarriers
 * @param features Output structure for extracted features (can be NULL to skip)
 * @param selected_features Array of feature indices to calculate (ignored if features is NULL)
 * @param num_features Number of features to calculate (ignored if features is NULL)
 */
void csi_process_packet(csi_processor_context_t *ctx,
                        const int8_t *csi_data,
                        size_t csi_len,
                        const uint8_t *selected_subcarriers,
                        uint8_t num_subcarriers,
                        csi_features_t *features,
                        const uint8_t *selected_features,
                        uint8_t num_features);

/**
 * Extract features from CSI data (orchestrator function)
 * 
 * This function orchestrates feature extraction by calling individual
 * feature calculation functions from csi_features.c. It has direct access
 * to the turbulence buffer for skewness/kurtosis calculation.
 * 
 * Can be called standalone for testing or when features are needed without
 * motion detection.
 * 
 * @param csi_data Raw CSI data (int8_t array)
 * @param csi_len Length of CSI data
 * @param turbulence_buffer Buffer of turbulence values for skewness/kurtosis (can be NULL)
 * @param turbulence_count Number of valid values in turbulence buffer
 * @param features Output structure for extracted features
 * @param selected_features Array of feature indices to calculate
 * @param num_features Number of features to calculate
 */
void csi_extract_features(const int8_t *csi_data,
                         size_t csi_len,
                         const float *turbulence_buffer,
                         uint16_t turbulence_count,
                         csi_features_t *features,
                         const uint8_t *selected_features,
                         uint8_t num_features);

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

// ============================================================================
// PARAMETER SETTERS
// ============================================================================

/**
 * Set window size for moving variance
 * 
 * @param ctx CSI processor context
 * @param window_size New window size (10 - 200 packets)
 * @return true if value is valid and was set
 */
bool csi_processor_set_window_size(csi_processor_context_t *ctx, uint16_t window_size);

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

#endif // CSI_PROCESSOR_H
