/*
 * ESPectre - Moving Variance Segmentation (MVS) Module
 * 
 * Implements temporal segmentation using spatial turbulence and moving variance for motion detection. 
 * 
 * Algorithm:
 * 1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
 * 2. Compute moving variance on turbulence signal
 * 3. Apply configurable threshold
 * 4. Segment motion using state machine
 * 
 * All parameters are now configurable at runtime via MQTT commands.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include "sdkconfig.h"

#include <stdint.h>
#include <stdbool.h>
#include "espectre.h"

// Segmentation state
typedef enum {
    SEG_STATE_IDLE,           // No motion detected
    SEG_STATE_MOTION          // Motion in progress
} segmentation_state_t;

// Main segmentation context
typedef struct {
    // Turbulence circular buffer (allocated at max size to support runtime window_size changes)
    // Only the first window_size elements are used (window_size can be 10-200)
    float turbulence_buffer[SEGMENTATION_WINDOW_SIZE_MAX];
    uint16_t buffer_index;
    uint16_t buffer_count;
    
    // Moving variance state
    float current_moving_variance;
    
    // Configurable parameters
    uint16_t window_size;        // Moving variance window size (packets)
    float threshold;             // Motion detection threshold value
    
    // State machine
    segmentation_state_t state;
    uint32_t packet_index;         // Global packet counter
    
    // Statistics
    uint32_t total_packets_processed;
    
} segmentation_context_t;

/**
 * Initialize segmentation context with default parameters
 * 
 * @param ctx Segmentation context to initialize
 */
void segmentation_init(segmentation_context_t *ctx);

/**
 * Set window size for moving variance
 * 
 * @param ctx Segmentation context
 * @param window_size New window size (10 - 200 packets)
 * @return true if value is valid and was set
 */
bool segmentation_set_window_size(segmentation_context_t *ctx, uint16_t window_size);


/**
 * Set threshold directly
 * 
 * @param ctx Segmentation context
 * @param threshold New threshold value (must be positive)
 * @return true if value is valid and was set
 */
bool segmentation_set_threshold(segmentation_context_t *ctx, float threshold);

/**
 * Get current window size
 * 
 * @param ctx Segmentation context
 * @return Current window size
 */
uint16_t segmentation_get_window_size(const segmentation_context_t *ctx);

/**
 * Add turbulence value to segmentation
 * 
 * @param ctx Segmentation context
 * @param turbulence Spatial turbulence value
 */
void segmentation_add_turbulence(segmentation_context_t *ctx, float turbulence);

/**
 * Get current segmentation state
 * 
 * @param ctx Segmentation context
 * @return Current state (IDLE or MOTION)
 */
segmentation_state_t segmentation_get_state(const segmentation_context_t *ctx);

/**
 * Reset segmentation context (clear state machine only)
 * 
 * Resets the state machine (IDLE/MOTION state, packet counters) but preserves:
 * - Turbulence buffer (keeps buffer "warm" to avoid cold start)
 * - Buffer index and count
 * - Configured parameters and threshold
 * 
 * This prevents the "cold start" problem where the first window_size packets
 * after reset would have moving_variance = 0, causing detection issues.
 * 
 * NOTE: This function is primarily used by unit tests to reset state between
 * test phases. It is NOT used in the main application code (espectre.c).
 * 
 * @param ctx Segmentation context
 */
void segmentation_reset(segmentation_context_t *ctx);

/**
 * Get current threshold
 * 
 * @param ctx Segmentation context
 * @return Current threshold value
 */
float segmentation_get_threshold(const segmentation_context_t *ctx);

/**
 * Get current moving variance
 * 
 * @param ctx Segmentation context
 * @return Current moving variance value
 */
float segmentation_get_moving_variance(const segmentation_context_t *ctx);

/**
 * Get last turbulence value added
 * 
 * @param ctx Segmentation context
 * @return Last turbulence value
 */
float segmentation_get_last_turbulence(const segmentation_context_t *ctx);

/**
 * Get total packets processed
 * 
 * @param ctx Segmentation context
 * @return Total packets processed
 */
uint32_t segmentation_get_total_packets(const segmentation_context_t *ctx);

#endif // SEGMENTATION_H
