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
#include <stddef.h>

// Maximum buffer size for turbulence window (fixed allocation)
#define SEGMENTATION_MAX_WINDOW_SIZE 50

// Parameter limits for validation
#define SEGMENTATION_K_FACTOR_MIN 0.5f
#define SEGMENTATION_K_FACTOR_MAX 5.0f
#define SEGMENTATION_WINDOW_SIZE_MIN 3
#define SEGMENTATION_MIN_LENGTH_MIN 5
#define SEGMENTATION_MIN_LENGTH_MAX 100
#define SEGMENTATION_MAX_LENGTH_MIN 10
#define SEGMENTATION_MAX_LENGTH_MAX 200

// Default configuration parameters (optimized from testing)
#define SEGMENTATION_DEFAULT_K_FACTOR 2.5f
#define SEGMENTATION_DEFAULT_WINDOW_SIZE 30
#define SEGMENTATION_DEFAULT_MIN_LENGTH 10
#define SEGMENTATION_DEFAULT_MAX_LENGTH 60
#define SEGMENTATION_DEFAULT_THRESHOLD 3.0f

// Segmentation state
typedef enum {
    SEG_STATE_IDLE,           // No motion detected
    SEG_STATE_MOTION          // Motion in progress
} segmentation_state_t;

// Main segmentation context
typedef struct {
    // Turbulence circular buffer (fixed size, use first window_size elements)
    float turbulence_buffer[SEGMENTATION_MAX_WINDOW_SIZE];
    uint16_t buffer_index;
    uint16_t buffer_count;
    
    // Moving variance state
    float current_moving_variance;
    
    // Configurable parameters
    float k_factor;              // Threshold sensitivity multiplier
    uint16_t window_size;        // Moving variance window size (packets)
    uint16_t min_length;         // Minimum segment length (packets)
    uint16_t max_length;         // Maximum segment length (packets)
    float adaptive_threshold;    // Current threshold value
    
    // State machine
    segmentation_state_t state;
    uint32_t motion_start_index;
    uint16_t motion_length;
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
 * Set K factor (threshold sensitivity)
 * 
 * @param ctx Segmentation context
 * @param k_factor New K factor value (0.5 - 5.0)
 * @return true if value is valid and was set
 */
bool segmentation_set_k_factor(segmentation_context_t *ctx, float k_factor);

/**
 * Set window size for moving variance
 * 
 * @param ctx Segmentation context
 * @param window_size New window size (3 - 50 packets)
 * @return true if value is valid and was set
 */
bool segmentation_set_window_size(segmentation_context_t *ctx, uint16_t window_size);

/**
 * Set minimum segment length
 * 
 * @param ctx Segmentation context
 * @param min_length New minimum length (5 - 100 packets)
 * @return true if value is valid and was set
 */
bool segmentation_set_min_length(segmentation_context_t *ctx, uint16_t min_length);

/**
 * Set maximum segment length
 * 
 * @param ctx Segmentation context
 * @param max_length New maximum length (10 - 200 packets, 0 = no limit)
 * @return true if value is valid and was set
 */
bool segmentation_set_max_length(segmentation_context_t *ctx, uint16_t max_length);

/**
 * Set threshold directly
 * 
 * @param ctx Segmentation context
 * @param threshold New threshold value (must be positive)
 * @return true if value is valid and was set
 */
bool segmentation_set_threshold(segmentation_context_t *ctx, float threshold);

/**
 * Get current K factor
 * 
 * @param ctx Segmentation context
 * @return Current K factor value
 */
float segmentation_get_k_factor(const segmentation_context_t *ctx);

/**
 * Get current window size
 * 
 * @param ctx Segmentation context
 * @return Current window size
 */
uint16_t segmentation_get_window_size(const segmentation_context_t *ctx);

/**
 * Get current minimum segment length
 * 
 * @param ctx Segmentation context
 * @return Current minimum length
 */
uint16_t segmentation_get_min_length(const segmentation_context_t *ctx);

/**
 * Get current maximum segment length
 * 
 * @param ctx Segmentation context
 * @return Current maximum length (0 = no limit)
 */
uint16_t segmentation_get_max_length(const segmentation_context_t *ctx);

/**
 * Add turbulence value to segmentation
 * 
 * @param ctx Segmentation context
 * @param turbulence Spatial turbulence value
 * @return true if a new segment was completed
 */
bool segmentation_add_turbulence(segmentation_context_t *ctx, float turbulence);

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
