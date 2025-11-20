/*
 * ESPectre - Moving Variance Segmentation (MVS) Module
 * 
 * Implements temporal segmentation using spatial turbulence and moving variance for motion detection. 
 * 
 * Algorithm:
 * 1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
 * 2. Compute moving variance on turbulence signal (window: 30 packets = 1.5s @ 20Hz)
 * 3. Apply adaptive threshold (mean + K*std, K=2.5)
 * 4. Segment motion using state machine (min: 10 packets, max: 60 packets)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef SEGMENTATION_H
#define SEGMENTATION_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Configuration parameters (optimized from Python testing)
#define SEGMENTATION_WINDOW_SIZE 30      // Moving variance window (packets) - 1.5s @ 20Hz
#define SEGMENTATION_K_FACTOR 2.5f       // Adaptive threshold sensitivity (higher = less sensitive)
#define SEGMENTATION_MIN_LENGTH 10       // Minimum segment length (packets) - 0.5s
#define SEGMENTATION_MAX_LENGTH 60       // Maximum segment length (packets) - 3.0s
#define SEGMENTATION_MAX_SEGMENTS 20     // Maximum concurrent segments to track (Python has no limit)

// Calibration parameters
#define SEGMENTATION_CALIBRATION_MIN_SAMPLES 100  // Minimum samples for threshold calibration

// Default threshold based on empirical testing (test_segmentation_local.py)
// Baseline: mean=1.26, std=0.38 → threshold = 1.26 + 2.5*0.38 = 2.22
// This allows segmentation to work immediately without calibration
#define SEGMENTATION_DEFAULT_THRESHOLD 2.2f

// Segmentation state
typedef enum {
    SEG_STATE_IDLE,           // No motion detected
    SEG_STATE_MOTION          // Motion in progress
} segmentation_state_t;

// Main segmentation context
typedef struct {
    // Turbulence circular buffer
    float turbulence_buffer[SEGMENTATION_WINDOW_SIZE];
    uint16_t buffer_index;
    uint16_t buffer_count;
    
    // Moving variance state
    float current_moving_variance;
    
    // Adaptive threshold (calibrated from baseline)
    float adaptive_threshold;
    float baseline_mean_variance;
    float baseline_std_variance;
    bool threshold_calibrated;
    
    // Calibration state
    float *calibration_variances;  // Dynamic array for calibration
    uint32_t calibration_count;
    uint32_t calibration_target;
    bool calibrating;
    
    // State machine
    segmentation_state_t state;
    uint32_t motion_start_index;
    uint16_t motion_length;
    uint32_t packet_index;         // Global packet counter
    
    // Statistics
    uint32_t total_packets_processed;
    
} segmentation_context_t;

/**
 * Initialize segmentation context
 * 
 * @param ctx Segmentation context to initialize
 */
void segmentation_init(segmentation_context_t *ctx);

/**
 * Start baseline calibration for adaptive threshold
 * 
 * @param ctx Segmentation context
 * @param num_samples Number of baseline samples to collect
 * @return true if calibration started successfully
 */
bool segmentation_start_calibration(segmentation_context_t *ctx, uint32_t num_samples);

/**
 * Add turbulence value to segmentation (during calibration or normal operation)
 * 
 * @param ctx Segmentation context
 * @param turbulence Spatial turbulence value
 * @return true if a new segment was completed
 */
bool segmentation_add_turbulence(segmentation_context_t *ctx, float turbulence);

/**
 * Finalize calibration and calculate adaptive threshold
 * 
 * @param ctx Segmentation context
 * @return true if calibration successful
 */
bool segmentation_finalize_calibration(segmentation_context_t *ctx);

/**
 * Get current segmentation state
 * 
 * @param ctx Segmentation context
 * @return Current state (IDLE or MOTION)
 */
segmentation_state_t segmentation_get_state(const segmentation_context_t *ctx);

/**
 * Reset segmentation context (clear all state)
 * 
 * @param ctx Segmentation context
 */
void segmentation_reset(segmentation_context_t *ctx);

/**
 * Get calibration status
 * 
 * @param ctx Segmentation context
 * @return true if threshold is calibrated
 */
bool segmentation_is_calibrated(const segmentation_context_t *ctx);

/**
 * Get current adaptive threshold
 * 
 * @param ctx Segmentation context
 * @return Adaptive threshold value
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
