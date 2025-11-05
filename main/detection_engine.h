/*
 * ESPectre - Detection Engine Module
 * 
 * Movement detection state machine and scoring algorithm:
 * - Combines CSI features with configurable weights
 * - Debouncing and hysteresis for stability
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef DETECTION_ENGINE_H
#define DETECTION_ENGINE_H

#include <stdint.h>
#include <stdbool.h>
#include "csi_processor.h"

// Detection states
typedef enum {
    STATE_IDLE,
    STATE_DETECTED
} detection_state_t;

// Detection configuration
typedef struct {
    float threshold_high;
    float threshold_low;
    uint8_t debounce_count;
    int persistence_timeout;
    
    // Feature weights array
    // Indices: 0=variance, 1=skewness, 2=kurtosis, 3=entropy, 4=iqr,
    //          5=spatial_variance, 6=spatial_correlation, 7=spatial_gradient
    const float *feature_weights;  // Pointer to weights array from config
} detection_config_t;

// Detection result
typedef struct {
    float score;
    float confidence;
    detection_state_t state;
    int64_t timestamp;
} detection_result_t;

// Detection engine state
typedef struct {
    detection_state_t current_state;
    uint8_t consecutive_detections;
    int64_t last_detection_time;
    float last_score;
    float confidence;
} detection_engine_state_t;

/**
 * Initialize detection engine
 * 
 * @param state Pointer to detection engine state
 */
void detection_engine_init(detection_engine_state_t *state);

/**
 * Calculate detection score from CSI features
 * Combines multiple features with configurable weights
 * 
 * @param features Extracted CSI features
 * @param config Detection configuration
 * @return Detection score (0.0 to 1.0)
 */
float detection_calculate_score(const csi_features_t *features, 
                                const detection_config_t *config);

/**
 * Calculate detection score using calibrated features
 * Uses dynamically selected features and weights from calibration
 * 
 * @param features Extracted CSI features
 * @param selected_features Array of selected feature indices
 * @param weights Array of feature weights
 * @param num_features Number of selected features
 * @return Detection score (0.0 to 1.0)
 */
float detection_calculate_score_calibrated(const csi_features_t *features,
                                           const uint8_t *selected_features,
                                           const float *weights,
                                           uint8_t num_features);

/**
 * Update detection state based on score
 * Handles debouncing, hysteresis, and state transitions
 * 
 * @param engine_state Detection engine state
 * @param score Current detection score
 * @param config Detection configuration
 * @param current_time Current timestamp in seconds
 */
void detection_update_state(detection_engine_state_t *engine_state,
                            float score,
                            const detection_config_t *config,
                            int64_t current_time);

/**
 * Get state name as string
 * 
 * @param state Detection state
 * @return State name string
 */
const char* detection_state_to_string(detection_state_t state);

#endif // DETECTION_ENGINE_H
