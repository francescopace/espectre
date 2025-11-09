/*
 * ESPectre - Detection Engine Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "detection_engine.h"
#include "calibration.h"
#include <math.h>
#include <string.h>
#include "esp_log.h"

static const char *TAG = "Detection";

// Numerical stability constant
#define EPSILON_SMALL 1e-6f

void detection_engine_init(detection_engine_state_t *state) {
    if (!state) {
        ESP_LOGE(TAG, "detection_engine_init: NULL state pointer");
        return;
    }
    
    memset(state, 0, sizeof(detection_engine_state_t));
    state->current_state = STATE_IDLE;
    state->consecutive_detections = 0;
    state->last_detection_time = 0;
    state->last_score = 0.0f;
    state->confidence = 0.0f;
}

static inline float normalize_feature(float value, float min_val, float max_val) {
    if (max_val - min_val < EPSILON_SMALL) return 0.0f;
    float normalized = (value - min_val) / (max_val - min_val);
    return fmaxf(0.0f, fminf(1.0f, normalized));
}

float detection_calculate_score(const csi_features_t *features, 
                                const detection_config_t *config) {
    if (!features || !config) {
        ESP_LOGE(TAG, "detection_calculate_score: NULL pointer");
        return 0.0f;
    }
    
    // Normalize features to 0-1 range
    // Ranges determined from real-world testing in various environments
    float norm_variance = normalize_feature(features->variance, 0.0f, 800.0f);
    float norm_spatial_gradient = normalize_feature(features->spatial_gradient, 0.0f, 40.0f);
    float norm_iqr = normalize_feature(features->iqr, 0.0f, 50.0f);
    float norm_entropy = normalize_feature(features->entropy, 4.0f, 7.0f);
    float norm_spatial_correlation = normalize_feature(features->spatial_correlation, -1.0f, 1.0f);
    
    // Weighted combination using feature_weights array
    // Indices: 0=variance, 1=skewness, 2=kurtosis, 3=entropy, 4=iqr,
    //          5=spatial_variance, 6=spatial_correlation, 7=spatial_gradient
    if (!config->feature_weights) {
        ESP_LOGE(TAG, "feature_weights pointer is NULL");
        return 0.0f;
    }
    
    float score = config->feature_weights[0] * norm_variance +           // variance
                  config->feature_weights[3] * norm_entropy +             // entropy
                  config->feature_weights[4] * norm_iqr +                 // iqr
                  config->feature_weights[6] * norm_spatial_correlation + // spatial_correlation
                  config->feature_weights[7] * norm_spatial_gradient;     // spatial_gradient
    
    return score;
}

float detection_calculate_score_calibrated(const csi_features_t *features,
                                           const uint8_t *selected_features,
                                           const float *weights,
                                           uint8_t num_features) {
    if (!features || !selected_features || !weights) {
        ESP_LOGE(TAG, "detection_calculate_score_calibrated: NULL pointer");
        return 0.0f;
    }
    
    if (num_features == 0 || num_features > 15) {
        ESP_LOGE(TAG, "detection_calculate_score_calibrated: invalid num_features=%d", num_features);
        return 0.0f;
    }
    
    // Get calibrated normalization ranges from calibration module
    const float *feature_min = calibration_get_feature_min();
    const float *feature_max = calibration_get_feature_max();
    
    // Map feature index to actual feature value
    // Feature indices (8 total): 0=variance, 1=skewness, 2=kurtosis, 3=entropy, 4=iqr,
    //                            5=spatial_variance, 6=spatial_correlation, 7=spatial_gradient
    
    float score = 0.0f;
    
    for (uint8_t i = 0; i < num_features; i++) {
        uint8_t feat_idx = selected_features[i];
        float weight = weights[i];
        float feature_value = 0.0f;
        
        // Get feature value based on feature type
        // NOTE: Do NOT use fabsf() here - it destroys the sign information
        // which is critical for features like skewness, kurtosis, and correlation.
        // The calibration system captures the actual min/max ranges including
        // negative values, so we preserve the raw feature values.
        switch (feat_idx) {
            case 0: // variance
                feature_value = features->variance;
                break;
            case 1: // skewness (can be negative)
                feature_value = features->skewness;
                break;
            case 2: // kurtosis (can be negative)
                feature_value = features->kurtosis;
                break;
            case 3: // entropy
                feature_value = features->entropy;
                break;
            case 4: // iqr
                feature_value = features->iqr;
                break;
            case 5: // spatial_variance
                feature_value = features->spatial_variance;
                break;
            case 6: // spatial_correlation (can be negative)
                feature_value = features->spatial_correlation;
                break;
            case 7: // spatial_gradient
                feature_value = features->spatial_gradient;
                break;
            default:
                ESP_LOGW(TAG, "Unknown feature index: %d", feat_idx);
                continue;
        }
        
        // Use adaptive normalization with calibrated ranges
        float normalized_value = 0.0f;
        if (feature_min && feature_max) {
            normalized_value = normalize_feature(feature_value, feature_min[i], feature_max[i]);
        } else {
            // Fallback to fixed ranges if calibration data not available
            // NOTE: Updated ranges to support negative values for skewness, kurtosis, and correlation
            ESP_LOGW(TAG, "Calibration ranges not available, using fixed ranges");
            switch (feat_idx) {
                case 0: normalized_value = normalize_feature(feature_value, 0.0f, 400.0f); break;
                case 1: normalized_value = normalize_feature(feature_value, -3.0f, 3.0f); break;  // skewness: -3 to +3
                case 2: normalized_value = normalize_feature(feature_value, -10.0f, 10.0f); break;  // kurtosis: -10 to +10
                case 3: normalized_value = normalize_feature(feature_value, 0.0f, 8.0f); break;
                case 4: normalized_value = normalize_feature(feature_value, 0.0f, 25.0f); break;
                case 5: normalized_value = normalize_feature(feature_value, 0.0f, 400.0f); break;
                case 6: normalized_value = normalize_feature(feature_value, -1.0f, 1.0f); break;  // correlation: -1 to +1
                case 7: normalized_value = normalize_feature(feature_value, 0.0f, 25.0f); break;
            }
        }
        
        score += weight * normalized_value;
    }
    
    return score;
}

void detection_update_state(detection_engine_state_t *engine_state,
                            float score,
                            const detection_config_t *config,
                            int64_t current_time) {
    if (!engine_state || !config) {
        ESP_LOGE(TAG, "detection_update_state: NULL pointer");
        return;
    }
    
    engine_state->last_score = score;
    
    // Calculate confidence
    if (config->threshold_high > EPSILON_SMALL) {
        engine_state->confidence = fminf(1.0f, score / (config->threshold_high + EPSILON_SMALL));
    } else {
        engine_state->confidence = (score > 0) ? 1.0f : 0.0f;
    }
    
    // Determine new state based on score and thresholds
    detection_state_t new_state;
    
    // 2-state mode: IDLE, DETECTED
    if (score > config->threshold_high) {
        new_state = STATE_DETECTED;
    } else {
        new_state = STATE_IDLE;
    }
    
    // State transition logic with debouncing
    if (new_state > engine_state->current_state) {
        // Upgrading state - require consecutive detections
        engine_state->consecutive_detections++;
        if (engine_state->consecutive_detections >= config->debounce_count) {
            engine_state->current_state = new_state;
            engine_state->last_detection_time = current_time;
        }
    } else if (new_state < engine_state->current_state) {
        // Downgrading state - use persistence timeout
        if (current_time - engine_state->last_detection_time > config->persistence_timeout) {
            engine_state->consecutive_detections = 0;
            
            engine_state->current_state = new_state;
            engine_state->confidence = (new_state == STATE_IDLE) ? 0.0f : engine_state->confidence;
        }
    } else {
        // new_state == current_state
        // Reset consecutive counter if we're in IDLE and score is below threshold
        if (engine_state->current_state == STATE_IDLE && score <= config->threshold_high) {
            engine_state->consecutive_detections = 0;
        }
    }
}

const char* detection_state_to_string(detection_state_t state) {
    switch (state) {
        case STATE_IDLE:
            return "IDLE";
        case STATE_DETECTED:
            return "DETECTED";
        default:
            return "UNKNOWN";
    }
}
