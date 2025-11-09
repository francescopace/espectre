/*
 * ESPectre - Calibration Module
 * 
 * Automatic guided calibration system that:
 * - Collects baseline and movement data
 * - Analyzes feature discriminability using Fisher's criterion
 * - Selects top 4-6 most discriminant features
 * - Calculates optimal weights
 * - Reduces runtime CPU usage by 30-40%
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef CALIBRATION_H
#define CALIBRATION_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

// Calibration constants
#define CALIBRATION_DEFAULT_SAMPLES 1000  // default samples per phase
#define CALIBRATION_MIN_SAMPLES     50    // minimum samples needed
#define CALIBRATION_MAX_SAMPLES     10000 // maximum samples allowed
#define MAX_SELECTED_FEATURES       6     // maximum features to select
#define NUM_TOTAL_FEATURES          10    // total available features (statistical + spatial + temporal)

// Threshold bounds (shared with NVS validation)
#define THRESHOLD_MIN               0.15f // Minimum allowed threshold
#define THRESHOLD_MAX               0.80f // Maximum allowed threshold
#define EPSILON_SMALL               1e-6f // For division by zero checks

// Calibration mode
typedef enum {
    CALIB_MODE_NORMAL,      // uses only selected features
    CALIB_MODE_COLLECTING,  // collects all features
} calibration_mode_t;

// Calibration phase
typedef enum {
    CALIB_IDLE,
    CALIB_BASELINE,
    CALIB_MOVEMENT,
    CALIB_ANALYZING
} calibration_phase_t;

// Online statistics for a single feature (Welford's algorithm)
typedef struct {
    float mean;
    float m2;  // for variance calculation
    size_t count;
    float min_val;  // minimum value seen
    float max_val;  // maximum value seen
} feature_stats_t;

// Calibration state
typedef struct {
    calibration_mode_t mode;
    calibration_phase_t phase;
    
    // Statistics for each feature (baseline and movement)
    feature_stats_t baseline_stats[NUM_TOTAL_FEATURES];
    feature_stats_t movement_stats[NUM_TOTAL_FEATURES];
    
    // Results
    uint8_t selected_features[MAX_SELECTED_FEATURES];
    float optimized_weights[MAX_SELECTED_FEATURES];
    uint8_t num_selected;
    float optimal_threshold;
    
    // Feature normalization ranges (for adaptive normalization)
    float feature_min[MAX_SELECTED_FEATURES];
    float feature_max[MAX_SELECTED_FEATURES];
    
    // Filter configuration backup (saved before calibration)
    bool saved_butterworth_enabled;
    bool saved_wavelet_enabled;
    int saved_wavelet_level;
    float saved_wavelet_threshold;
    bool saved_hampel_enabled;
    float saved_hampel_threshold;
    bool saved_savgol_enabled;
    int saved_savgol_window_size;
    bool saved_adaptive_normalizer_enabled;
    float saved_adaptive_normalizer_alpha;
    uint32_t saved_adaptive_normalizer_reset_timeout;
    
    // Recommended filter settings (calculated during analysis)
    bool recommended_butterworth;
    bool recommended_wavelet;
    int recommended_wavelet_level;
    float recommended_wavelet_threshold;
    bool recommended_hampel;
    float recommended_hampel_threshold;
    bool recommended_savgol;
    bool recommended_adaptive_normalizer;
    float recommended_normalizer_alpha;
    
    // Sample-based calibration tracking
    uint32_t phase_target_samples;              // Target samples for current phase
    uint32_t samples_in_current_phase;          // Samples collected in current phase
    uint32_t baseline_movement_target_samples;  // Target samples for baseline/movement phases
    uint32_t traffic_rate;                      // Traffic rate for estimated duration display
    
    // Results for recap
    float baseline_mean_score;
    float movement_mean_score;
    float separation_ratio;
} calibration_state_t;

// Feature array for passing all features
typedef struct {
    float features[NUM_TOTAL_FEATURES];
} feature_array_t;

// Initialize calibration system
void calibration_init(void);

// Start calibration process (pass runtime_config_t pointer to save/restore filters, and normalizer to reset)
bool calibration_start(int target_samples, void *config, void *normalizer);

// Stop calibration process (pass runtime_config_t pointer to restore filters)
void calibration_stop(void *config);

// Update calibration with new feature values (called during collection phases)
void calibration_update(const feature_array_t *features);

// Check if calibration phase has completed and advance to next phase
void calibration_check_completion(void);

// Get current calibration status
calibration_phase_t calibration_get_phase(void);
calibration_mode_t calibration_get_mode(void);
bool calibration_is_active(void);

// Get calibration results
uint8_t calibration_get_num_selected(void);
const uint8_t* calibration_get_selected_features(void);
const float* calibration_get_weights(void);
float calibration_get_threshold(void);

// Get calibration progress (0.0 to 1.0)
float calibration_get_progress(void);

// Get number of samples collected in current phase
uint32_t calibration_get_samples_collected(void);

// Get calibration statistics for reporting
void calibration_get_stats(char *buffer, size_t buffer_size);

// Get current calibration results (for saving to NVS)
void calibration_get_results(calibration_state_t *state);

// Apply saved calibration results (from NVS)
void calibration_apply_saved(const calibration_state_t *state);

// Acknowledge calibration completion (called after saving to NVS)
void calibration_acknowledge_completion(void);

// Get recommended filter configuration from calibration analysis
void calibration_get_filter_config(bool *butterworth, bool *wavelet, int *wavelet_level, 
                                   float *wavelet_threshold, bool *hampel, float *hampel_threshold,
                                   bool *savgol, bool *adaptive_norm, float *norm_alpha);

// Get feature normalization ranges (for adaptive normalization in detection engine)
const float* calibration_get_feature_min(void);
const float* calibration_get_feature_max(void);

// Test helpers (for unit testing only)
void calibration_force_phase(calibration_phase_t phase);
void calibration_trigger_analysis(void);

#endif // CALIBRATION_H
