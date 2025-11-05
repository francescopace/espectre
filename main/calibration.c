/*
 * ESPectre - Calibration Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "calibration.h"
#include "traffic_generator.h"
#include <string.h>
#include <math.h>
#include <stdio.h>
#include "esp_log.h"
#include "esp_timer.h"

// Feature selection thresholds
#define FISHER_MIN_RATIO    0.2f    // Minimum Fisher score ratio for feature selection
#define SAFETY_MARGIN       1.05f   // Safety margin for threshold calculation (5%)

static const char *TAG = "Calibration";

// Global calibration state
static calibration_state_t g_calib = {
    .mode = CALIB_MODE_NORMAL,
    .phase = CALIB_IDLE,
    .num_selected = 0,
};

// Global config pointer (saved during calibration_start)
static void *g_config_ptr = NULL;

// CSI data capture for debugging (distributed sampling across entire phase)
#define MAX_CAPTURE_PACKETS 50  // Capture 50 packets distributed across the phase
#define CSI_PACKET_SIZE 128
static int8_t g_captured_baseline[MAX_CAPTURE_PACKETS][CSI_PACKET_SIZE];
static int8_t g_captured_movement[MAX_CAPTURE_PACKETS][CSI_PACKET_SIZE];
static size_t g_baseline_capture_count = 0;
static size_t g_movement_capture_count = 0;
static uint32_t g_baseline_capture_interval = 0;  // Capture every Nth packet
static uint32_t g_movement_capture_interval = 0;
static uint32_t g_baseline_packet_counter = 0;
static uint32_t g_movement_packet_counter = 0;
static int8_t g_last_csi_packet[CSI_PACKET_SIZE];
static bool g_has_last_packet = false;

// Feature names for logging (8 features: 5 time-domain + 3 spatial)
static const char* feature_names[NUM_TOTAL_FEATURES] = {
    "variance", "skewness", "kurtosis", "entropy", "iqr",
    "spatial_variance", "spatial_correlation", "spatial_gradient"
};


// Welford's algorithm for online mean and variance calculation
static void welford_update(feature_stats_t *stats, float value) {
    stats->count++;
    float delta = value - stats->mean;
    stats->mean += delta / stats->count;
    float delta2 = value - stats->mean;
    stats->m2 += delta * delta2;
    
    // Track min/max for adaptive normalization
    if (stats->count == 1) {
        stats->min_val = value;
        stats->max_val = value;
    } else {
        if (value < stats->min_val) stats->min_val = value;
        if (value > stats->max_val) stats->max_val = value;
    }
}

// Get variance from Welford's algorithm state
static float welford_get_variance(const feature_stats_t *stats) {
    if (stats->count < 2) return 0.0f;
    return stats->m2 / (stats->count - 1);
}

// Fisher's criterion: measures class separability
// Higher value = better discrimination between baseline and movement
static float calculate_fisher_score(const feature_stats_t *baseline, 
                                    const feature_stats_t *movement) {
    if (baseline->count < 2 || movement->count < 2) {
        return 0.0f;
    }
    
    float mean_diff = fabsf(movement->mean - baseline->mean);
    float var_baseline = welford_get_variance(baseline);
    float var_movement = welford_get_variance(movement);
    float var_sum = var_baseline + var_movement;
    
    if (var_sum < 1e-6f) {
        return 0.0f;
    }
    
    // Fisher's criterion: (μ1 - μ2)² / (σ1² + σ2²)
    return (mean_diff * mean_diff) / var_sum;
}

// Comparison function for sorting features by Fisher score
typedef struct {
    uint8_t index;
    float score;
} feature_score_t;

static int compare_feature_scores(const void *a, const void *b) {
    const feature_score_t *fa = (const feature_score_t*)a;
    const feature_score_t *fb = (const feature_score_t*)b;
    
    // Sort descending (highest scores first)
    if (fa->score > fb->score) return -1;
    if (fa->score < fb->score) return 1;
    return 0;
}

// Helper function: Calculate outlier ratio using MAD (Median Absolute Deviation)
static float calculate_outlier_ratio(const feature_stats_t *stats, size_t num_features) {
    if (!stats || num_features == 0) return 0.0f;
    
    // Count features with high variance relative to mean
    size_t outlier_count = 0;
    for (size_t i = 0; i < num_features; i++) {
        if (stats[i].count < 2) continue;
        
        float variance = welford_get_variance(&stats[i]);
        float mean = stats[i].mean;
        
        // Consider as outlier if variance is very high relative to mean
        if (mean > 0.01f && variance > (mean * mean * 4.0f)) {
            outlier_count++;
        }
    }
    
    return (float)outlier_count / num_features;
}

// Helper function: Calculate signal-to-noise ratio
static float calculate_snr(const feature_stats_t *baseline, 
                          const feature_stats_t *movement,
                          size_t num_features) {
    if (!baseline || !movement || num_features == 0) return 0.0f;
    
    float total_snr = 0.0f;
    size_t valid_count = 0;
    
    for (size_t i = 0; i < num_features; i++) {
        if (baseline[i].count < 2 || movement[i].count < 2) continue;
        
        float baseline_std = sqrtf(welford_get_variance(&baseline[i]));
        float signal_diff = fabsf(movement[i].mean - baseline[i].mean);
        
        if (baseline_std > 1e-6f) {
            total_snr += signal_diff / baseline_std;
            valid_count++;
        }
    }
    
    return valid_count > 0 ? total_snr / valid_count : 0.0f;
}


// Forward declaration for debugging function
static void calibration_dump_captured_data(void);

// Helper function: Calculate baseline drift (variance of variance over time)
static float calculate_baseline_drift(const feature_stats_t *stats, size_t num_features) {
    if (!stats || num_features == 0) return 0.0f;
    
    // Calculate coefficient of variation across features
    float total_cv = 0.0f;
    size_t valid_count = 0;
    
    for (size_t i = 0; i < num_features; i++) {
        if (stats[i].count < 2) continue;
        
        float variance = welford_get_variance(&stats[i]);
        float mean = stats[i].mean;
        
        if (mean > 1e-6f) {
            float cv = sqrtf(variance) / mean;
            total_cv += cv;
            valid_count++;
        }
    }
    
    return valid_count > 0 ? total_cv / valid_count : 0.0f;
}

// Analyze collected data and select best features
static void analyze_and_select_features(void *config) {
    ESP_LOGI(TAG, "🔬 Analyzing collected data...");
    
    // Validate we have enough data
    if (g_calib.baseline_stats[0].count < CALIBRATION_MIN_SAMPLES ||
        g_calib.movement_stats[0].count < CALIBRATION_MIN_SAMPLES) {
        ESP_LOGE(TAG, "Insufficient samples for analysis (baseline: %u, movement: %u)",
                 (unsigned int)g_calib.baseline_stats[0].count, 
                 (unsigned int)g_calib.movement_stats[0].count);
        return;
    }
    
    // Calculate Fisher scores for all features
    feature_score_t scores[NUM_TOTAL_FEATURES];
    bool has_valid_scores = false;
    
    for (uint8_t i = 0; i < NUM_TOTAL_FEATURES; i++) {
        scores[i].index = i;
        scores[i].score = calculate_fisher_score(
            &g_calib.baseline_stats[i],
            &g_calib.movement_stats[i]
        );
        
        // Check if we have at least one valid score
        if (scores[i].score > 0.0f) {
            has_valid_scores = true;
        }
        
        ESP_LOGD(TAG, "Feature %s: Fisher=%.4f (baseline: μ=%.4f σ²=%.4f n=%zu, movement: μ=%.4f σ²=%.4f n=%zu)",
                 feature_names[i],
                 scores[i].score,
                 g_calib.baseline_stats[i].mean,
                 welford_get_variance(&g_calib.baseline_stats[i]),
                 g_calib.baseline_stats[i].count,
                 g_calib.movement_stats[i].mean,
                 welford_get_variance(&g_calib.movement_stats[i]),
                 g_calib.movement_stats[i].count);
    }
    
    // Validate we have valid scores before sorting
    if (!has_valid_scores) {
        ESP_LOGE(TAG, "No valid Fisher scores calculated - calibration failed");
        return;
    }
    
    // Sort features by Fisher score
    qsort(scores, NUM_TOTAL_FEATURES, sizeof(feature_score_t), compare_feature_scores);
    
    // Select top features (at least 4, up to 6)
    g_calib.num_selected = 4;  // minimum
    
    // Add more features if they have significant discriminability
    for (uint8_t i = 4; i < MAX_SELECTED_FEATURES && i < NUM_TOTAL_FEATURES; i++) {
        // Only add if Fisher score meets minimum ratio threshold
        if (scores[i].score >= FISHER_MIN_RATIO * scores[0].score) {
            g_calib.num_selected++;
        } else {
            break;
        }
    }
    
    // Store selected feature indices and calculate normalization ranges
    for (uint8_t i = 0; i < g_calib.num_selected; i++) {
        uint8_t feat_idx = scores[i].index;
        g_calib.selected_features[i] = feat_idx;
        
        // Calculate normalization range from both baseline and movement data
        float baseline_min = g_calib.baseline_stats[feat_idx].min_val;
        float baseline_max = g_calib.baseline_stats[feat_idx].max_val;
        float movement_min = g_calib.movement_stats[feat_idx].min_val;
        float movement_max = g_calib.movement_stats[feat_idx].max_val;
        
        // Use the overall min/max across both phases
        g_calib.feature_min[i] = fminf(baseline_min, movement_min);
        g_calib.feature_max[i] = fmaxf(baseline_max, movement_max);
        
        // Add 10% margin to avoid edge effects
        float range = g_calib.feature_max[i] - g_calib.feature_min[i];
        if (range > EPSILON_SMALL) {
            g_calib.feature_min[i] -= range * 0.1f;
            g_calib.feature_max[i] += range * 0.1f;
        } else {
            // If range is too small, use a default range
            float center = (g_calib.feature_min[i] + g_calib.feature_max[i]) / 2.0f;
            g_calib.feature_min[i] = center - 0.05f;
            g_calib.feature_max[i] = center + 0.05f;
        }
    }
    
    // Calculate weights proportional to Fisher scores
    float total_score = 0.0f;
    for (uint8_t i = 0; i < g_calib.num_selected; i++) {
        total_score += scores[i].score;
    }
    
    // Use epsilon threshold to prevent division by very small numbers
    if (total_score > 1e-6f) {
        for (uint8_t i = 0; i < g_calib.num_selected; i++) {
            g_calib.optimized_weights[i] = scores[i].score / total_score;
        }
    } else {
        // Fallback to equal weights if total score is too small
        ESP_LOGW(TAG, "Total Fisher score too small (%.6f), using equal weights", total_score);
        for (uint8_t i = 0; i < g_calib.num_selected; i++) {
            g_calib.optimized_weights[i] = 1.0f / g_calib.num_selected;
        }
    }
    
    // Calculate optimal threshold using normalized scores
    // First, normalize each feature using the calibrated ranges
    float baseline_mean_score = 0.0f;
    float movement_mean_score = 0.0f;
    
    for (uint8_t i = 0; i < g_calib.num_selected; i++) {
        uint8_t feat_idx = g_calib.selected_features[i];
        float weight = g_calib.optimized_weights[i];
        
        // Normalize using the calibrated ranges
        float range = g_calib.feature_max[i] - g_calib.feature_min[i];
        float baseline_norm = 0.0f;
        float movement_norm = 0.0f;
        
        if (range > EPSILON_SMALL) {
            baseline_norm = (g_calib.baseline_stats[feat_idx].mean - g_calib.feature_min[i]) / range;
            movement_norm = (g_calib.movement_stats[feat_idx].mean - g_calib.feature_min[i]) / range;
            
            // Clamp to [0, 1]
            baseline_norm = fmaxf(0.0f, fminf(1.0f, baseline_norm));
            movement_norm = fmaxf(0.0f, fminf(1.0f, movement_norm));
        }
        
        baseline_mean_score += baseline_norm * weight;
        movement_mean_score += movement_norm * weight;
    }
    
    // Validate separation between baseline and movement
    float separation_ratio = 0.0f;
    if (baseline_mean_score > EPSILON_SMALL) {
        separation_ratio = movement_mean_score / baseline_mean_score;
    }
    
    // Store results for recap
    g_calib.baseline_mean_score = baseline_mean_score;
    g_calib.movement_mean_score = movement_mean_score;
    g_calib.separation_ratio = separation_ratio;
    
    // Check if we have sufficient separation (movement should be at least 2x baseline)
    if (separation_ratio < 2.0f) {
        ESP_LOGW(TAG, "⚠️  Low separation between baseline and movement (ratio: %.2f)", separation_ratio);
        ESP_LOGW(TAG, "    Baseline: %.4f, Movement: %.4f", baseline_mean_score, movement_mean_score);
        ESP_LOGW(TAG, "    Consider performing calibration with more intense movement");
    }
    
    // Calculate weighted variance scores for threshold optimization
    float baseline_var_score = 0.0f;
    float movement_var_score = 0.0f;
    
    for (uint8_t i = 0; i < g_calib.num_selected; i++) {
        uint8_t feat_idx = g_calib.selected_features[i];
        float weight = g_calib.optimized_weights[i];
        float var = welford_get_variance(&g_calib.baseline_stats[feat_idx]);
        baseline_var_score += var * weight * weight;
    }
    
    for (uint8_t i = 0; i < g_calib.num_selected; i++) {
        uint8_t feat_idx = g_calib.selected_features[i];
        float weight = g_calib.optimized_weights[i];
        float var = welford_get_variance(&g_calib.movement_stats[feat_idx]);
        movement_var_score += var * weight * weight;
    }
    
    // Fisher's optimal threshold (weighted by variances)
    float var_sum = baseline_var_score + movement_var_score;
    if (var_sum >= EPSILON_SMALL) {
        g_calib.optimal_threshold = 
            (movement_var_score * baseline_mean_score + 
             baseline_var_score * movement_mean_score) / var_sum;
    } else {
        // Fallback to midpoint
        g_calib.optimal_threshold = (baseline_mean_score + movement_mean_score) / 2.0f;
    }
    
    // Apply safety margin based on separation quality
    // If separation is poor, use a more conservative threshold
    float safety_factor = SAFETY_MARGIN;
    if (separation_ratio < 2.0f) {
        // Poor separation: move threshold closer to movement mean
        safety_factor = 1.0f + (separation_ratio - 1.0f) * 0.5f;
        safety_factor = fmaxf(1.0f, fminf(safety_factor, SAFETY_MARGIN));
    }
    
    g_calib.optimal_threshold *= safety_factor;
    
    // Ensure threshold is between baseline and movement means
    float min_threshold = baseline_mean_score + (movement_mean_score - baseline_mean_score) * 0.3f;
    float max_threshold = baseline_mean_score + (movement_mean_score - baseline_mean_score) * 0.7f;
    
    if (g_calib.optimal_threshold < min_threshold) {
        g_calib.optimal_threshold = min_threshold;
        ESP_LOGD(TAG, "Threshold adjusted to minimum: %.4f", min_threshold);
    }
    if (g_calib.optimal_threshold > max_threshold) {
        g_calib.optimal_threshold = max_threshold;
        ESP_LOGD(TAG, "Threshold adjusted to maximum: %.4f", max_threshold);
    }
    
    // Final clamp to absolute bounds
    if (g_calib.optimal_threshold < THRESHOLD_MIN) g_calib.optimal_threshold = THRESHOLD_MIN;
    if (g_calib.optimal_threshold > THRESHOLD_MAX) g_calib.optimal_threshold = THRESHOLD_MAX;
    
    // Log results
    ESP_LOGI(TAG, "✅ Calibration complete! Selected %d features:", g_calib.num_selected);
    for (uint8_t i = 0; i < g_calib.num_selected; i++) {
        uint8_t feat_idx = g_calib.selected_features[i];
        ESP_LOGI(TAG, "  %d. %s (Fisher=%.4f, weight=%.3f)",
                 i + 1,
                 feature_names[feat_idx],
                 scores[i].score,
                 g_calib.optimized_weights[i]);
    }
    
    ESP_LOGI(TAG, "🎯 Optimal threshold: %.4f (baseline: %.4f, movement: %.4f)",
             g_calib.optimal_threshold, baseline_mean_score, movement_mean_score);
    
    // Analyze data to determine optimal filter configuration
    ESP_LOGI(TAG, "🔧 Analyzing optimal filter configuration...");
    
    // 1. Butterworth filter - always recommended (removes high-freq noise >8Hz)
    g_calib.recommended_butterworth = true;
    
    // 2. Hampel filter - based on outlier detection
    float outlier_ratio = calculate_outlier_ratio(g_calib.baseline_stats, NUM_TOTAL_FEATURES);
    if (outlier_ratio > 0.05f) {  // >5% outliers detected
        g_calib.recommended_hampel = true;
        // Adjust threshold based on outlier severity
        if (outlier_ratio > 0.15f) {
            g_calib.recommended_hampel_threshold = 3.0f;  // more tolerant
        } else {
            g_calib.recommended_hampel_threshold = 2.0f;  // standard
        }
        ESP_LOGI(TAG, "  Hampel filter: ON (threshold: %.1f) - detected %.1f%% outliers",
                 g_calib.recommended_hampel_threshold, outlier_ratio * 100.0f);
    } else {
        g_calib.recommended_hampel = false;
        ESP_LOGI(TAG, "  Hampel filter: OFF - low outlier rate (%.1f%%)", outlier_ratio * 100.0f);
    }
    
    // 3. Savitzky-Golay filter - based on signal-to-noise ratio
    float snr = calculate_snr(g_calib.baseline_stats, g_calib.movement_stats, NUM_TOTAL_FEATURES);
    if (snr < 10.0f) {  // Low SNR = noisy signal
        g_calib.recommended_savgol = true;
        ESP_LOGI(TAG, "  Savitzky-Golay filter: ON - SNR=%.2f (noisy signal)", snr);
    } else {
        g_calib.recommended_savgol = false;
        ESP_LOGI(TAG, "  Savitzky-Golay filter: OFF - SNR=%.2f (clean signal)", snr);
    }
    
    // 4. Wavelet filter - based on baseline variance (NEW)
    // Check variance feature (index 0) in baseline
    float baseline_variance = g_calib.baseline_stats[0].mean;
    if (baseline_variance > 500.0f) {  // High noise environment
        g_calib.recommended_wavelet = true;
        g_calib.recommended_wavelet_level = 3;  // Maximum denoising
        
        // Adjust threshold based on noise level
        if (baseline_variance > 800.0f) {
            g_calib.recommended_wavelet_threshold = 2.0f;  // Very aggressive
        } else if (baseline_variance > 600.0f) {
            g_calib.recommended_wavelet_threshold = 1.5f;  // Aggressive
        } else {
            g_calib.recommended_wavelet_threshold = 1.0f;  // Balanced
        }
        ESP_LOGI(TAG, "  Wavelet filter: ON (level: %d, threshold: %.1f) - high baseline variance=%.1f",
                 g_calib.recommended_wavelet_level, 
                 g_calib.recommended_wavelet_threshold,
                 baseline_variance);
    } else {
        g_calib.recommended_wavelet = false;
        g_calib.recommended_wavelet_level = 3;
        g_calib.recommended_wavelet_threshold = 1.0f;
        ESP_LOGI(TAG, "  Wavelet filter: OFF - low baseline variance=%.1f", baseline_variance);
    }
    
    // 5. Adaptive normalizer - based on baseline stability
    float baseline_drift = calculate_baseline_drift(g_calib.baseline_stats, NUM_TOTAL_FEATURES);
    if (baseline_drift > 0.1f) {  // Significant drift detected
        g_calib.recommended_adaptive_normalizer = true;
        // Adjust alpha based on drift rate
        if (baseline_drift > 0.3f) {
            g_calib.recommended_normalizer_alpha = 0.02f;  // faster adaptation
        } else {
            g_calib.recommended_normalizer_alpha = 0.01f;  // standard
        }
        ESP_LOGI(TAG, "  Adaptive normalizer: ON (alpha: %.3f) - baseline drift=%.3f",
                 g_calib.recommended_normalizer_alpha, baseline_drift);
    } else {
        g_calib.recommended_adaptive_normalizer = false;
        ESP_LOGI(TAG, "  Adaptive normalizer: OFF - stable baseline (drift=%.3f)", baseline_drift);
    }
    
    ESP_LOGI(TAG, "  Butterworth filter: ON (always recommended for noise reduction)");
    
    // Apply recommended filter configuration (if config provided)
    if (config) {
        typedef struct {
            bool csi_logs_enabled;
            bool verbose_mode;
            uint8_t debounce_count;
            float hysteresis_ratio;
            int persistence_timeout;
            float variance_scale;
            bool hampel_filter_enabled;
            float hampel_threshold;
            bool savgol_filter_enabled;
            int savgol_window_size;
            bool butterworth_enabled;
            bool wavelet_enabled;
            int wavelet_level;
            float wavelet_threshold;
            bool cusum_enabled;
            float cusum_threshold;
            float cusum_drift;
            bool smart_publishing_enabled;
            bool adaptive_normalizer_enabled;
            float adaptive_normalizer_alpha;
            uint32_t adaptive_normalizer_reset_timeout_sec;
        } runtime_config_t;
        
        runtime_config_t *cfg = (runtime_config_t*)config;
        
        cfg->butterworth_enabled = g_calib.recommended_butterworth;
        cfg->wavelet_enabled = g_calib.recommended_wavelet;
        cfg->wavelet_level = g_calib.recommended_wavelet_level;
        cfg->wavelet_threshold = g_calib.recommended_wavelet_threshold;
        cfg->hampel_filter_enabled = g_calib.recommended_hampel;
        cfg->hampel_threshold = g_calib.recommended_hampel_threshold;
        cfg->savgol_filter_enabled = g_calib.recommended_savgol;
        cfg->adaptive_normalizer_enabled = g_calib.recommended_adaptive_normalizer;
        cfg->adaptive_normalizer_alpha = g_calib.recommended_normalizer_alpha;
        
        ESP_LOGI(TAG, "✅ Optimal filter configuration applied (including wavelet if needed)");
    }
    
    // Automatically dump captured CSI data for debugging
    if (g_baseline_capture_count > 0 || g_movement_capture_count > 0) {
        calibration_dump_captured_data();
    }
}

// Initialize calibration system
void calibration_init(void) {
    memset(&g_calib, 0, sizeof(g_calib));
    g_calib.mode = CALIB_MODE_NORMAL;
    g_calib.phase = CALIB_IDLE;
    g_calib.num_selected = 0;
    
    ESP_LOGI(TAG, "Calibration system initialized");
}

// Start calibration process
bool calibration_start(int samples_or_duration, void *config, void *normalizer) {
    if (g_calib.phase != CALIB_IDLE) {
        ESP_LOGW(TAG, "Calibration already in progress");
        return false;
    }
    
    if (samples_or_duration < 100 || samples_or_duration > 10000) {
        ESP_LOGE(TAG, "Invalid sample count: %d (must be 100-10000 samples)", samples_or_duration);
        return false;
    }
    
    // Read traffic rate for estimated duration calculation
    typedef struct {
        bool csi_logs_enabled;
        bool verbose_mode;
        uint8_t debounce_count;
        float hysteresis_ratio;
        int persistence_timeout;
        float variance_scale;
        bool hampel_filter_enabled;
        float hampel_threshold;
        bool savgol_filter_enabled;
        int savgol_window_size;
        bool butterworth_enabled;
        bool cusum_enabled;
        float cusum_threshold;
        float cusum_drift;
        float micro_movement_threshold;
        float intense_movement_threshold;
        bool smart_publishing_enabled;
        bool adaptive_normalizer_enabled;
        float adaptive_normalizer_alpha;
        uint32_t adaptive_normalizer_reset_timeout_sec;
        uint32_t traffic_generator_rate;
    } runtime_config_t;
    
    runtime_config_t *cfg = (runtime_config_t*)config;
    uint32_t rate = cfg ? cfg->traffic_generator_rate : 15;  // Default if not configured
    
    uint32_t target_samples = (uint32_t)samples_or_duration;
    int estimated_duration = (rate > 0) ? (target_samples / rate) : 0;
    
    ESP_LOGI(TAG, "🎯 Calibration configuration:");
    ESP_LOGI(TAG, "   Target samples: %u per phase", target_samples);
    ESP_LOGI(TAG, "   Traffic rate: %u pps", rate);
    ESP_LOGI(TAG, "   Estimated duration: ~%d seconds per phase", estimated_duration);
    
    // Reset adaptive normalizer to prevent contamination of calibration data
    if (normalizer && config) {
        typedef struct {
            bool csi_logs_enabled;
            bool verbose_mode;
            uint8_t debounce_count;
            float hysteresis_ratio;
            int persistence_timeout;
            float variance_scale;
            bool hampel_filter_enabled;
            float hampel_threshold;
            bool savgol_filter_enabled;
            int savgol_window_size;
            bool butterworth_enabled;
            bool cusum_enabled;
            float cusum_threshold;
            float cusum_drift;
            bool smart_publishing_enabled;
            bool adaptive_normalizer_enabled;
            float adaptive_normalizer_alpha;
            uint32_t adaptive_normalizer_reset_timeout_sec;
        } runtime_config_t;
        
        runtime_config_t *cfg = (runtime_config_t*)config;
        
        // Reset normalizer with current alpha before disabling
        typedef struct {
            float running_mean;
            float running_variance;
            size_t sample_count;
            float alpha;
            bool initialized;
        } adaptive_normalizer_t;
        
        adaptive_normalizer_t *norm = (adaptive_normalizer_t*)normalizer;
        norm->running_mean = 0.0f;
        norm->running_variance = 1.0f;
        norm->sample_count = 0;
        norm->alpha = cfg->adaptive_normalizer_alpha;
        norm->initialized = false;
        
        ESP_LOGI(TAG, "🔄 Adaptive normalizer reset for calibration");
    }
    
    // Save current filter configuration (if config provided)
    if (config) {
        // Cast to runtime_config_t (we use void* to avoid circular dependency)
        typedef struct {
            bool csi_logs_enabled;
            bool verbose_mode;
            uint8_t debounce_count;
            float hysteresis_ratio;
            int persistence_timeout;
            float variance_scale;
            bool hampel_filter_enabled;
            float hampel_threshold;
            bool savgol_filter_enabled;
            int savgol_window_size;
            bool butterworth_enabled;
            bool cusum_enabled;
            float cusum_threshold;
            float cusum_drift;
            bool smart_publishing_enabled;
            bool adaptive_normalizer_enabled;
            float adaptive_normalizer_alpha;
            uint32_t adaptive_normalizer_reset_timeout_sec;
        } runtime_config_t;
        
        runtime_config_t *cfg = (runtime_config_t*)config;
        
        // Save current filter settings
        g_calib.saved_butterworth_enabled = cfg->butterworth_enabled;
        g_calib.saved_hampel_enabled = cfg->hampel_filter_enabled;
        g_calib.saved_hampel_threshold = cfg->hampel_threshold;
        g_calib.saved_savgol_enabled = cfg->savgol_filter_enabled;
        g_calib.saved_adaptive_normalizer_enabled = cfg->adaptive_normalizer_enabled;
        g_calib.saved_adaptive_normalizer_alpha = cfg->adaptive_normalizer_alpha;
        
        // Disable all filters for calibration
        cfg->butterworth_enabled = false;
        cfg->hampel_filter_enabled = false;
        cfg->savgol_filter_enabled = false;
        cfg->adaptive_normalizer_enabled = false;
        
        ESP_LOGI(TAG, "🔧 All filters disabled for calibration");
    }
    
    // Save config pointer for later use
    g_config_ptr = config;
    
    // Reset state
    memset(&g_calib.baseline_stats, 0, sizeof(g_calib.baseline_stats));
    memset(&g_calib.movement_stats, 0, sizeof(g_calib.movement_stats));
    memset(&g_calib.selected_features, 0, sizeof(g_calib.selected_features));
    memset(&g_calib.optimized_weights, 0, sizeof(g_calib.optimized_weights));
    g_calib.num_selected = 0;
    
    // Reset CSI capture counters
    g_baseline_capture_count = 0;
    g_movement_capture_count = 0;
    g_baseline_packet_counter = 0;
    g_movement_packet_counter = 0;
    g_has_last_packet = false;
    
    // Calculate capture interval for distributed sampling
    // Capture 50 packets evenly distributed across the phase
    g_baseline_capture_interval = target_samples / MAX_CAPTURE_PACKETS;
    g_movement_capture_interval = target_samples / MAX_CAPTURE_PACKETS;
    if (g_baseline_capture_interval < 1) g_baseline_capture_interval = 1;
    if (g_movement_capture_interval < 1) g_movement_capture_interval = 1;
    
    ESP_LOGI(TAG, "📸 Will capture 1 packet every %u samples (total: %u packets per phase)",
             g_baseline_capture_interval, MAX_CAPTURE_PACKETS);
    
    // Save calibration parameters
    g_calib.traffic_rate = rate;
    g_calib.baseline_movement_target_samples = target_samples;
    
    // Start baseline phase
    g_calib.mode = CALIB_MODE_COLLECTING;
    g_calib.phase = CALIB_BASELINE;
    g_calib.phase_target_samples = target_samples;
    g_calib.samples_in_current_phase = 0;
    
    ESP_LOGI(TAG, "🎯 Calibration started");
    ESP_LOGI(TAG, "🎯 Phase 1: BASELINE (target: %u samples)", g_calib.phase_target_samples);
    ESP_LOGI(TAG, "📋 Please ensure the room is EMPTY and STATIC");
    
    return true;
}

// Stop calibration process
void calibration_stop(void *config) {
    if (g_calib.phase == CALIB_IDLE) {
        return;
    }
    
    // Restore original filter configuration if interrupted (if config provided)
    if (config) {
        typedef struct {
            bool csi_logs_enabled;
            bool verbose_mode;
            uint8_t debounce_count;
            float hysteresis_ratio;
            int persistence_timeout;
            float variance_scale;
            bool hampel_filter_enabled;
            float hampel_threshold;
            bool savgol_filter_enabled;
            int savgol_window_size;
            bool butterworth_enabled;
            bool cusum_enabled;
            float cusum_threshold;
            float cusum_drift;
            float micro_movement_threshold;
            float intense_movement_threshold;
            bool smart_publishing_enabled;
            bool adaptive_normalizer_enabled;
            float adaptive_normalizer_alpha;
            uint32_t adaptive_normalizer_reset_timeout_sec;
        } runtime_config_t;
        
        runtime_config_t *cfg = (runtime_config_t*)config;
        
        cfg->butterworth_enabled = g_calib.saved_butterworth_enabled;
        cfg->hampel_filter_enabled = g_calib.saved_hampel_enabled;
        cfg->hampel_threshold = g_calib.saved_hampel_threshold;
        cfg->savgol_filter_enabled = g_calib.saved_savgol_enabled;
        cfg->adaptive_normalizer_enabled = g_calib.saved_adaptive_normalizer_enabled;
        cfg->adaptive_normalizer_alpha = g_calib.saved_adaptive_normalizer_alpha;
        
        ESP_LOGI(TAG, "🔧 Original filter configuration restored");
    }
    
    ESP_LOGI(TAG, "Calibration stopped");
    g_calib.mode = CALIB_MODE_NORMAL;
    g_calib.phase = CALIB_IDLE;
}

// Update calibration with new feature values
void calibration_update(const feature_array_t *features) {
    if (g_calib.phase != CALIB_BASELINE && g_calib.phase != CALIB_MOVEMENT) {
        return;
    }
    
    // Increment sample counter
    g_calib.samples_in_current_phase++;
    
    // Update statistics for current phase
    feature_stats_t *stats = (g_calib.phase == CALIB_BASELINE) ?
                             g_calib.baseline_stats :
                             g_calib.movement_stats;
    
    // Update statistics for all features using Welford's algorithm
    for (uint8_t i = 0; i < NUM_TOTAL_FEATURES; i++) {
        welford_update(&stats[i], features->features[i]);
    }
}

// Check if calibration phase has completed and advance to next phase
void calibration_check_completion(void) {
    if (g_calib.phase == CALIB_IDLE || g_calib.phase == CALIB_ANALYZING) {
        return;
    }
    
    // Check if we've collected enough samples for current phase
    if (g_calib.samples_in_current_phase >= g_calib.phase_target_samples) {
        
        if (g_calib.phase == CALIB_BASELINE) {
            // Check if we have enough samples (should always be true now)
            if (g_calib.baseline_stats[0].count < CALIBRATION_MIN_SAMPLES) {
                ESP_LOGW(TAG, "Not enough baseline samples (%u < %d), continuing...",
                         (unsigned int)g_calib.baseline_stats[0].count, CALIBRATION_MIN_SAMPLES);
                return;
            }
            
            // Move to movement phase
            g_calib.phase = CALIB_MOVEMENT;
            g_calib.phase_target_samples = g_calib.baseline_movement_target_samples;
            g_calib.samples_in_current_phase = 0;
            
            ESP_LOGI(TAG, "✅ Baseline phase complete (%u samples collected)",
                     (unsigned int)g_calib.baseline_stats[0].count);
            ESP_LOGI(TAG, "🎯 Phase 2: MOVEMENT (target: %u samples)",
                     g_calib.phase_target_samples);
            ESP_LOGI(TAG, "📋 Please perform NORMAL MOVEMENT in the room");
            
        } else if (g_calib.phase == CALIB_MOVEMENT) {
            // Check if we have enough samples (should always be true now)
            if (g_calib.movement_stats[0].count < CALIBRATION_MIN_SAMPLES) {
                ESP_LOGW(TAG, "Not enough movement samples (%u < %d), continuing...",
                         (unsigned int)g_calib.movement_stats[0].count, CALIBRATION_MIN_SAMPLES);
                return;
            }
            
            // Move to analysis phase
            g_calib.phase = CALIB_ANALYZING;
            
            ESP_LOGI(TAG, "✅ Movement phase complete (%u samples collected)",
                     (unsigned int)g_calib.movement_stats[0].count);
            
            // Perform analysis and apply optimal filter configuration
            analyze_and_select_features(g_config_ptr);
            
            // Set mode to normal but KEEP phase as ANALYZING
            // The phase will be reset to IDLE by calibration_acknowledge_completion()
            g_calib.mode = CALIB_MODE_NORMAL;
            
            ESP_LOGI(TAG, "⏳ Waiting for calibration results to be saved...");
        }
    }
}

// Get current calibration status
calibration_phase_t calibration_get_phase(void) {
    return g_calib.phase;
}

calibration_mode_t calibration_get_mode(void) {
    return g_calib.mode;
}

bool calibration_is_active(void) {
    return g_calib.phase != CALIB_IDLE;
}

// Get calibration results
uint8_t calibration_get_num_selected(void) {
    return g_calib.num_selected;
}

const uint8_t* calibration_get_selected_features(void) {
    return g_calib.selected_features;
}

const float* calibration_get_weights(void) {
    return g_calib.optimized_weights;
}

float calibration_get_threshold(void) {
    return g_calib.optimal_threshold;
}

// Get calibration progress (0.0 to 1.0)
float calibration_get_progress(void) {
    if (g_calib.phase == CALIB_IDLE) {
        return 0.0f;
    }
    
    if (g_calib.phase == CALIB_ANALYZING) {
        return 1.0f;
    }
    
    // Calculate progress based on samples collected
    if (g_calib.phase_target_samples == 0) {
        return 0.0f;
    }
    
    float phase_progress = (float)g_calib.samples_in_current_phase / g_calib.phase_target_samples;
    
    if (g_calib.phase == CALIB_BASELINE) {
        // First half of total progress
        return phase_progress * 0.5f;
    } else if (g_calib.phase == CALIB_MOVEMENT) {
        // Second half of total progress
        return 0.5f + phase_progress * 0.5f;
    }
    
    return 0.0f;
}

// Get number of samples collected in current phase
uint32_t calibration_get_samples_collected(void) {
    return g_calib.samples_in_current_phase;
}

// Get calibration statistics for reporting
void calibration_get_stats(char *buffer, size_t buffer_size) {
    if (!buffer || buffer_size == 0) {
        return;
    }
    
    int offset = 0;
    
    const char *phase_names[] = {"IDLE", "BASELINE", "MOVEMENT", "ANALYZING"};
    
    // Check buffer space before each write
    if (offset >= (int)buffer_size - 1) return;
    int written = snprintf(buffer + offset, buffer_size - offset,
                          "Phase: %s\n", phase_names[g_calib.phase]);
    if (written < 0 || written >= (int)(buffer_size - offset)) {
        buffer[buffer_size - 1] = '\0';
        return;
    }
    offset += written;
    
    if (g_calib.phase == CALIB_BASELINE || g_calib.phase == CALIB_MOVEMENT) {
        uint32_t samples_collected = g_calib.samples_in_current_phase;
        uint32_t samples_target = g_calib.phase_target_samples;
        uint32_t samples_remaining = (samples_target > samples_collected) ? 
                                     (samples_target - samples_collected) : 0;
        
        if (offset >= (int)buffer_size - 1) return;
        written = snprintf(buffer + offset, buffer_size - offset,
                          "Samples: %lu / %lu (remaining: %lu)\n", 
                          (unsigned long)samples_collected, 
                          (unsigned long)samples_target, 
                          (unsigned long)samples_remaining);
        if (written < 0 || written >= (int)(buffer_size - offset)) {
            buffer[buffer_size - 1] = '\0';
            return;
        }
        offset += written;
        
        // Estimate time remaining based on traffic rate
        if (g_calib.traffic_rate > 0 && samples_remaining > 0) {
            uint32_t estimated_sec = samples_remaining / g_calib.traffic_rate;
            if (offset >= (int)buffer_size - 1) return;
            written = snprintf(buffer + offset, buffer_size - offset,
                              "Estimated time remaining: ~%lu seconds\n", 
                              (unsigned long)estimated_sec);
            if (written < 0 || written >= (int)(buffer_size - offset)) {
                buffer[buffer_size - 1] = '\0';
                return;
            }
            offset += written;
        }
    }
    
    if (g_calib.num_selected > 0) {
        if (offset >= (int)buffer_size - 1) return;
        written = snprintf(buffer + offset, buffer_size - offset,
                          "\nSelected features (%d):\n", g_calib.num_selected);
        if (written < 0 || written >= (int)(buffer_size - offset)) {
            buffer[buffer_size - 1] = '\0';
            return;
        }
        offset += written;
        
        for (uint8_t i = 0; i < g_calib.num_selected; i++) {
            // Check if we have enough space for at least 50 more characters
            if (offset >= (int)buffer_size - 50) {
                // Truncate gracefully
                if (offset < (int)buffer_size - 4) {
                    snprintf(buffer + offset, buffer_size - offset, "...\n");
                }
                return;
            }
            
            uint8_t feat_idx = g_calib.selected_features[i];
            written = snprintf(buffer + offset, buffer_size - offset,
                              "  %s (weight: %.3f)\n",
                              feature_names[feat_idx],
                              g_calib.optimized_weights[i]);
            if (written < 0 || written >= (int)(buffer_size - offset)) {
                buffer[buffer_size - 1] = '\0';
                return;
            }
            offset += written;
        }
    }
}

// Get current calibration results (for saving to NVS)
void calibration_get_results(calibration_state_t *state) {
    if (!state) {
        return;
    }
    
    // Copy the entire calibration state including feature ranges
    memcpy(state, &g_calib, sizeof(calibration_state_t));
}

// Acknowledge calibration completion (called after saving to NVS)
void calibration_acknowledge_completion(void) {
    if (g_calib.phase == CALIB_ANALYZING) {
        g_calib.phase = CALIB_IDLE;
        ESP_LOGI(TAG, "✅ Calibration acknowledged and finalized");
    }
}

// Apply saved calibration results (from NVS)
void calibration_apply_saved(const calibration_state_t *state) {
    if (!state) {
        ESP_LOGE(TAG, "Cannot apply NULL calibration state");
        return;
    }
    
    if (state->num_selected == 0 || state->num_selected > MAX_SELECTED_FEATURES) {
        ESP_LOGE(TAG, "Invalid num_selected: %d", state->num_selected);
        return;
    }
    
    // Validate optimal_threshold is in reasonable range
    if (isnan(state->optimal_threshold) || isinf(state->optimal_threshold) ||
        state->optimal_threshold < 0.05f || state->optimal_threshold > 0.95f) {
        ESP_LOGE(TAG, "Invalid optimal_threshold: %.4f (must be between 0.05 and 0.95)", 
                 state->optimal_threshold);
        return;
    }
    
    // Validate selected_features for uniqueness
    for (uint8_t i = 0; i < state->num_selected; i++) {
        // Check if feature index is within valid range
        if (state->selected_features[i] >= NUM_TOTAL_FEATURES) {
            ESP_LOGE(TAG, "Invalid feature index[%d]: %d (max: %d)", 
                     i, state->selected_features[i], NUM_TOTAL_FEATURES - 1);
            return;
        }
        
        // Check for duplicate feature indices
        for (uint8_t j = i + 1; j < state->num_selected; j++) {
            if (state->selected_features[i] == state->selected_features[j]) {
                ESP_LOGE(TAG, "Duplicate feature index: %d at positions %d and %d", 
                         state->selected_features[i], i, j);
                return;
            }
        }
    }
    
    // Validate weights sum to approximately 1.0
    float weight_sum = 0.0f;
    for (uint8_t i = 0; i < state->num_selected; i++) {
        float weight = state->optimized_weights[i];
        
        if (isnan(weight) || isinf(weight)) {
            ESP_LOGE(TAG, "Invalid weight[%d]: %.4f (NaN or Infinity)", i, weight);
            return;
        }
        
        if (weight < 0.0f) {
            ESP_LOGE(TAG, "Invalid weight[%d]: %.4f (negative)", i, weight);
            return;
        }
        
        if (weight > 1.0f) {
            ESP_LOGE(TAG, "Invalid weight[%d]: %.4f (exceeds 1.0)", i, weight);
            return;
        }
        
        weight_sum += weight;
    }
    
    if (weight_sum < 0.8f || weight_sum > 1.2f) {
        ESP_LOGE(TAG, "Invalid weight sum: %.4f (must be between 0.8 and 1.2)", weight_sum);
        return;
    }
    
    // Copy calibration results to global state including feature ranges
    g_calib.num_selected = state->num_selected;
    g_calib.optimal_threshold = state->optimal_threshold;
    
    memcpy(g_calib.selected_features, state->selected_features, 
           sizeof(g_calib.selected_features));
    memcpy(g_calib.optimized_weights, state->optimized_weights, 
           sizeof(g_calib.optimized_weights));
    memcpy(g_calib.feature_min, state->feature_min,
           sizeof(g_calib.feature_min));
    memcpy(g_calib.feature_max, state->feature_max,
           sizeof(g_calib.feature_max));
    
    ESP_LOGI(TAG, "✅ Calibration applied with feature normalization ranges");
}

// Get recommended filter configuration
void calibration_get_filter_config(bool *butterworth, bool *wavelet, int *wavelet_level,
                                   float *wavelet_threshold, bool *hampel, float *hampel_threshold,
                                   bool *savgol, bool *adaptive_norm, float *norm_alpha) {
    if (butterworth) *butterworth = g_calib.recommended_butterworth;
    if (wavelet) *wavelet = g_calib.recommended_wavelet;
    if (wavelet_level) *wavelet_level = g_calib.recommended_wavelet_level;
    if (wavelet_threshold) *wavelet_threshold = g_calib.recommended_wavelet_threshold;
    if (hampel) *hampel = g_calib.recommended_hampel;
    if (hampel_threshold) *hampel_threshold = g_calib.recommended_hampel_threshold;
    if (savgol) *savgol = g_calib.recommended_savgol;
    if (adaptive_norm) *adaptive_norm = g_calib.recommended_adaptive_normalizer;
    if (norm_alpha) *norm_alpha = g_calib.recommended_normalizer_alpha;
}

// Get feature normalization ranges
const float* calibration_get_feature_min(void) {
    return g_calib.feature_min;
}

const float* calibration_get_feature_max(void) {
    return g_calib.feature_max;
}

// Test helper: Force phase change (for unit testing only)
void calibration_force_phase(calibration_phase_t phase) {
    g_calib.phase = phase;
    g_calib.samples_in_current_phase = 0;
}

// Test helper: Trigger analysis manually (for unit testing only)
void calibration_trigger_analysis(void) {
    analyze_and_select_features(g_config_ptr);
}

// CSI data capture function
void calibration_capture_csi(const int8_t *csi_data, size_t csi_len) {
    if (!csi_data || csi_len == 0 || csi_len > CSI_PACKET_SIZE) {
        return;
    }
    
    // Store last packet for potential use
    memcpy(g_last_csi_packet, csi_data, csi_len);
    if (csi_len < CSI_PACKET_SIZE) {
        memset(g_last_csi_packet + csi_len, 0, CSI_PACKET_SIZE - csi_len);
    }
    g_has_last_packet = true;
    
    // Capture packets with distributed sampling during baseline and movement phases
    if (g_calib.phase == CALIB_BASELINE) {
        g_baseline_packet_counter++;
        // Capture every Nth packet to get distributed samples across entire phase
        if (g_baseline_capture_count < MAX_CAPTURE_PACKETS && 
            (g_baseline_packet_counter % g_baseline_capture_interval == 0)) {
            memcpy(g_captured_baseline[g_baseline_capture_count], csi_data, csi_len);
            if (csi_len < CSI_PACKET_SIZE) {
                memset(g_captured_baseline[g_baseline_capture_count] + csi_len, 0, CSI_PACKET_SIZE - csi_len);
            }
            g_baseline_capture_count++;
        }
    } else if (g_calib.phase == CALIB_MOVEMENT) {
        g_movement_packet_counter++;
        // Capture every Nth packet to get distributed samples across entire phase
        if (g_movement_capture_count < MAX_CAPTURE_PACKETS && 
            (g_movement_packet_counter % g_movement_capture_interval == 0)) {
            memcpy(g_captured_movement[g_movement_capture_count], csi_data, csi_len);
            if (csi_len < CSI_PACKET_SIZE) {
                memset(g_captured_movement[g_movement_capture_count] + csi_len, 0, CSI_PACKET_SIZE - csi_len);
            }
            g_movement_capture_count++;
        }
    }
}

// Dump captured CSI data as C arrays (for copying to test files)
// Compact format: one packet per line to save space
__attribute__((unused)) static void calibration_dump_captured_data(void) {
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "========== CAPTURED CSI DATA ==========");
    ESP_LOGI(TAG, "Baseline packets: %zu, Movement packets: %zu", 
             g_baseline_capture_count, g_movement_capture_count);
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "Copy the following to test_app/main/real_csi_data.h:");
    ESP_LOGI(TAG, "");
    
    // Print baseline packets (one per line)
    ESP_LOGI(TAG, "// Baseline CSI packets (static environment) - %zu packets", g_baseline_capture_count);
    for (size_t p = 0; p < g_baseline_capture_count; p++) {
        printf("static const int8_t real_baseline_%u[128] = {", (unsigned int)p);
        for (int i = 0; i < CSI_PACKET_SIZE; i++) {
            printf("%d", g_captured_baseline[p][i]);
            if (i < CSI_PACKET_SIZE - 1) printf(", ");
        }
        printf("};\n");
    }
    printf("\n");
    
    // Print movement packets (one per line)
    ESP_LOGI(TAG, "// Movement CSI packets (with human movement) - %zu packets", g_movement_capture_count);
    for (size_t p = 0; p < g_movement_capture_count; p++) {
        printf("static const int8_t real_movement_%u[128] = {", (unsigned int)p);
        for (int i = 0; i < CSI_PACKET_SIZE; i++) {
            printf("%d", g_captured_movement[p][i]);
            if (i < CSI_PACKET_SIZE - 1) printf(", ");
        }
        printf("};\n");
    }
    
    ESP_LOGI(TAG, "");
    ESP_LOGI(TAG, "========================================");
    ESP_LOGI(TAG, "");
}
