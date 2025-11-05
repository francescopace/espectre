/*
 * ESPectre - Configuration Manager Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "config_manager.h"
#include "nvs_storage.h"
#include "esp_log.h"
#include <string.h>

static const char *TAG = "Config_Manager";

// Default configuration values
// Optimized for noisy/challenging environments based on real-world testing
#define DEFAULT_DEBOUNCE_COUNT 10          // Increased from 3 to reduce false positives
#define DEFAULT_HYSTERESIS_RATIO 0.7f
#define DEFAULT_PERSISTENCE_TIMEOUT 3
#define DEFAULT_VARIANCE_SCALE 600.0f     // Increased from 400 for high-variance environments
#define DEFAULT_HAMPEL_THRESHOLD 3.0f     // Increased from 2.0 for better outlier tolerance
#define DEFAULT_SAVGOL_WINDOW 5

void config_init_defaults(runtime_config_t *config) {
    if (!config) {
        ESP_LOGE(TAG, "config_init_defaults: NULL pointer");
        return;
    }
    
    memset(config, 0, sizeof(runtime_config_t));
    
    config->csi_logs_enabled = true;
    config->verbose_mode = false;
    config->debounce_count = DEFAULT_DEBOUNCE_COUNT;
    config->hysteresis_ratio = DEFAULT_HYSTERESIS_RATIO;
    config->persistence_timeout = DEFAULT_PERSISTENCE_TIMEOUT;
    config->variance_scale = DEFAULT_VARIANCE_SCALE;
    
    // Default feature weights based on real-world testing in noisy environments
    // Using 5 features that proved most robust and discriminant
    // Weights sum to 1.0 for proper scoring (0-1 range)
    // Indices: 0=variance, 1=skewness, 2=kurtosis, 3=entropy, 4=iqr,
    //          5=spatial_variance, 6=spatial_correlation, 7=spatial_gradient
    config->feature_weights[0] = 0.15f;  // variance
    config->feature_weights[1] = 0.0f;   // skewness (not used by default)
    config->feature_weights[2] = 0.0f;   // kurtosis (not used by default)
    config->feature_weights[3] = 0.25f;  // entropy (often best discriminant)
    config->feature_weights[4] = 0.20f;  // iqr (robust to outliers)
    config->feature_weights[5] = 0.0f;   // spatial_variance (not used by default)
    config->feature_weights[6] = 0.20f;  // spatial_correlation (very discriminant)
    config->feature_weights[7] = 0.20f;  // spatial_gradient (very robust)
    // Total: 100% (0.15 + 0.25 + 0.20 + 0.20 + 0.20 = 1.0)
    
    
    // Enable adaptive normalizer by default for better adaptation to environment changes
    config->adaptive_normalizer_enabled = true;
    config->adaptive_normalizer_alpha = 0.01f;
    config->adaptive_normalizer_reset_timeout_sec = 60;  // Increased from 30 for stability
    
    // Traffic generator (20 pps = good balance for continuous CSI and stability)
    config->traffic_generator_rate = 20;
    
    // Enable key filters by default for robust operation in noisy environments
    config->hampel_filter_enabled = true;   // Outlier removal
    config->hampel_threshold = DEFAULT_HAMPEL_THRESHOLD;
    config->savgol_filter_enabled = true;   // Smoothing
    config->savgol_window_size = DEFAULT_SAVGOL_WINDOW;
    config->butterworth_enabled = true;     // High-frequency noise reduction
    
    // Wavelet filter settings (disabled by default, enable for high-noise environments)
    config->wavelet_enabled = false;        // Enable manually if variance > 500
    config->wavelet_level = 3;              // Maximum denoising when enabled
    config->wavelet_threshold = 1.0f;       // Balanced threshold
    
    config->cusum_enabled = false;
    config->cusum_threshold = 0.5f;
    config->cusum_drift = 0.01f;
    
    config->smart_publishing_enabled = false;
}

esp_err_t config_validate(const runtime_config_t *config) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (config->debounce_count < 1 || config->debounce_count > 10) {
        ESP_LOGE(TAG, "Invalid debounce_count: %d", config->debounce_count);
        return ESP_ERR_INVALID_ARG;
    }
    
    if (config->hysteresis_ratio < 0.1f || config->hysteresis_ratio > 1.0f) {
        ESP_LOGE(TAG, "Invalid hysteresis_ratio: %.2f", config->hysteresis_ratio);
        return ESP_ERR_INVALID_ARG;
    }
    
    if (config->persistence_timeout < 1 || config->persistence_timeout > 30) {
        ESP_LOGE(TAG, "Invalid persistence_timeout: %d", config->persistence_timeout);
        return ESP_ERR_INVALID_ARG;
    }
    
    return ESP_OK;
}

esp_err_t config_load_from_nvs(runtime_config_t *config, const nvs_config_data_t *nvs_cfg) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // If nvs_cfg is provided, use it directly (avoid duplicate NVS read)
    nvs_config_data_t local_nvs_cfg;
    const nvs_config_data_t *cfg_to_use;
    
    if (nvs_cfg) {
        cfg_to_use = nvs_cfg;
    } else {
        // Load from NVS if not provided
        if (!nvs_has_control_params()) {
            return ESP_ERR_NOT_FOUND;
        }
        
        esp_err_t err = nvs_load_control_params(&local_nvs_cfg);
        if (err != ESP_OK) {
            return err;
        }
        cfg_to_use = &local_nvs_cfg;
    }
    
    // Convert NVS data to runtime config
    config->debounce_count = cfg_to_use->debounce_count;
    config->hysteresis_ratio = cfg_to_use->hysteresis_ratio;
    config->persistence_timeout = cfg_to_use->persistence_timeout;
    config->variance_scale = cfg_to_use->variance_scale;
    
    // Load feature weights array from NVS
    memcpy(config->feature_weights, cfg_to_use->feature_weights, sizeof(config->feature_weights));
    
    config->hampel_filter_enabled = cfg_to_use->hampel_filter_enabled;
    config->hampel_threshold = cfg_to_use->hampel_threshold;
    config->savgol_filter_enabled = cfg_to_use->savgol_filter_enabled;
    config->savgol_window_size = cfg_to_use->savgol_window_size;
    config->butterworth_enabled = cfg_to_use->butterworth_enabled;
    
    // Load wavelet settings from NVS
    config->wavelet_enabled = cfg_to_use->wavelet_enabled;
    config->wavelet_level = cfg_to_use->wavelet_level;
    config->wavelet_threshold = cfg_to_use->wavelet_threshold;
    
    config->csi_logs_enabled = cfg_to_use->csi_logs_enabled;
    config->adaptive_normalizer_enabled = cfg_to_use->adaptive_normalizer_enabled;
    config->adaptive_normalizer_alpha = cfg_to_use->adaptive_normalizer_alpha;
    config->adaptive_normalizer_reset_timeout_sec = cfg_to_use->adaptive_normalizer_reset_timeout_sec;
    config->traffic_generator_rate = cfg_to_use->traffic_generator_rate;
    
    return ESP_OK;
}

esp_err_t config_save_to_nvs(const runtime_config_t *config, 
                             float threshold_high, 
                             float threshold_low) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    nvs_config_data_t nvs_cfg;
    nvs_cfg.version = NVS_CONFIG_VERSION;
    nvs_cfg.threshold_high = threshold_high;
    nvs_cfg.threshold_low = threshold_low;
    nvs_cfg.debounce_count = config->debounce_count;
    nvs_cfg.hysteresis_ratio = config->hysteresis_ratio;
    nvs_cfg.persistence_timeout = config->persistence_timeout;
    nvs_cfg.variance_scale = config->variance_scale;
    
    // Save feature weights array to NVS
    memcpy(nvs_cfg.feature_weights, config->feature_weights, sizeof(config->feature_weights));
    
    nvs_cfg.hampel_filter_enabled = config->hampel_filter_enabled;
    nvs_cfg.hampel_threshold = config->hampel_threshold;
    nvs_cfg.savgol_filter_enabled = config->savgol_filter_enabled;
    nvs_cfg.savgol_window_size = config->savgol_window_size;
    nvs_cfg.butterworth_enabled = config->butterworth_enabled;
    
    // Save wavelet settings to NVS
    nvs_cfg.wavelet_enabled = config->wavelet_enabled;
    nvs_cfg.wavelet_level = config->wavelet_level;
    nvs_cfg.wavelet_threshold = config->wavelet_threshold;
    
    nvs_cfg.csi_logs_enabled = config->csi_logs_enabled;
    nvs_cfg.adaptive_normalizer_enabled = config->adaptive_normalizer_enabled;
    nvs_cfg.adaptive_normalizer_alpha = config->adaptive_normalizer_alpha;
    nvs_cfg.adaptive_normalizer_reset_timeout_sec = config->adaptive_normalizer_reset_timeout_sec;
    nvs_cfg.traffic_generator_rate = config->traffic_generator_rate;
    
    return nvs_save_control_params(&nvs_cfg);
}

bool config_exists_in_nvs(void) {
    return nvs_has_control_params();
}
