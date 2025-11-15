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
#define DEFAULT_HAMPEL_THRESHOLD 3.0f     // Increased from 2.0 for better outlier tolerance
#define DEFAULT_SAVGOL_WINDOW 5

// Helper: Copy runtime config to NVS structure
static inline void config_to_nvs(nvs_config_data_t *nvs_cfg, const runtime_config_t *config,
                                 float segmentation_threshold) {
    nvs_cfg->version = NVS_CONFIG_VERSION;
    nvs_cfg->features_enabled = config->features_enabled;
    nvs_cfg->hampel_filter_enabled = config->hampel_filter_enabled;
    nvs_cfg->hampel_threshold = config->hampel_threshold;
    nvs_cfg->savgol_filter_enabled = config->savgol_filter_enabled;
    nvs_cfg->savgol_window_size = config->savgol_window_size;
    nvs_cfg->butterworth_enabled = config->butterworth_enabled;
    nvs_cfg->wavelet_enabled = config->wavelet_enabled;
    nvs_cfg->wavelet_level = config->wavelet_level;
    nvs_cfg->wavelet_threshold = config->wavelet_threshold;
    nvs_cfg->traffic_generator_rate = config->traffic_generator_rate;
    nvs_cfg->segmentation_threshold = segmentation_threshold;
}

// Helper: Copy NVS structure to runtime config
static inline void nvs_to_config(runtime_config_t *config, const nvs_config_data_t *nvs_cfg) {
    config->features_enabled = nvs_cfg->features_enabled;
    config->hampel_filter_enabled = nvs_cfg->hampel_filter_enabled;
    config->hampel_threshold = nvs_cfg->hampel_threshold;
    config->savgol_filter_enabled = nvs_cfg->savgol_filter_enabled;
    config->savgol_window_size = nvs_cfg->savgol_window_size;
    config->butterworth_enabled = nvs_cfg->butterworth_enabled;
    config->wavelet_enabled = nvs_cfg->wavelet_enabled;
    config->wavelet_level = nvs_cfg->wavelet_level;
    config->wavelet_threshold = nvs_cfg->wavelet_threshold;
    config->traffic_generator_rate = nvs_cfg->traffic_generator_rate;
}

void config_init_defaults(runtime_config_t *config) {
    if (!config) {
        ESP_LOGE(TAG, "config_init_defaults: NULL pointer");
        return;
    }
    
    memset(config, 0, sizeof(runtime_config_t));
    
    config->verbose_mode = false;
    
    // Feature extraction enabled by default
    config->features_enabled = true;
    
    // Traffic generator (20 pps = good balance for continuous CSI and stability)
    config->traffic_generator_rate = 20;
    
    // Enable key filters by default for robust operation in noisy environments
    // Hampel disabled by default (calibration shows 0% outlier rate in typical environments)
    config->hampel_filter_enabled = false;  // Enable manually if high outlier rate detected
    config->hampel_threshold = 2.0f;        // Optimized threshold (reduced from 3.0 for better sensitivity)
    config->savgol_filter_enabled = true;   // Smoothing (recommended for noisy signals)
    config->savgol_window_size = DEFAULT_SAVGOL_WINDOW;
    config->butterworth_enabled = true;     // High-frequency noise reduction (always recommended)
    
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
    
    // Minimal validation - most parameters are boolean or have reasonable defaults
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
    
    // Use helper to convert NVS to runtime config
    nvs_to_config(config, cfg_to_use);
    
    return ESP_OK;
}

esp_err_t config_save_to_nvs(const runtime_config_t *config, 
                             float segmentation_threshold) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    nvs_config_data_t nvs_cfg;
    
    // Use helper to convert runtime config to NVS
    config_to_nvs(&nvs_cfg, config, segmentation_threshold);
    
    return nvs_save_control_params(&nvs_cfg);
}

bool config_exists_in_nvs(void) {
    return nvs_has_control_params();
}
