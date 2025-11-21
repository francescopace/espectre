/*
 * ESPectre - Configuration Manager Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "config_manager.h"
#include "nvs_storage.h"
#include "segmentation.h"
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
    
    // Copy subcarrier selection
    memcpy(nvs_cfg->selected_subcarriers, config->selected_subcarriers, 
           config->num_selected_subcarriers * sizeof(uint8_t));
    nvs_cfg->num_selected_subcarriers = config->num_selected_subcarriers;
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
    
    // Copy subcarrier selection
    memcpy(config->selected_subcarriers, nvs_cfg->selected_subcarriers, 
           nvs_cfg->num_selected_subcarriers * sizeof(uint8_t));
    config->num_selected_subcarriers = nvs_cfg->num_selected_subcarriers;
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
    // Hampel disabled by default (testing shows 0% outlier rate in typical environments)
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
    
    // Segmentation parameters (platform-specific defaults)
    config->segmentation_k_factor = SEGMENTATION_DEFAULT_K_FACTOR;
    config->segmentation_window_size = SEGMENTATION_DEFAULT_WINDOW_SIZE;
    config->segmentation_min_length = SEGMENTATION_DEFAULT_MIN_LENGTH;
    config->segmentation_max_length = SEGMENTATION_DEFAULT_MAX_LENGTH;
    
    // ESP32-S3: Top 12 most informative subcarriers
    const uint8_t default_subcarriers[] = {47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58};
    config->num_selected_subcarriers = 12;
    
    memcpy(config->selected_subcarriers, default_subcarriers, 
           config->num_selected_subcarriers * sizeof(uint8_t));
}

esp_err_t config_validate(const runtime_config_t *config) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Validate segmentation parameters
    if (config->segmentation_k_factor < SEGMENTATION_K_FACTOR_MIN || 
        config->segmentation_k_factor > SEGMENTATION_K_FACTOR_MAX) {
        ESP_LOGE(TAG, "Invalid K factor: %.2f", config->segmentation_k_factor);
        return ESP_ERR_INVALID_ARG;
    }
    
    if (config->segmentation_window_size < SEGMENTATION_WINDOW_SIZE_MIN || 
        config->segmentation_window_size > SEGMENTATION_MAX_WINDOW_SIZE) {
        ESP_LOGE(TAG, "Invalid window size: %d", config->segmentation_window_size);
        return ESP_ERR_INVALID_ARG;
    }
    
    if (config->segmentation_min_length < SEGMENTATION_MIN_LENGTH_MIN || 
        config->segmentation_min_length > SEGMENTATION_MIN_LENGTH_MAX) {
        ESP_LOGE(TAG, "Invalid min length: %d", config->segmentation_min_length);
        return ESP_ERR_INVALID_ARG;
    }
    
    if (config->segmentation_max_length != 0 && 
        (config->segmentation_max_length < SEGMENTATION_MAX_LENGTH_MIN || 
         config->segmentation_max_length > SEGMENTATION_MAX_LENGTH_MAX)) {
        ESP_LOGE(TAG, "Invalid max length: %d", config->segmentation_max_length);
        return ESP_ERR_INVALID_ARG;
    }
    
    // Validate subcarrier selection
    if (config->num_selected_subcarriers == 0 || 
        config->num_selected_subcarriers > MAX_SUBCARRIERS) {
        ESP_LOGE(TAG, "Invalid number of subcarriers: %d", config->num_selected_subcarriers);
        return ESP_ERR_INVALID_ARG;
    }
    
    // Validate each subcarrier index (0-63 for 64 total subcarriers)
    for (uint8_t i = 0; i < config->num_selected_subcarriers; i++) {
        if (config->selected_subcarriers[i] >= 64) {
            ESP_LOGE(TAG, "Invalid subcarrier index: %d", config->selected_subcarriers[i]);
            return ESP_ERR_INVALID_ARG;
        }
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
