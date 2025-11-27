/*
 * ESPectre - Configuration Manager Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "config_manager.h"
#include "nvs_storage.h"
#include "filters.h"
#include "esp_log.h"
#include <string.h>

static const char *TAG = "Config_Manager";

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
    nvs_cfg->segmentation_window_size = config->segmentation_window_size;
    
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
    config->segmentation_window_size = nvs_cfg->segmentation_window_size;
    
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
    
    // System settings
    config->verbose_mode = DEFAULT_VERBOSE_MODE;
    config->features_enabled = DEFAULT_FEATURES_ENABLED;
    config->smart_publishing_enabled = DEFAULT_SMART_PUBLISHING;
    
    // Traffic generator
    config->traffic_generator_rate = DEFAULT_TRAFFIC_GENERATOR_RATE;
    
    // Filter enable flags (all from espectre.h)
    config->hampel_filter_enabled = DEFAULT_HAMPEL_ENABLED;
    config->savgol_filter_enabled = DEFAULT_SAVGOL_ENABLED;
    config->butterworth_enabled = DEFAULT_BUTTERWORTH_ENABLED;
    config->wavelet_enabled = DEFAULT_WAVELET_ENABLED;
    config->cusum_enabled = DEFAULT_CUSUM_ENABLED;
    
    // Filter parameters
    config->hampel_threshold = HAMPEL_DEFAULT_THRESHOLD;
    config->savgol_window_size = SAVGOL_WINDOW_SIZE;
    config->wavelet_level = WAVELET_LEVEL_MAX;
    config->wavelet_threshold = WAVELET_DEFAULT_THRESHOLD;
    config->cusum_threshold = CUSUM_DEFAULT_THRESHOLD;
    config->cusum_drift = CUSUM_DEFAULT_DRIFT;
    
    // Segmentation parameters
    config->segmentation_window_size = SEGMENTATION_DEFAULT_WINDOW_SIZE;
    
    // Default subcarrier selection (top most informative subcarriers)
    // Calculate count from array size - single source of truth
    const uint8_t default_subcarriers[] = DEFAULT_SUBCARRIERS;
    config->num_selected_subcarriers = sizeof(default_subcarriers) / sizeof(default_subcarriers[0]);
    
    memcpy(config->selected_subcarriers, default_subcarriers, sizeof(default_subcarriers));
}

esp_err_t config_validate(const runtime_config_t *config) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    // Validate segmentation parameters
    if (config->segmentation_window_size < SEGMENTATION_WINDOW_SIZE_MIN ||
        config->segmentation_window_size > SEGMENTATION_WINDOW_SIZE_MAX) {
        ESP_LOGE(TAG, "Invalid window size: %d", config->segmentation_window_size);
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
