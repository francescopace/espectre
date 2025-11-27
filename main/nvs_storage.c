/*
 * ESPectre - NVS Storage Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "nvs_storage.h"
#include "filters.h"
#include "espectre.h"
#include "validation.h"
#include <string.h>
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"
#include "esp_err.h"

static const char *TAG = "NVS_Storage";

// Helper function to validate config data
static bool validate_config_data(nvs_config_data_t *config) {
    if (!config) return false;
    
    if (config->version != NVS_CONFIG_VERSION) {
        ESP_LOGW(TAG, "Config version mismatch: %d != %d", 
                 config->version, NVS_CONFIG_VERSION);
        return false;
    }
    
    bool valid = true;
    
    // Validate hampel threshold
    if (!validate_hampel_threshold(config->hampel_threshold)) {
        ESP_LOGW(TAG, "Invalid hampel_threshold: %.1f, using default", config->hampel_threshold);
        config->hampel_threshold = HAMPEL_DEFAULT_THRESHOLD;
        valid = false;
    }
    
    // Validate savgol window size
    if (!validate_savgol_window_size(config->savgol_window_size)) {
        ESP_LOGW(TAG, "Invalid savgol_window_size: %d, using default", config->savgol_window_size);
        config->savgol_window_size = SAVGOL_WINDOW_SIZE;
        valid = false;
    }
    
    // Validate wavelet level
    if (!validate_wavelet_level(config->wavelet_level)) {
        ESP_LOGW(TAG, "Invalid wavelet_level: %d, using default", config->wavelet_level);
        config->wavelet_level = WAVELET_LEVEL_MAX;
        valid = false;
    }
    
    // Validate wavelet threshold
    if (!validate_wavelet_threshold(config->wavelet_threshold)) {
        ESP_LOGW(TAG, "Invalid wavelet_threshold: %.2f, using default", config->wavelet_threshold);
        config->wavelet_threshold = 1.0f;
        valid = false;
    }
    
    // Validate segmentation threshold
    if (!validate_segmentation_threshold(config->segmentation_threshold)) {
        ESP_LOGW(TAG, "Invalid segmentation_threshold: %.2f, using default", config->segmentation_threshold);
        config->segmentation_threshold = SEGMENTATION_DEFAULT_THRESHOLD;
        valid = false;
    }
    
    // Validate segmentation window size
    if (!validate_segmentation_window_size(config->segmentation_window_size)) {
        ESP_LOGW(TAG, "Invalid segmentation_window_size: %d, using default", config->segmentation_window_size);
        config->segmentation_window_size = SEGMENTATION_DEFAULT_WINDOW_SIZE;
        valid = false;
    }
    
    return valid;
}

esp_err_t nvs_storage_init(void) {
    ESP_LOGI(TAG, "Initializing NVS storage system");
    return ESP_OK;
}

esp_err_t nvs_save_control_params(const nvs_config_data_t *config) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CONFIG, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS namespace: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save entire config structure as blob
    err = nvs_set_blob(handle, NVS_KEY_CFG_DATA, config, sizeof(nvs_config_data_t));
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to save config data: %s", esp_err_to_name(err));
        goto cleanup;
    }
    
    // Commit changes
    err = nvs_commit(handle);
    if (err == ESP_OK) {
        ESP_LOGD(TAG, "💾 Control parameters saved to NVS");
    }
    
cleanup:
    nvs_close(handle);
    return err;
}

esp_err_t nvs_load_control_params(nvs_config_data_t *config) {
    if (!config) {
        return ESP_ERR_INVALID_ARG;
    }
    
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CONFIG, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        if (err == ESP_ERR_NVS_NOT_FOUND) {
            ESP_LOGD(TAG, "No control parameters found in NVS");
        } else {
            ESP_LOGE(TAG, "Failed to open NVS namespace: %s", esp_err_to_name(err));
        }
        return err;
    }
    
    // Load config structure
    size_t size = sizeof(nvs_config_data_t);
    err = nvs_get_blob(handle, NVS_KEY_CFG_DATA, config, &size);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to load config data: %s", esp_err_to_name(err));
        goto cleanup;
    }
    
    // Validate and fix loaded data
    if (!validate_config_data(config)) {
        ESP_LOGW(TAG, "Some config parameters were invalid and have been corrected");
    }
    
    ESP_LOGI(TAG, "✅ Loaded control parameters from NVS");
    
cleanup:
    nvs_close(handle);
    return err;
}

bool nvs_has_control_params(void) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CONFIG, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        return false;
    }
    
    size_t size = 0;
    err = nvs_get_blob(handle, NVS_KEY_CFG_DATA, NULL, &size);
    nvs_close(handle);
    
    return (err == ESP_OK && size == sizeof(nvs_config_data_t));
}

esp_err_t nvs_clear_control_params(void) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CONFIG, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }
    
    err = nvs_erase_all(handle);
    if (err == ESP_OK) {
        err = nvs_commit(handle);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "🗑️  Control parameters cleared from NVS");
        }
    }
    
    nvs_close(handle);
    return err;
}

esp_err_t nvs_factory_reset(void) {
    ESP_LOGW(TAG, "⚠️  Performing factory reset...");
    
    esp_err_t err = nvs_clear_control_params();
    
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Factory reset complete");
        return ESP_OK;
    }
    
    return err;
}

void nvs_print_stats(void) {
    nvs_stats_t nvs_stats;
    esp_err_t err = nvs_get_stats(NULL, &nvs_stats);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "NVS Statistics:");
        ESP_LOGI(TAG, "  Used entries:  %d", nvs_stats.used_entries);
        ESP_LOGI(TAG, "  Free entries:  %d", nvs_stats.free_entries);
        ESP_LOGI(TAG, "  Total entries: %d", nvs_stats.total_entries);
        ESP_LOGI(TAG, "  Namespaces:    %d", nvs_stats.namespace_count);
    } else {
        ESP_LOGE(TAG, "Failed to get NVS stats: %s", esp_err_to_name(err));
    }
    
    // Check if our namespace exists
    ESP_LOGI(TAG, "ESPectre Storage:");
    ESP_LOGI(TAG, "  Control params: %s", nvs_has_control_params() ? "YES" : "NO");
}
