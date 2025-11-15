/*
 * ESPectre - NVS Storage Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "nvs_storage.h"
#include <string.h>
#include <math.h>
#include "nvs_flash.h"
#include "nvs.h"
#include "esp_log.h"
#include "esp_err.h"

// Total number of features (for validation)
#define NUM_TOTAL_FEATURES 10

// Weight validation bounds
#define WEIGHT_SUM_MIN      0.8f    // Minimum acceptable weight sum
#define WEIGHT_SUM_MAX      1.2f    // Maximum acceptable weight sum

static const char *TAG = "NVS_Storage";

// NVS keys for calibration data
#define NVS_KEY_CAL_VERSION      "cal_ver"
#define NVS_KEY_CAL_NUM_SEL      "cal_nsel"
#define NVS_KEY_CAL_FEATURES     "cal_feat"
#define NVS_KEY_CAL_WEIGHTS      "cal_wght"
#define NVS_KEY_CAL_THRESHOLD    "cal_thrs"

// NVS keys for config data
#define NVS_KEY_CFG_VERSION      "cfg_ver"
#define NVS_KEY_CFG_DATA         "cfg_data"

// Helper function to validate float values
static bool is_valid_float(float value) {
    return !isnan(value) && !isinf(value);
}

// Helper function to validate calibration data
static bool validate_calibration_data(const nvs_calibration_data_t *calib) {
    if (!calib) return false;
    
    if (calib->version != NVS_CALIBRATION_VERSION) {
        ESP_LOGW(TAG, "Calibration version mismatch: %d != %d", 
                 calib->version, NVS_CALIBRATION_VERSION);
        return false;
    }
    
    if (calib->num_selected == 0 || calib->num_selected > MAX_SELECTED_FEATURES) {
        ESP_LOGW(TAG, "Invalid num_selected: %d", calib->num_selected);
        return false;
    }
    
    if (!is_valid_float(calib->optimal_threshold) || 
        calib->optimal_threshold <= 0.0f || 
        calib->optimal_threshold >= 1.0f) {
        ESP_LOGW(TAG, "Invalid threshold: %.4f", calib->optimal_threshold);
        return false;
    }
    
    // Validate selected_features array
    for (uint8_t i = 0; i < calib->num_selected; i++) {
        // Check if feature index is within valid range
        if (calib->selected_features[i] >= NUM_TOTAL_FEATURES) {
            ESP_LOGW(TAG, "Invalid feature index[%d]: %d (max: %d)", 
                     i, calib->selected_features[i], NUM_TOTAL_FEATURES - 1);
            return false;
        }
        
        // Check for duplicate feature indices
        for (uint8_t j = i + 1; j < calib->num_selected; j++) {
            if (calib->selected_features[i] == calib->selected_features[j]) {
                ESP_LOGW(TAG, "Duplicate feature index: %d at positions %d and %d", 
                         calib->selected_features[i], i, j);
                return false;
            }
        }
    }
    
    // Validate weights sum to approximately 1.0
    float weight_sum = 0.0f;
    for (uint8_t i = 0; i < calib->num_selected; i++) {
        float weight = calib->optimized_weights[i];
        
        // Check if weight is a valid float
        if (!is_valid_float(weight)) {
            ESP_LOGW(TAG, "Invalid weight[%d]: %.4f (NaN or Infinity)", i, weight);
            return false;
        }
        
        // Check if weight is non-negative
        if (weight < 0.0f) {
            ESP_LOGW(TAG, "Invalid weight[%d]: %.4f (negative)", i, weight);
            return false;
        }
        
        // Check if weight is not unreasonably large
        if (weight > 1.0f) {
            ESP_LOGW(TAG, "Invalid weight[%d]: %.4f (exceeds 1.0)", i, weight);
            return false;
        }
        
        weight_sum += weight;
    }
    
    if (weight_sum < WEIGHT_SUM_MIN || weight_sum > WEIGHT_SUM_MAX) {
        ESP_LOGW(TAG, "Invalid weight sum: %.4f (must be between %.2f and %.2f)", 
                 weight_sum, WEIGHT_SUM_MIN, WEIGHT_SUM_MAX);
        return false;
    }
    
    return true;
}

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
    if (!is_valid_float(config->hampel_threshold) || 
        config->hampel_threshold < 1.0f || 
        config->hampel_threshold > 10.0f) {
        ESP_LOGW(TAG, "Invalid hampel_threshold: %.1f, using default", config->hampel_threshold);
        config->hampel_threshold = 2.0f;
        valid = false;
    }
    
    // Validate savgol window size (must be odd)
    if (config->savgol_window_size < 3 || config->savgol_window_size > 11 || 
        (config->savgol_window_size % 2) == 0) {
        ESP_LOGW(TAG, "Invalid savgol_window_size: %d, using default", config->savgol_window_size);
        config->savgol_window_size = 5;
        valid = false;
    }
    
    // Validate wavelet level
    if (config->wavelet_level < 1 || config->wavelet_level > 3) {
        ESP_LOGW(TAG, "Invalid wavelet_level: %d, using default", config->wavelet_level);
        config->wavelet_level = 3;
        valid = false;
    }
    
    // Validate wavelet threshold
    if (!is_valid_float(config->wavelet_threshold) || 
        config->wavelet_threshold < 0.5f || 
        config->wavelet_threshold > 2.0f) {
        ESP_LOGW(TAG, "Invalid wavelet_threshold: %.2f, using default", config->wavelet_threshold);
        config->wavelet_threshold = 1.0f;
        valid = false;
    }
    
    // Validate segmentation threshold
    if (!is_valid_float(config->segmentation_threshold) || 
        config->segmentation_threshold < 0.5f || 
        config->segmentation_threshold > 10.0f) {
        ESP_LOGW(TAG, "Invalid segmentation_threshold: %.2f, using default", config->segmentation_threshold);
        config->segmentation_threshold = 2.2f;
        valid = false;
    }
    
    return valid;
}

esp_err_t nvs_storage_init(void) {
    ESP_LOGI(TAG, "Initializing NVS storage system");
    return ESP_OK;
}

esp_err_t nvs_save_calibration(const nvs_calibration_data_t *calib) {
    if (!calib) {
        return ESP_ERR_INVALID_ARG;
    }
    
    if (!validate_calibration_data(calib)) {
        ESP_LOGE(TAG, "Invalid calibration data, not saving");
        return ESP_ERR_INVALID_ARG;
    }
    
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CALIBRATION, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to open NVS namespace: %s", esp_err_to_name(err));
        return err;
    }
    
    // Save version
    err = nvs_set_u8(handle, NVS_KEY_CAL_VERSION, calib->version);
    if (err != ESP_OK) goto cleanup;
    
    // Save num_selected
    err = nvs_set_u8(handle, NVS_KEY_CAL_NUM_SEL, calib->num_selected);
    if (err != ESP_OK) goto cleanup;
    
    // Save selected features array
    err = nvs_set_blob(handle, NVS_KEY_CAL_FEATURES, 
                       calib->selected_features, 
                       sizeof(calib->selected_features));
    if (err != ESP_OK) goto cleanup;
    
    // Save weights array
    err = nvs_set_blob(handle, NVS_KEY_CAL_WEIGHTS, 
                       calib->optimized_weights, 
                       sizeof(calib->optimized_weights));
    if (err != ESP_OK) goto cleanup;
    
    // Save threshold
    err = nvs_set_blob(handle, NVS_KEY_CAL_THRESHOLD, 
                       &calib->optimal_threshold, 
                       sizeof(calib->optimal_threshold));
    if (err != ESP_OK) goto cleanup;
    
    // Commit changes
    err = nvs_commit(handle);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "💾 Calibration saved to NVS (%d features, threshold: %.4f)",
                 calib->num_selected, calib->optimal_threshold);
    }
    
cleanup:
    nvs_close(handle);
    return err;
}

esp_err_t nvs_load_calibration(nvs_calibration_data_t *calib) {
    if (!calib) {
        return ESP_ERR_INVALID_ARG;
    }
    
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CALIBRATION, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        if (err == ESP_ERR_NVS_NOT_FOUND) {
            ESP_LOGD(TAG, "No calibration data found in NVS");
        } else {
            ESP_LOGE(TAG, "Failed to open NVS namespace: %s", esp_err_to_name(err));
        }
        return err;
    }
    
    // Load version
    err = nvs_get_u8(handle, NVS_KEY_CAL_VERSION, &calib->version);
    if (err != ESP_OK) goto cleanup;
    
    // Load num_selected
    err = nvs_get_u8(handle, NVS_KEY_CAL_NUM_SEL, &calib->num_selected);
    if (err != ESP_OK) goto cleanup;
    
    // Load selected features
    size_t size = sizeof(calib->selected_features);
    err = nvs_get_blob(handle, NVS_KEY_CAL_FEATURES, calib->selected_features, &size);
    if (err != ESP_OK) goto cleanup;
    
    // Load weights
    size = sizeof(calib->optimized_weights);
    err = nvs_get_blob(handle, NVS_KEY_CAL_WEIGHTS, calib->optimized_weights, &size);
    if (err != ESP_OK) goto cleanup;
    
    // Load threshold
    size = sizeof(calib->optimal_threshold);
    err = nvs_get_blob(handle, NVS_KEY_CAL_THRESHOLD, &calib->optimal_threshold, &size);
    if (err != ESP_OK) goto cleanup;
    
    // Validate loaded data
    if (!validate_calibration_data(calib)) {
        ESP_LOGE(TAG, "Loaded calibration data is invalid");
        err = ESP_ERR_INVALID_STATE;
        goto cleanup;
    }
    
    ESP_LOGI(TAG, "✅ Loaded calibration from NVS (%d features, threshold: %.4f)",
             calib->num_selected, calib->optimal_threshold);
    
cleanup:
    nvs_close(handle);
    return err;
}

bool nvs_has_calibration(void) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CALIBRATION, NVS_READONLY, &handle);
    if (err != ESP_OK) {
        return false;
    }
    
    uint8_t version;
    err = nvs_get_u8(handle, NVS_KEY_CAL_VERSION, &version);
    nvs_close(handle);
    
    return (err == ESP_OK);
}

esp_err_t nvs_clear_calibration(void) {
    nvs_handle_t handle;
    esp_err_t err = nvs_open(NVS_NAMESPACE_CALIBRATION, NVS_READWRITE, &handle);
    if (err != ESP_OK) {
        return err;
    }
    
    err = nvs_erase_all(handle);
    if (err == ESP_OK) {
        err = nvs_commit(handle);
        if (err == ESP_OK) {
            ESP_LOGI(TAG, "🗑️  Calibration data cleared from NVS");
        }
    }
    
    nvs_close(handle);
    return err;
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
    
    esp_err_t err1 = nvs_clear_calibration();
    esp_err_t err2 = nvs_clear_control_params();
    
    if (err1 == ESP_OK && err2 == ESP_OK) {
        ESP_LOGI(TAG, "✅ Factory reset complete - all settings cleared");
        return ESP_OK;
    }
    
    // Return first error encountered
    return (err1 != ESP_OK) ? err1 : err2;
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
    
    // Check if our namespaces exist
    ESP_LOGI(TAG, "ESPectre Storage:");
    ESP_LOGI(TAG, "  Calibration data: %s", nvs_has_calibration() ? "YES" : "NO");
    ESP_LOGI(TAG, "  Control params:   %s", nvs_has_control_params() ? "YES" : "NO");
}
