/*
 * ESPectre - NVS Storage Module
 * 
 * Handles persistent storage of runtime configuration
 * in ESP32's Non-Volatile Storage (NVS).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef NVS_STORAGE_H
#define NVS_STORAGE_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

// NVS Namespaces
#define NVS_NAMESPACE_CONFIG        "espectre_cfg"

// NVS keys for config data
#define NVS_KEY_CFG_DATA            "cfg_data"

// Versioning for future compatibility
#define NVS_CONFIG_VERSION 10  // Incremented: removed min_length parameter

// Control parameters structure for NVS storage
typedef struct {
    uint8_t version;
    
    // Feature extraction control
    bool features_enabled;  // Enable/disable feature extraction during MOTION state
    
    // Filter settings
    bool hampel_filter_enabled;
    float hampel_threshold;
    bool savgol_filter_enabled;
    int savgol_window_size;
    bool butterworth_enabled;
    
    // Wavelet filter settings
    bool wavelet_enabled;
    int wavelet_level;           // 1-3
    float wavelet_threshold;     // 0.5-2.0
    
    // Traffic generator
    uint32_t traffic_generator_rate;  // packets/sec (0=disabled)
    
    // Segmentation parameters
    float segmentation_threshold;       // Adaptive threshold for MVS (0.5-10.0)
    uint16_t segmentation_window_size;  // Moving variance window (10-200 packets)
    
    // Subcarrier selection
    uint8_t selected_subcarriers[64];  // Array of selected subcarrier indices (0-63)
    uint8_t num_selected_subcarriers;  // Number of selected subcarriers (1-64)
    
} nvs_config_data_t;

/**
 * Initialize NVS storage system
 * Must be called after nvs_flash_init()
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t nvs_storage_init(void);

/**
 * Save control parameters to NVS
 * 
 * @param config Control parameters to save
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t nvs_save_control_params(const nvs_config_data_t *config);

/**
 * Load control parameters from NVS
 * 
 * @param config Pointer to store loaded control parameters
 * @return ESP_OK on success, ESP_ERR_NVS_NOT_FOUND if no data exists
 */
esp_err_t nvs_load_control_params(nvs_config_data_t *config);

/**
 * Check if control parameters exist in NVS
 * 
 * @return true if control parameters exist, false otherwise
 */
bool nvs_has_control_params(void);

/**
 * Clear control parameters from NVS
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t nvs_clear_control_params(void);

/**
 * Factory reset - clear all stored data and restore defaults
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t nvs_factory_reset(void);

/**
 * Print NVS storage statistics
 */
void nvs_print_stats(void);

#endif // NVS_STORAGE_H
