/*
 * ESPectre - NVS Storage Module
 * 
 * Handles persistent storage of calibration results and runtime configuration
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
#define NVS_NAMESPACE_CALIBRATION "espectre_cal"
#define NVS_NAMESPACE_CONFIG      "espectre_cfg"

// Versioning for future compatibility
#define NVS_CALIBRATION_VERSION 1
#define NVS_CONFIG_VERSION 3

// Maximum sizes
#define MAX_SELECTED_FEATURES 6

// Calibration data structure for NVS storage
typedef struct {
    uint8_t version;
    uint8_t num_selected;
    uint8_t selected_features[MAX_SELECTED_FEATURES];
    float optimized_weights[MAX_SELECTED_FEATURES];
    float optimal_threshold;
    float feature_min[MAX_SELECTED_FEATURES];
    float feature_max[MAX_SELECTED_FEATURES];
} nvs_calibration_data_t;

// Control parameters structure for NVS storage
typedef struct {
    uint8_t version;
    
    // Detection parameters
    float threshold_high;
    float threshold_low;
    uint8_t debounce_count;
    float hysteresis_ratio;
    int persistence_timeout;
    float variance_scale;
    
    // Feature weights array
    // Indices: 0=variance, 1=skewness, 2=kurtosis, 3=entropy, 4=iqr,
    //          5=spatial_variance, 6=spatial_correlation, 7=spatial_gradient
    float feature_weights[8];
    
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
    
    // Logging
    bool csi_logs_enabled;
    
    // Adaptive Normalizer settings
    bool adaptive_normalizer_enabled;
    float adaptive_normalizer_alpha;
    uint32_t adaptive_normalizer_reset_timeout_sec;
    
    // Traffic generator
    uint32_t traffic_generator_rate;  // packets/sec (0=disabled)
    
} nvs_config_data_t;

/**
 * Initialize NVS storage system
 * Must be called after nvs_flash_init()
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t nvs_storage_init(void);

/**
 * Save calibration data to NVS
 * 
 * @param calib Calibration data to save
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t nvs_save_calibration(const nvs_calibration_data_t *calib);

/**
 * Load calibration data from NVS
 * 
 * @param calib Pointer to store loaded calibration data
 * @return ESP_OK on success, ESP_ERR_NVS_NOT_FOUND if no data exists
 */
esp_err_t nvs_load_calibration(nvs_calibration_data_t *calib);

/**
 * Check if calibration data exists in NVS
 * 
 * @return true if calibration data exists, false otherwise
 */
bool nvs_has_calibration(void);

/**
 * Clear calibration data from NVS
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t nvs_clear_calibration(void);

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
