/*
 * ESPectre - Configuration Manager Module
 * 
 * Centralized configuration management:
 * - Runtime configuration structure
 * - NVS persistence integration
 * - Configuration validation
 * - Default values management
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef CONFIG_MANAGER_H
#define CONFIG_MANAGER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"
#include "nvs_storage.h"

// Default threshold for motion detection (optimized for amplitude skewness)
#define DEFAULT_THRESHOLD 0.50f

// Subcarrier selection limits
#define MAX_SUBCARRIERS 64  // Maximum number of subcarriers that can be selected

// Runtime configuration structure
typedef struct {
    // Logging (always enabled in monitor mode, disabled otherwise)
    bool verbose_mode;
    
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
    
    // CUSUM (unused but kept for compatibility)
    bool cusum_enabled;
    float cusum_threshold;
    float cusum_drift;
    
    // Smart publishing
    bool smart_publishing_enabled;
    
    // Traffic generator (for continuous CSI packets)
    uint32_t traffic_generator_rate;  // packets/sec (0=disabled, 1-50, recommended: 15)
    
    // Segmentation parameters (configurable at runtime)
    float segmentation_k_factor;        // Threshold sensitivity (0.5-5.0)
    uint16_t segmentation_window_size;  // Moving variance window (3-50 packets)
    uint16_t segmentation_min_length;   // Minimum segment length (5-100 packets)
    uint16_t segmentation_max_length;   // Maximum segment length (10-200 packets, 0=no limit)
    
    // Subcarrier selection (configurable at runtime)
    uint8_t selected_subcarriers[MAX_SUBCARRIERS];  // Array of selected subcarrier indices (0-63)
    uint8_t num_selected_subcarriers;               // Number of selected subcarriers (1-64)
} runtime_config_t;

/**
 * Initialize configuration with defaults
 * 
 * @param config Configuration structure to initialize
 */
void config_init_defaults(runtime_config_t *config);

/**
 * Validate configuration parameters
 * 
 * @param config Configuration to validate
 * @return ESP_OK if valid, error code otherwise
 */
esp_err_t config_validate(const runtime_config_t *config);

/**
 * Load configuration from NVS
 * 
 * @param config Output configuration structure
 * @param nvs_cfg Pre-loaded NVS configuration data (optional, can be NULL)
 * @return ESP_OK on success, ESP_ERR_NVS_NOT_FOUND if no config exists
 */
esp_err_t config_load_from_nvs(runtime_config_t *config, const nvs_config_data_t *nvs_cfg);

/**
 * Save configuration to NVS
 * 
 * @param config Configuration to save
 * @param segmentation_threshold Current segmentation threshold
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t config_save_to_nvs(const runtime_config_t *config, 
                             float segmentation_threshold);

/**
 * Check if configuration exists in NVS
 * 
 * @return true if configuration exists, false otherwise
 */
bool config_exists_in_nvs(void);

#endif // CONFIG_MANAGER_H
