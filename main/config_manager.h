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

// Runtime configuration structure
typedef struct {
    // Logging
    bool csi_logs_enabled;
    bool verbose_mode;
    
    // Detection parameters
    uint8_t debounce_count;
    float hysteresis_ratio;
    int persistence_timeout;
    float variance_scale;
    
    // Feature weights array (for both default and calibrated detection)
    // Indices: 0=variance, 1=skewness, 2=kurtosis, 3=entropy, 4=iqr,
    //          5=spatial_variance, 6=spatial_correlation, 7=spatial_gradient
    float feature_weights[10];
    
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
    
    // Adaptive Normalizer settings
    bool adaptive_normalizer_enabled;
    float adaptive_normalizer_alpha;
    uint32_t adaptive_normalizer_reset_timeout_sec;
    
    // Traffic generator (for continuous CSI packets)
    uint32_t traffic_generator_rate;  // packets/sec (0=disabled, 1-50, recommended: 15)
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
 * @param threshold_high Current high threshold
 * @param threshold_low Current low threshold
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t config_save_to_nvs(const runtime_config_t *config, 
                             float threshold_high, 
                             float threshold_low);

/**
 * Check if configuration exists in NVS
 * 
 * @return true if configuration exists, false otherwise
 */
bool config_exists_in_nvs(void);

#endif // CONFIG_MANAGER_H
