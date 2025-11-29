#ifndef ESP_SPIFFS_H
#define ESP_SPIFFS_H

#include "esp_err.h"
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Mock SPIFFS configuration structure
 */
typedef struct {
    const char* base_path;
    const char* partition_label;
    size_t max_files;
    bool format_if_mount_failed;
} esp_vfs_spiffs_conf_t;

/**
 * Mock SPIFFS register - always succeeds in test environment
 */
static inline esp_err_t esp_vfs_spiffs_register(const esp_vfs_spiffs_conf_t* conf) {
    (void)conf;
    return ESP_OK;
}

/**
 * Mock SPIFFS unregister
 */
static inline esp_err_t esp_vfs_spiffs_unregister(const char* partition_label) {
    (void)partition_label;
    return ESP_OK;
}

/**
 * Mock SPIFFS info - returns fake values
 */
static inline esp_err_t esp_spiffs_info(const char* partition_label, size_t* total_bytes, size_t* used_bytes) {
    (void)partition_label;
    if (total_bytes) *total_bytes = 458752;  // ~448KB
    if (used_bytes) *used_bytes = 0;
    return ESP_OK;
}

/**
 * Mock SPIFFS check
 */
static inline esp_err_t esp_spiffs_check(const char* partition_label) {
    (void)partition_label;
    return ESP_OK;
}

/**
 * Mock SPIFFS format
 */
static inline esp_err_t esp_spiffs_format(const char* partition_label) {
    (void)partition_label;
    return ESP_OK;
}

/**
 * Mock SPIFFS mounted check
 */
static inline bool esp_spiffs_mounted(const char* partition_label) {
    (void)partition_label;
    return true;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_SPIFFS_H

