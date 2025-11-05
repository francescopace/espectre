/*
 * ESPectre - WiFi Manager Module
 * 
 * Handles WiFi connection management:
 * - Station mode initialization
 * - Event handling (connect/disconnect/got IP)
 * - Automatic reconnection
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef WIFI_MANAGER_H
#define WIFI_MANAGER_H

#include <stdbool.h>
#include "esp_err.h"

// WiFi credentials configuration
typedef struct {
    const char *ssid;
    const char *password;
} wifi_credentials_t;

// WiFi manager state
typedef struct {
    bool connected;
    bool initialized;
} wifi_manager_state_t;

/**
 * Initialize WiFi manager
 * 
 * @param state WiFi manager state
 * @param config WiFi credentials (SSID and password)
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t wifi_manager_init(wifi_manager_state_t *state, const wifi_credentials_t *config);

/**
 * Wait for WiFi connection
 * Blocks until WiFi is connected
 * 
 * @return ESP_OK on success, error code otherwise
 */
esp_err_t wifi_manager_wait_connected(void);

/**
 * Check if WiFi is connected
 * 
 * @param state WiFi manager state
 * @return true if connected, false otherwise
 */
bool wifi_manager_is_connected(const wifi_manager_state_t *state);

/**
 * Cleanup WiFi manager
 * 
 * @param state WiFi manager state
 */
void wifi_manager_cleanup(wifi_manager_state_t *state);

#endif // WIFI_MANAGER_H
