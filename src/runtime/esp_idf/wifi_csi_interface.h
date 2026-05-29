/*
 * ESPectre - WiFi CSI Interface
 * 
 * Abstract interface for WiFi CSI operations.
 * Allows dependency injection for testing without real WiFi hardware.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_err.h"
#include "esp_wifi.h"

namespace esphome {
namespace espectre {

/**
 * WiFi CSI Interface
 * 
 * Abstract interface for WiFi CSI hardware operations.
 * Implementations:
 * - WiFiCSIReal: Uses real ESP-IDF WiFi functions
 * - WiFiCSIMock: Returns ESP_OK for testing
 */
class IWiFiCSI {
 public:
  virtual ~IWiFiCSI() = default;
  
  /**
   * Set CSI configuration
   * 
   * @param config CSI configuration structure
   * @return ESP_OK on success
   */
  virtual esp_err_t set_csi_config(const wifi_csi_config_t* config) = 0;
  
  /**
   * Set CSI receive callback
   * 
   * @param cb Callback function
   * @param ctx Context pointer passed to callback
   * @return ESP_OK on success
   */
  virtual esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void* ctx) = 0;
  
  /**
   * Enable or disable CSI
   * 
   * @param enable true to enable, false to disable
   * @return ESP_OK on success
   */
  virtual esp_err_t set_csi(bool enable) = 0;
};

/**
 * Real WiFi CSI Implementation
 * 
 * Uses actual ESP-IDF WiFi functions.
 * Used in production code.
 */
class WiFiCSIReal : public IWiFiCSI {
 public:
  esp_err_t set_csi_config(const wifi_csi_config_t* config) override {
    return esp_wifi_set_csi_config(config);
  }
  
  esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void* ctx) override {
    return esp_wifi_set_csi_rx_cb(cb, ctx);
  }
  
  esp_err_t set_csi(bool enable) override {
    return esp_wifi_set_csi(enable);
  }
};

}  // namespace espectre
}  // namespace esphome

