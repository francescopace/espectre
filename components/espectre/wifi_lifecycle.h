/*
 * ESPectre - WiFi Lifecycle Manager
 * 
 * Manages WiFi connection lifecycle and coordinates service startup/shutdown.
 * Handles CSI, Traffic Generator, and Band Calibration orchestration.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_event.h"
#include "esp_err.h"
#include <functional>

namespace esphome {
namespace espectre {

// Callback types
using wifi_connected_callback_t = std::function<void()>;
using wifi_disconnected_callback_t = std::function<void()>;

/**
 * WiFi Lifecycle Manager
 * 
 * Manages WiFi connection events and coordinates service lifecycle.
 * Handles startup sequence: CSI → Traffic Generator → Band Calibration
 */
class WiFiLifecycleManager {
 public:
  /**
   * Initialize WiFi for optimal CSI capture
   * 
   * Configures WiFi settings critical for CSI:
   * - Promiscuous mode
   * - Power save disabled
   * - Protocol (b/g/n or b/g/n/ax for ESP32-C6)
   * - Bandwidth HT20
   * 
   * @return ESP_OK on success
   */
  esp_err_t init();
  
  /**
   * Register WiFi event handlers
   * 
   * @param connected_cb Callback when WiFi connects
   * @param disconnected_cb Callback when WiFi disconnects
   * @return ESP_OK on success
   */
  esp_err_t register_handlers(wifi_connected_callback_t connected_cb,
                              wifi_disconnected_callback_t disconnected_cb);
  
  /**
   * Unregister WiFi event handlers
   */
  void unregister_handlers();
  
 private:
  // Static handlers for ESP-IDF C API (separated by event type)
  static void ip_event_handler_(void* arg, esp_event_base_t event_base,
                                int32_t event_id, void* event_data);
  static void wifi_event_handler_(void* arg, esp_event_base_t event_base,
                                  int32_t event_id, void* event_data);
  
  // Callbacks
  wifi_connected_callback_t connected_callback_;
  wifi_disconnected_callback_t disconnected_callback_;
  
  // Event handler instances
  esp_event_handler_instance_t connected_instance_{nullptr};
  esp_event_handler_instance_t disconnected_instance_{nullptr};
};

}  // namespace espectre
}  // namespace esphome
