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

#include "pending_event.h"

namespace espectre {

// Callback types
using wifi_connected_callback_t = std::function<void()>;
using wifi_disconnected_callback_t = std::function<void()>;

/**
 * WiFi Lifecycle Manager
 *
 * Manages WiFi connection events and coordinates service lifecycle.
 * Handles startup sequence: CSI → Traffic Generator → Band Calibration
 *
 * Event handlers only record connect/disconnect events; the registered
 * callbacks run from process_pending_events(), which the runtime must call
 * from its loop task. This keeps service startup and log formatting off the
 * small default event loop task (sys_evt) stack.
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
   * Re-apply the CSI Wi-Fi policy once the STA interface is started.
   *
   * Some ESP-IDF targets reject band/protocol changes before esp_wifi_start().
   * Calling this from WIFI_EVENT_STA_START ensures the first association uses
   * the HT20/11n policy instead of the target's HE/11ax default.
  */
  static esp_err_t apply_csi_wifi_policy();
  
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

  /**
   * Invoke the registered callbacks for events recorded by the handlers.
   *
   * Must be called periodically from the runtime loop task. A recorded
   * disconnect is processed before a recorded connect so a reconnect cycle
   * tears services down before starting them again.
   */
  void process_pending_events();

 private:
  void log_connect_diagnostics_();

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

  // Events recorded on the event loop task, drained by the runtime loop.
  PendingEvent<> connected_event_;
  PendingEvent<> disconnected_event_;
};

}  // namespace espectre
