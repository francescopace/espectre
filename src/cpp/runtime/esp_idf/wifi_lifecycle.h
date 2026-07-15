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

#include <atomic>
#include "esp_event.h"
#include "esp_err.h"
#include "esp_netif.h"
#include <functional>

#include "pending_event.h"

namespace espectre {

// Callback types
using wifi_connected_callback_t = std::function<void(const esp_netif_ip_info_t &)>;
using wifi_disconnected_callback_t = std::function<void()>;

/**
 * WiFi Lifecycle Manager
 *
 * Manages WiFi connection events and coordinates service lifecycle.
 * Handles startup sequence: CSI → Traffic Generator → Band Calibration
 *
 * The STA-start handler applies the short radio policy synchronously, before
 * association. Connect/disconnect callbacks run from process_pending_events(),
 * which the runtime must call from its loop task. This keeps service startup
 * and log formatting off the small default event loop task (sys_evt) stack.
 */
class WiFiLifecycleManager {
 public:
  /**
   * Register WiFi event handlers
   * 
   * @param connected_cb Callback when WiFi obtains an IPv4 configuration;
   *        receives the address, netmask, and gateway from GOT_IP
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
  esp_err_t process_pending_events();

 private:
  esp_err_t init();
  static esp_err_t apply_csi_wifi_policy();
  static esp_err_t apply_started_csi_policy();
  static void log_csi_runtime_state(const char *tag);

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
  esp_event_handler_instance_t started_instance_{nullptr};

  // Events recorded on the event loop task, drained by the runtime loop.
  PendingEvent<esp_netif_ip_info_t> connected_event_;
  PendingEvent<> disconnected_event_;
  std::atomic<esp_err_t> started_policy_err_{ESP_ERR_INVALID_STATE};
  bool ready_{false};
};

}  // namespace espectre
