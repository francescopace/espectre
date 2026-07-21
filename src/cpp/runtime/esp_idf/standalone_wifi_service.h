/*
 * ESPectre - Standalone Wi-Fi Service
 *
 * Starts and monitors the standalone station connection used by sensing
 * frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <functional>

#include "esp_err.h"
#include "esp_event.h"
#include "wifi_lifecycle.h"

namespace espectre {

using standalone_wifi_callback_t = std::function<void()>;

struct StandaloneWifiConfig {
  const char *ssid{""};
  const char *password{""};
  const char *bssid{""};
  uint8_t channel{0U};
  int max_retry{8};
  bool manage_csi_lifecycle{false};
};

struct StandaloneWifiInfo {
  bool connected{false};
  char ip_address[16]{};
  char mac_address[18]{};
  uint8_t channel{0U};
};

class StandaloneWifiService {
 public:
  esp_err_t setup(const StandaloneWifiConfig &config,
                  standalone_wifi_callback_t connected_cb = {},
                  standalone_wifi_callback_t disconnected_cb = {});
  esp_err_t start();
  esp_err_t update_station_config(const StandaloneWifiConfig &config);
  void loop();
  bool get_info(StandaloneWifiInfo *info) const;
  void shutdown();

 private:
  friend struct StandaloneWifiServiceTestAccess;
  static void wifi_event_handler_(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);

  esp_err_t configure_station_();
  void handle_wifi_started_();
  void handle_wifi_stopped_();
  void handle_wifi_disconnected_(void *event_data);
  void handle_lifecycle_connected_();
  void handle_lifecycle_disconnected_();
  void maybe_run_deferred_connect_fallback_();
  void clear_cached_ip_info_();

  StandaloneWifiConfig config_{};
  WiFiLifecycleManager wifi_lifecycle_;
  standalone_wifi_callback_t connected_cb_;
  standalone_wifi_callback_t disconnected_cb_;
  esp_event_handler_instance_t wifi_event_instance_{nullptr};
  esp_event_handler_instance_t ip_event_instance_{nullptr};
  bool setup_complete_{false};
  bool wifi_connect_requested_{false};
  bool defer_connect_once_after_start_{false};
  bool deferred_connect_fallback_pending_{false};
  bool wifi_started_{false};
  uint64_t deferred_connect_fallback_deadline_us_{0U};
  int wifi_retry_count_{0};
  esp_netif_ip_info_t cached_ip_info_{};
};

}  // namespace espectre
