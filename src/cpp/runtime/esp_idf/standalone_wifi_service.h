/*
 * ESPectre - Standalone Wi-Fi Service
 *
 * Starts and monitors the standalone station connection used by sensing
 * frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "esp_err.h"
#include "esp_event.h"
#include "pending_queue.h"
#include "wifi_lifecycle.h"

namespace espectre {

using standalone_wifi_callback_t = std::function<void()>;

struct StandaloneWifiAccessPoint {
  std::string ssid;
  std::string bssid;
  int8_t rssi_dbm{0};
  uint8_t channel{0U};
};

using standalone_wifi_scan_callback_t =
    std::function<void(esp_err_t, const std::vector<StandaloneWifiAccessPoint> &)>;

struct StandaloneWifiConfig {
  const char *ssid{""};
  const char *password{""};
  const char *bssid{""};
  uint8_t channel{0U};
  int max_retry{8};
  bool manage_csi_lifecycle{false};
  WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
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
  /** Scan every allowed channel for the configured SSID and report its bounded snapshot from loop(). */
  esp_err_t request_scan(standalone_wifi_scan_callback_t callback);
  void loop();
  bool get_info(StandaloneWifiInfo *info) const;
  void shutdown();

 private:
  enum class PendingWifiEventType : uint8_t {
    STARTED = 0,
    STOPPED,
    DISCONNECTED,
    GOT_IP,
    SCAN_DONE,
  };

  struct PendingWifiEvent {
    PendingWifiEventType type{PendingWifiEventType::STARTED};
    uint8_t disconnect_reason{0U};
    uint8_t scan_status{0U};
    esp_netif_ip_info_t ip_info{};
  };

  friend struct StandaloneWifiServiceTestAccess;
  static void wifi_event_handler_(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);

  esp_err_t configure_station_();
  void handle_wifi_started_();
  void handle_wifi_stopped_();
  void handle_wifi_disconnected_(uint8_t reason);
  void handle_lifecycle_connected_();
  void handle_lifecycle_disconnected_();
  void handle_scan_done_(uint8_t status);
  void maybe_run_deferred_connect_fallback_();
  void clear_cached_ip_info_();

  StandaloneWifiConfig config_{};
  WiFiLifecycleManager wifi_lifecycle_;
  standalone_wifi_callback_t connected_cb_;
  standalone_wifi_callback_t disconnected_cb_;
  standalone_wifi_scan_callback_t scan_callback_;
  esp_event_handler_instance_t wifi_event_instance_{nullptr};
  esp_event_handler_instance_t ip_event_instance_{nullptr};
  bool setup_complete_{false};
  bool wifi_connect_requested_{false};
  bool defer_connect_once_after_start_{false};
  bool deferred_connect_fallback_pending_{false};
  bool wifi_started_{false};
  bool scan_pending_{false};
  uint64_t deferred_connect_fallback_deadline_us_{0U};
  int wifi_retry_count_{0};
  esp_netif_ip_info_t cached_ip_info_{};
  static constexpr size_t kPendingWifiEventCapacity = 8U;
  PendingQueue<PendingWifiEvent, kPendingWifiEventCapacity> pending_events_{};
  std::atomic<uint32_t> dropped_events_{0U};
};

}  // namespace espectre
