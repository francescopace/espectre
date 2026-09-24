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
#include "runtime/pending_queue.h"
#include "wifi_lifecycle.h"

namespace espectre {

/** Connection-state callback, delivered from StandaloneWifiService::loop(). */
using standalone_wifi_callback_t = std::function<void()>;

/** One access point reported by StandaloneWifiService::request_scan(). */
struct StandaloneWifiAccessPoint {
  std::string ssid;
  /** Upper-case, colon-separated MAC address. */
  std::string bssid;
  int8_t rssi_dbm{0};
  /** Primary channel. */
  uint8_t channel{0U};
};

/**
 * Scan result callback, delivered from StandaloneWifiService::loop().
 *
 * Receives `ESP_OK` or the driver error, and at most 32 access points sorted
 * by descending RSSI.
 */
using standalone_wifi_scan_callback_t =
    std::function<void(esp_err_t, const std::vector<StandaloneWifiAccessPoint> &)>;

/**
 * Station settings for StandaloneWifiService.
 *
 * The strings are borrowed, null-terminated values. Keep them alive until
 * shutdown or until update_station_config() replaces them.
 */
struct StandaloneWifiConfig {
  /** Network name, up to 32 bytes. Empty leaves the station idle. */
  const char *ssid{""};
  /** Passphrase, up to 64 bytes. Empty for open networks. */
  const char *password{""};
  /**
   * Optional access point to pin, as `AA:BB:CC:DD:EE:FF`.
   *
   * Empty lets the driver pick the strongest access point with this SSID.
   */
  const char *bssid{""};
  /** Optional channel hint; `WIFI_CHANNEL_AUTO` (0) scans every allowed channel. */
  uint8_t channel{0U};
  /** Immediate retries per burst; exhausted bursts restart after 30 seconds. Zero skips immediate retries. */
  int max_retry{8};
  /**
   * Register the WiFiLifecycleManager handlers in this service.
   *
   * Leave it false when `RuntimeFrontendController` runs, because the runtime
   * registers its own. Set it only when this service alone applies the CSI
   * radio policy and delivers connection callbacks.
   */
  bool manage_csi_lifecycle{false};
  /** Band the station may use. Fixed for the life of one setup(). */
  WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
};

/** Station state reported by StandaloneWifiService::get_info(). */
struct StandaloneWifiInfo {
  /** True while associated with a cached IPv4 address. */
  bool connected{false};
  /** Dotted IPv4 address, or empty without one. */
  char ip_address[16]{};
  /** Upper-case, colon-separated station MAC address. */
  char mac_address[18]{};
  /** Primary channel of the association, or zero. */
  uint8_t channel{0U};
};

/**
 * Owns the ESP-IDF Wi-Fi station for firmware without its own Wi-Fi stack.
 *
 * The service creates the station netif and driver, connects, retries in
 * bounded bursts, and reports connection changes. Firmware that already owns
 * Wi-Fi, such as ESPHome, does not use it.
 *
 * @code
 * espectre::StandaloneWifiConfig wifi;
 * wifi.ssid = ssid;
 * wifi.password = password;
 * ESP_ERROR_CHECK(service.setup(wifi, on_connected, on_disconnected));
 * ESP_ERROR_CHECK(service.start());
 * // from the owner task's loop:
 * service.loop();
 * @endcode
 *
 * @par Threading
 * Call every method from one owner task. Wi-Fi and IP events are queued by
 * the event handlers, and callbacks run from loop().
 */
class StandaloneWifiService {
 public:
  /** Release the station driver, netif, and registered handlers. */
  ~StandaloneWifiService();
  /**
   * Own a new station driver and netif; the application must not already own them.
   *
   * Requires initialized NVS. The default event loop is created if needed and
   * remains available after shutdown. Credentials are borrowed; SSIDs up to
   * 32 bytes and passwords up to 64 bytes are preserved without truncation.
   * Failure releases acquired resources so setup can be retried. Calling
   * setup while active returns ESP_ERR_INVALID_STATE. Use one owner task for
   * all methods; callbacks run from loop().
   */
  esp_err_t setup(const StandaloneWifiConfig &config,
                  standalone_wifi_callback_t connected_cb = {},
                  standalone_wifi_callback_t disconnected_cb = {});
  /**
   * Start the station driver after setup().
   *
   * Connection proceeds asynchronously; the connected callback reports the
   * IPv4 address. Returns `ESP_ERR_INVALID_STATE` before setup().
   */
  esp_err_t start();
  /**
   * Replace the station settings while the service is set up.
   *
   * An active connection is dropped and re-established with the new settings.
   * Returns `ESP_ERR_INVALID_ARG` for oversized credentials, a negative retry
   * count, or an unusable channel, and `ESP_ERR_INVALID_STATE` before setup(),
   * while another reconfiguration is pending, or for a different band policy.
   */
  esp_err_t update_station_config(const StandaloneWifiConfig &config);
  /** Scan every allowed channel for the configured SSID and report its bounded snapshot from loop(). */
  esp_err_t request_scan(standalone_wifi_scan_callback_t callback);
  /** Deliver queued Wi-Fi events and callbacks, and drive reconnection. */
  void loop();
  /**
   * Read the station MAC address and, while connected, its IPv4 address and channel.
   *
   * Uses the cached address instead of querying an unassociated driver.
   * Returns false for a null `info` or when nothing is known yet.
   */
  bool get_info(StandaloneWifiInfo *info) const;
  /** Stop and release owned Wi-Fi resources. Safe to repeat; setup can be called again. */
  void shutdown();

 private:
  enum class PendingWifiEventType : uint8_t {
    STARTED = 0,
    STOPPED,
    DISCONNECTED,
    ASSOCIATED,
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
  esp_err_t apply_station_config_and_connect_();
  esp_err_t stop_wifi_for_driver_reconfigure_();
  esp_err_t restart_wifi_driver_();
  void handle_wifi_started_();
  void handle_wifi_stopped_();
  void handle_wifi_disconnected_(uint8_t reason);
  void handle_wifi_associated_();
  void maybe_restore_retained_ip_();
  void handle_lifecycle_connected_(const esp_netif_ip_info_t &ip_info);
  void handle_lifecycle_disconnected_();
  void handle_scan_done_(uint8_t status);
  void maybe_run_deferred_connect_fallback_();
  void maybe_retry_connect_();
  esp_err_t connect_station_();
  void clear_cached_ip_info_();

  StandaloneWifiConfig config_{};
  WiFiLifecycleManager wifi_lifecycle_;
  standalone_wifi_callback_t connected_cb_;
  standalone_wifi_callback_t disconnected_cb_;
  standalone_wifi_scan_callback_t scan_callback_;
  esp_event_handler_instance_t wifi_event_instance_{nullptr};
  esp_event_handler_instance_t ip_event_instance_{nullptr};
  esp_netif_t *station_netif_{nullptr};
  bool wifi_initialized_{false};
  bool setup_complete_{false};
  bool wifi_connect_requested_{false};
  bool defer_connect_once_after_start_{false};
  bool deferred_connect_fallback_pending_{false};
  bool wifi_started_{false};
  bool station_reconfigure_pending_{false};
  bool station_disconnect_pending_{false};
  bool roaming_{false};
  bool retained_ip_pending_{false};
  bool scan_pending_{false};
  uint64_t deferred_connect_fallback_deadline_us_{0U};
  uint64_t reconnect_deadline_us_{0U};
  int wifi_retry_count_{0};
  esp_netif_ip_info_t cached_ip_info_{};
  static constexpr size_t kPendingWifiEventCapacity = 8U;
  PendingQueue<PendingWifiEvent, kPendingWifiEventCapacity> pending_events_{};
  std::atomic<uint32_t> dropped_events_{0U};
};

}  // namespace espectre
