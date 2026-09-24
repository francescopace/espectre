/*
 * ESPectre - Wi-Fi Provisioning Service
 *
 * Stores Wi-Fi credentials and applies live station provisioning changes.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "device_config_store.h"
#include "esp_err.h"
#include "standalone_wifi_service.h"

namespace espectre {

/**
 * Build-time station settings used when nothing is saved in NVS.
 *
 * The strings are copied by load_or_set_defaults(); `nullptr` means empty.
 */
struct WifiProvisioningDefaults {
  const char *ssid{nullptr};
  const char *password{nullptr};
  /** Optional access point to pin, as `AA:BB:CC:DD:EE:FF`. */
  const char *bssid{nullptr};
  /** Optional channel hint; `WIFI_CHANNEL_AUTO` (0) scans every allowed channel. */
  uint8_t channel{0U};
  /** Forwarded to `StandaloneWifiConfig::max_retry`. */
  int max_retry{8};
  /** Forwarded to `StandaloneWifiConfig::manage_csi_lifecycle`. */
  bool manage_csi_lifecycle{false};
  /** Band policy used when no saved policy exists or the saved one is unsupported. */
  WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
  /**
   * Time allowed for a staged candidate to associate and obtain an address,
   * and again for the rollback that follows a failure.
   */
  uint32_t candidate_timeout_ms{30000U};
};

/** Progress of a staged Wi-Fi configuration change. */
enum class WifiProvisioningApplyState : uint8_t {
  /** No change in progress. */
  IDLE = 0,
  /** The candidate is applied and waiting for association and an address. */
  VERIFYING,
  /** The candidate failed; the last-known-good settings are being restored. */
  ROLLING_BACK,
  /** The candidate connected and was saved. */
  APPLIED,
  /** The candidate failed and the last-known-good settings are back. */
  ROLLED_BACK,
  /** Rollback failed or timed out; reprovision over Improv Serial. */
  RECOVERY_REQUIRED,
};

/** Protocol name of an apply state, such as `verifying` or `rolled_back`. */
const char *wifi_provisioning_apply_state_name(WifiProvisioningApplyState state);

/**
 * Stores Wi-Fi credentials and applies changes with verification and rollback.
 *
 * New credentials are staged as a candidate in NVS, applied through
 * StandaloneWifiService, and saved as the station configuration only after
 * the station associates and obtains an address. A candidate that does not
 * connect within `WifiProvisioningDefaults::candidate_timeout_ms` is rolled
 * back to the last-known-good settings. A candidate interrupted by a reboot
 * resumes verification at the next load_or_set_defaults().
 *
 * @par Threading
 * Call every method from the task that owns the StandaloneWifiService, and
 * call loop() from that task's loop. Callbacks run on that task.
 */
class WifiProvisioningService {
 public:
  /** Notification without arguments, delivered on the owner task. */
  using ChangeCallback = std::function<void()>;

  /** Bind the service to the station it configures. Not owned; it must outlive this service. */
  explicit WifiProvisioningService(StandaloneWifiService *wifi_manager);

  /** Called whenever the configuration, apply state, or scan results change. */
  void set_change_callback(ChangeCallback callback);
  /**
   * Bracket every live station reconfiguration.
   *
   * `prepare_callback` runs before the station is reconfigured, for example to
   * quiesce sensing, and `resume_callback` runs after the station reconnects
   * or the reconfiguration fails.
   */
  void set_reconfigure_callbacks(ChangeCallback prepare_callback,
                                 ChangeCallback resume_callback);
  /** Bracket every access point scan, like set_reconfigure_callbacks(). */
  void set_scan_callbacks(ChangeCallback prepare_callback,
                          ChangeCallback resume_callback);
  /** Called when a staged change ends as `APPLIED` or `ROLLED_BACK`. */
  void set_apply_completed_callback(ChangeCallback callback);
  /**
   * Load the saved configuration, or adopt `defaults` when nothing is saved.
   *
   * An unsupported saved band policy falls back to the default, and a channel
   * that does not match the band becomes automatic. A pending candidate left
   * by a reboot resumes in `VERIFYING`. Load errors fall back to the defaults
   * and are reported by last_load_result(); the call itself returns `ESP_OK`.
   */
  esp_err_t load_or_set_defaults(const WifiProvisioningDefaults &defaults);
  /**
   * Load the configuration and set up the bound StandaloneWifiService with it.
   *
   * The connected callback runs after the service has checked a pending
   * candidate. Call StandaloneWifiService::start() afterwards.
   *
   * @return `ESP_ERR_INVALID_STATE` without a bound station, otherwise the
   *         result of StandaloneWifiService::setup().
   */
  esp_err_t setup_station(const WifiProvisioningDefaults &defaults,
                          standalone_wifi_callback_t connected_cb = {},
                          standalone_wifi_callback_t disconnected_cb = {});
  /**
   * Run a provisioning command: `SET_WIFI_BSSID:bssid=<mac>[&force=true]` or
   * `CLEAR_WIFI`.
   *
   * `SET_WIFI_BSSID` pins an access point of the provisioned SSID, taking its
   * channel from the latest scan; an empty `bssid` removes the pin. Pinning
   * the access point already in use is saved directly unless `force` is set.
   * `CLEAR_WIFI` erases the saved and pending configuration and leaves the
   * station unprovisioned.
   *
   * @return false for an unknown or malformed command, or while another
   *         change or scan is in progress. `message` receives the outcome.
   */
  bool handle_command(const std::string &command, std::string *message);
  /** Start an asynchronous scan; results are limited to the provisioned SSID. */
  bool request_access_point_scan(std::string *message);
  /** Stage credentials received through the standard Improv Serial RPC. */
  bool begin_serial_provisioning(const std::string &ssid,
                                 const std::string &password,
                                 std::string *message);
  /** Advance candidate verification and bounded rollback after Wi-Fi events. */
  void loop();
  /**
   * Reapply the current configuration to the station without verification.
   *
   * @return false when the station rejects the configuration; `message`
   *         receives the outcome.
   */
  bool apply_live(std::string *message);

  /** Configuration the station currently uses, including its password. */
  const StoredWifiConfig &config() const { return wifi_config_; }
  /** Whether a password is configured, for display without exposing it. */
  bool password_set() const { return !wifi_config_.password.empty(); }
  /** Result of the last NVS load; not `ESP_OK` when defaults replaced unreadable data. */
  esp_err_t last_load_result() const { return last_load_result_; }
  WifiProvisioningApplyState apply_state() const { return apply_state_; }
  /** Human-readable detail for apply_state(). */
  const std::string &apply_message() const { return apply_message_; }
  /** True while a change is `VERIFYING` or `ROLLING_BACK`. */
  bool apply_pending() const;
  /** True while an access point scan runs. */
  bool scan_pending() const { return scan_active_; }
  /** Access points of the provisioned SSID from the latest scan, strongest first. */
  const std::vector<StandaloneWifiAccessPoint> &access_points() const { return access_points_; }
  /** Human-readable outcome of the latest scan. */
  const std::string &scan_message() const { return scan_message_; }

 private:
  bool apply_config_live_(const StoredWifiConfig &config, std::string *message);
  bool begin_candidate_apply_(StoredWifiConfig candidate, std::string *message);
  void handle_connected_();
  void begin_rollback_(const char *reason);
  void set_apply_state_(WifiProvisioningApplyState state, const char *message);
  void resume_reconfigure_();
  void refresh_cached_strings_();
  void notify_changed_();

  StandaloneWifiService *wifi_manager_;
  ChangeCallback change_callback_;
  ChangeCallback prepare_reconfigure_callback_;
  ChangeCallback resume_reconfigure_callback_;
  ChangeCallback prepare_scan_callback_;
  ChangeCallback resume_scan_callback_;
  ChangeCallback apply_completed_callback_;
  StoredWifiConfig wifi_config_;
  StoredWifiConfig last_good_config_;
  StoredWifiConfig candidate_config_;
  WifiProvisioningDefaults defaults_;
  esp_err_t last_load_result_{ESP_OK};
  std::string wifi_ssid_;
  std::string wifi_password_;
  std::string wifi_bssid_;
  std::vector<StandaloneWifiAccessPoint> access_points_;
  std::string scan_message_;
  WifiProvisioningApplyState apply_state_{WifiProvisioningApplyState::IDLE};
  std::string apply_message_;
  bool candidate_apply_pending_{false};
  bool reconfigure_active_{false};
  bool scan_active_{false};
  uint32_t apply_started_ms_{0U};
};

}  // namespace espectre
