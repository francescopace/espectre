/*
 * ESPectre - Wi-Fi BSSID Pin Service
 *
 * Persists an SSID-bound BSSID override and applies it transactionally without
 * taking ownership of frontend-managed Wi-Fi credentials.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "esp_err.h"

namespace espectre {

/** Station state as reported by the firmware's Wi-Fi stack. */
struct WifiBssidPinStationState {
  /** Whether the station has credentials. */
  bool configured{false};
  /** Whether the station is associated. */
  bool connected{false};
  /** Whether the station holds an IPv4 address. */
  bool has_ipv4{false};
  /** Provisioned network name. */
  std::string ssid;
  /** Current access point, as upper-case `AA:BB:CC:DD:EE:FF`. */
  std::string bssid;
};

/** Progress of a BSSID pin change. */
enum class WifiBssidPinApplyState : uint8_t {
  /** No change in progress. */
  IDLE = 0,
  /** The pin is applied and waiting for association and an address. */
  VERIFYING,
  /** The change failed; the previous pin is being restored. */
  ROLLING_BACK,
  /** The station connected with the requested pin, and it was saved. */
  APPLIED,
  /** The change failed or was discarded, and the previous pin is back. */
  ROLLED_BACK,
  /** Rollback failed or timed out; the station may need recovery. */
  RECOVERY_REQUIRED,
};

/** Protocol name of an apply state, such as `verifying` or `rolled_back`. */
const char *wifi_bssid_pin_apply_state_name(WifiBssidPinApplyState state);

/** Hooks that connect WifiBssidPinService to the firmware's Wi-Fi stack. */
struct WifiBssidPinServiceConfig {
  /**
   * Apply a BSSID pin to the station, or clear it for an empty `bssid`.
   *
   * Return false when the pin cannot be applied, with a reason in `message`.
   * Set `*station_transition_started` when the failure happened after the
   * station began reconnecting, so the service waits for the previous
   * configuration to come back. The pointer is `nullptr` during
   * rollback.
   */
  using ApplyCallback = std::function<bool(const std::string &bssid,
                                           std::string *message,
                                           bool *station_transition_started)>;
  /** Read the current station state. */
  using StationStateGetter = std::function<WifiBssidPinStationState()>;
  /** Notification without arguments, delivered on the owner task. */
  using ChangeCallback = std::function<void()>;

  /** Required. */
  ApplyCallback apply_callback;
  /** Required. */
  StationStateGetter station_state_getter;
  /** Optional; runs before the station is reconfigured, for example to quiesce sensing. */
  ChangeCallback prepare_callback;
  /** Optional; runs when a change ends, except in `RECOVERY_REQUIRED`. */
  ChangeCallback resume_callback;
  /** Time allowed for the station to verify a pin, and again for a rollback. Must be nonzero. */
  uint32_t candidate_timeout_ms{60000U};
};

/**
 * Persists an access point pin for the provisioned SSID and applies it with
 * verification and rollback.
 *
 * For firmware whose Wi-Fi stack owns the credentials, such as Matter
 * commissioning. The service stores only the SSID and BSSID pair in NVS. A
 * pin takes effect through `apply_callback`, is saved after the station
 * reconnects to that access point with an IPv4 address, and is rolled back
 * when that does not happen within `candidate_timeout_ms`. At boot, a stored
 * pin for the current SSID is reapplied when the station associates elsewhere.
 *
 * @par Threading
 * Call every method from one owner task, and call loop() from its loop.
 */
class WifiBssidPinService {
 public:
  /**
   * Install the hooks and load the stored pin.
   *
   * @return `ESP_ERR_INVALID_ARG` when a required hook is missing or the
   *         timeout is zero, or the NVS error that prevented loading.
   */
  esp_err_t setup(WifiBssidPinServiceConfig config);

  /**
   * Stage a pin update. An empty BSSID clears the current override.
   *
   * Pinning the access point already in use is saved directly unless `force`
   * is set.
   *
   * @return false before setup(), while another change is in progress, for a
   *         malformed BSSID, or without a provisioned SSID. `message`
   *         receives the outcome.
   */
  bool request_update(const std::string &bssid, std::string *message, bool force = false);
  /** Schedule a station snapshot refresh after a Wi-Fi or IP event. */
  void notify_station_changed();
  /** Advance verification, rollback, and boot-time enforcement. */
  void loop();

  WifiBssidPinApplyState apply_state() const { return apply_state_; }
  /** Human-readable detail for apply_state(). */
  const std::string &apply_message() const { return apply_message_; }
  /** SSID the stored pin belongs to; empty without a pin. */
  const std::string &stored_ssid() const { return stored_ssid_; }
  /** Stored pin; empty without one. */
  const std::string &stored_bssid() const { return stored_bssid_; }
  /** True while a change is `VERIFYING` or `ROLLING_BACK`. */
  bool apply_pending() const;

 private:
  bool begin_apply_(const WifiBssidPinStationState &station,
                    const std::string &target_bssid,
                    bool persist_on_success,
                    std::string *message);
  void process_station_state_(const WifiBssidPinStationState &station);
  void begin_rollback_(const char *reason);
  void finish_apply_(WifiBssidPinApplyState state, const char *message);
  esp_err_t load_stored_pin_();
  esp_err_t persist_pending_pin_(const std::string &ssid, const std::string &bssid);
  esp_err_t clear_pending_pin_();
  esp_err_t commit_candidate_pin_();
  esp_err_t persist_pin_(const std::string &ssid, const std::string &bssid);
  esp_err_t clear_stored_pin_();

  WifiBssidPinServiceConfig config_;
  WifiBssidPinApplyState apply_state_{WifiBssidPinApplyState::IDLE};
  std::string apply_message_;
  std::string stored_ssid_;
  std::string stored_bssid_;
  std::string candidate_ssid_;
  std::string candidate_bssid_;
  std::string previous_bssid_;
  std::string pending_ssid_;
  std::string pending_bssid_;
  uint32_t apply_started_ms_{0U};
  bool initialized_{false};
  bool station_refresh_pending_{false};
  bool persist_on_success_{false};
  bool reconfigure_active_{false};
  bool pending_pin_loaded_{false};
};

}  // namespace espectre
