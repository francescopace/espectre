/*
 * ESPectre - Direct Wi-Fi Snapshot
 *
 * Shared, credential-free ESP-IDF station state for Direct frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

namespace espectre {

/** `DirectWifiSnapshot::apply_state`: no Wi-Fi change in progress. */
inline constexpr char ESPECTRE_WIFI_APPLY_STATE_IDLE[] = "idle";
/** `DirectWifiSnapshot::apply_state`: the change is applied and awaits association and an address. */
inline constexpr char ESPECTRE_WIFI_APPLY_STATE_VERIFYING[] = "verifying";
/** `DirectWifiSnapshot::apply_state`: the change failed and the previous settings are being restored. */
inline constexpr char ESPECTRE_WIFI_APPLY_STATE_ROLLING_BACK[] = "rolling_back";
/** `DirectWifiSnapshot::apply_state`: the change connected and was saved. */
inline constexpr char ESPECTRE_WIFI_APPLY_STATE_APPLIED[] = "applied";
/** `DirectWifiSnapshot::apply_state`: the change failed and the previous settings are back. */
inline constexpr char ESPECTRE_WIFI_APPLY_STATE_ROLLED_BACK[] = "rolled_back";
/** `DirectWifiSnapshot::apply_state`: the rollback failed and the station needs recovery. */
inline constexpr char ESPECTRE_WIFI_APPLY_STATE_RECOVERY_REQUIRED[] = "recovery_required";

/** Whether `apply_state` reports a Wi-Fi change that is still verifying or rolling back. */
inline bool espectre_wifi_apply_in_progress(const std::string &apply_state) {
  return apply_state == ESPECTRE_WIFI_APPLY_STATE_VERIFYING ||
         apply_state == ESPECTRE_WIFI_APPLY_STATE_ROLLING_BACK;
}

/** Station configuration and association, without credentials, for the `wifi` resource. */
struct DirectWifiSnapshot {
  /** Whether the station has an SSID configured. */
  bool configured{false};
  /** Whether the station is associated. */
  bool connected{false};
  std::string ssid;
  /** Access point in use, as `AA:BB:CC:DD:EE:FF`; empty when not associated. */
  std::string bssid;
  /** `2g` or `5g` while associated; empty otherwise. */
  std::string band;
  /** Primary channel, or zero. */
  uint8_t channel{0U};
  /** Signal strength, or `INT16_MIN` when not associated. */
  int16_t rssi_dbm{INT16_MIN};
  /** Saved BSSID for this SSID, or empty for automatic selection. Available with apply_state. */
  std::string pinned_bssid{};
  /**
   * Wi-Fi transaction state, one of the `ESPECTRE_WIFI_APPLY_STATE_*` values,
   * or empty if the frontend does not report pin status. The bridge refuses a
   * raw CSI stream while espectre_wifi_apply_in_progress() holds.
   */
  std::string apply_state{};
  /** Human-readable detail for apply_state. */
  std::string apply_message{};
};

/** Read the current ESP-IDF station configuration and association without credentials. */
DirectWifiSnapshot read_direct_wifi_snapshot();

/** Read cached IPv4 link readiness without querying the Wi-Fi driver. */
bool read_direct_wifi_connected();

/** Read the active station MAC for discovery, as lowercase colon-separated hex; empty when unavailable. */
std::string read_direct_station_mac_address();

}  // namespace espectre
