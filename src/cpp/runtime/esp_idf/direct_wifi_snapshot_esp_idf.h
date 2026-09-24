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
};

/** Read the current ESP-IDF station configuration and association without credentials. */
DirectWifiSnapshot read_direct_wifi_snapshot();

/** Read cached IPv4 link readiness without querying the Wi-Fi driver. */
bool read_direct_wifi_connected();

}  // namespace espectre
