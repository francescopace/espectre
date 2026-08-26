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

struct DirectWifiSnapshot {
  bool configured{false};
  bool connected{false};
  std::string ssid;
  std::string bssid;
  std::string band;
  uint8_t channel{0U};
  int16_t rssi_dbm{INT16_MIN};
};

/** Read the current ESP-IDF station configuration and association without credentials. */
DirectWifiSnapshot read_direct_wifi_snapshot();

/** Read cached IPv4 link readiness without querying the Wi-Fi driver. */
bool read_direct_wifi_connected();

}  // namespace espectre
