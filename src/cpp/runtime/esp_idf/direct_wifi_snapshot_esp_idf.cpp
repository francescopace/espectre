/*
 * ESPectre - Direct Wi-Fi Snapshot
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "direct_wifi_snapshot_esp_idf.h"

#include <cstdio>
#include <cstring>

#if defined(ESP_PLATFORM)
#include <esp_wifi.h>
#endif

namespace espectre {

DirectWifiSnapshot read_direct_wifi_snapshot() {
  DirectWifiSnapshot snapshot;
#if defined(ESP_PLATFORM)
  wifi_config_t config{};
  if (esp_wifi_get_config(WIFI_IF_STA, &config) == ESP_OK) {
    const char *ssid = reinterpret_cast<const char *>(config.sta.ssid);
    snapshot.ssid.assign(ssid, strnlen(ssid, sizeof(config.sta.ssid)));
    snapshot.configured = !snapshot.ssid.empty();
  }

  wifi_ap_record_t access_point{};
  if (esp_wifi_sta_get_ap_info(&access_point) == ESP_OK) {
    snapshot.connected = true;
    const char *ssid = reinterpret_cast<const char *>(access_point.ssid);
    snapshot.ssid.assign(ssid, strnlen(ssid, sizeof(access_point.ssid)));
    char bssid[18]{};
    std::snprintf(bssid,
                  sizeof(bssid),
                  "%02X:%02X:%02X:%02X:%02X:%02X",
                  access_point.bssid[0],
                  access_point.bssid[1],
                  access_point.bssid[2],
                  access_point.bssid[3],
                  access_point.bssid[4],
                  access_point.bssid[5]);
    snapshot.bssid = bssid;
    snapshot.channel = access_point.primary;
    snapshot.band = access_point.primary > 14U ? "5g" : "2g";
    snapshot.rssi_dbm = access_point.rssi;
  }
#endif
  return snapshot;
}

}  // namespace espectre
