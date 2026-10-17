/*
 * ESPectre - Direct Wi-Fi Snapshot
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "direct_wifi_snapshot_esp_idf.h"
#include "runtime/mac_address_helpers.h"

#include <cstdio>
#include <cstring>

#if defined(ESP_PLATFORM) || defined(ESPECTRE_HOST_WIFI_CONTROL_TEST)
#include <esp_netif.h>
#include <esp_wifi.h>
#endif

namespace espectre {

std::string read_direct_station_mac_address() {
#if defined(ESP_PLATFORM) || defined(ESPECTRE_HOST_WIFI_CONTROL_TEST)
  uint8_t mac[6]{};
  if (esp_wifi_get_mac(WIFI_IF_STA, mac) == ESP_OK && !is_zero_mac_address(mac) && (mac[0] & 1U) == 0U) {
    char text[18]{};
    std::snprintf(text, sizeof(text), "%02x:%02x:%02x:%02x:%02x:%02x",
                  mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    return text;
  }
#endif
  return {};
}

DirectWifiSnapshot read_direct_wifi_snapshot() {
  DirectWifiSnapshot snapshot;
#if defined(ESP_PLATFORM) || defined(ESPECTRE_HOST_WIFI_CONTROL_TEST)
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

bool read_direct_wifi_connected() {
#if defined(ESP_PLATFORM) || defined(ESPECTRE_HOST_WIFI_CONTROL_TEST)
  esp_netif_t *station = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  esp_netif_ip_info_t ip_info{};
  return station != nullptr && esp_netif_get_ip_info(station, &ip_info) == ESP_OK &&
         ip_info.ip.addr != 0U;
#else
  return false;
#endif
}

}  // namespace espectre
