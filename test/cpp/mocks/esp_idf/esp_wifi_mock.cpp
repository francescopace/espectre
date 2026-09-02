/*
 * ESPectre - Mock esp_wifi_mock.cpp
 *
 * Host-side mock of esp_wifi_mock.cpp for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "esp_wifi.h"

esp_wifi_mock_state_t g_esp_wifi_mock{};

void esp_wifi_mock_reset(void) {
  g_esp_wifi_mock = {};
  g_esp_wifi_mock.protocol_bitmap =
      WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N | WIFI_PROTOCOL_11AX;
  g_esp_wifi_mock.protocols.ghz_2g =
      WIFI_PROTOCOL_11B | WIFI_PROTOCOL_11G | WIFI_PROTOCOL_11N | WIFI_PROTOCOL_11AX;
  g_esp_wifi_mock.protocols.ghz_5g =
      WIFI_PROTOCOL_11A | WIFI_PROTOCOL_11N | WIFI_PROTOCOL_11AC | WIFI_PROTOCOL_11AX;
  g_esp_wifi_mock.bandwidth = WIFI_BW_HT20;
  g_esp_wifi_mock.bandwidths.ghz_2g = WIFI_BW_HT20;
  g_esp_wifi_mock.bandwidths.ghz_5g = WIFI_BW_HT20;
  g_esp_wifi_mock.ps_type = WIFI_PS_NONE;
  g_esp_wifi_mock.last_set_ps_type = WIFI_PS_NONE;
  g_esp_wifi_mock.get_mac_result = ESP_OK;
  g_esp_wifi_mock.get_config_result = ESP_OK;
  g_esp_wifi_mock.get_ap_info_result = ESP_OK;
  g_esp_wifi_mock.current_ap_info.rssi = -55;
  g_esp_wifi_mock.current_ap_info.primary = 6;
  g_esp_wifi_mock.scan_start_result = ESP_OK;
  g_esp_wifi_mock.scan_get_ap_num_result = ESP_OK;
  g_esp_wifi_mock.scan_get_ap_records_result = ESP_OK;
  g_esp_wifi_mock.mac[0] = 0x7C;
  g_esp_wifi_mock.mac[1] = 0x2C;
  g_esp_wifi_mock.mac[2] = 0x67;
  g_esp_wifi_mock.mac[3] = 0x42;
  g_esp_wifi_mock.mac[4] = 0xBB;
  g_esp_wifi_mock.mac[5] = 0xAC;
  g_esp_wifi_mock.primary_channel = 6;
  g_esp_wifi_mock.second_channel = WIFI_SECOND_CHAN_NONE;
  g_esp_wifi_mock.get_protocol_result = ESP_OK;
  g_esp_wifi_mock.get_protocols_result = ESP_OK;
  g_esp_wifi_mock.get_bandwidth_result = ESP_OK;
  g_esp_wifi_mock.get_bandwidths_result = ESP_OK;
  g_esp_wifi_mock.get_promiscuous_result = ESP_OK;
  g_esp_wifi_mock.get_ps_result = ESP_OK;
  g_esp_wifi_mock.get_channel_result = ESP_OK;
  g_esp_wifi_mock.set_bandwidth_result = ESP_OK;
  g_esp_wifi_mock.set_bandwidths_result = ESP_OK;
  g_esp_wifi_mock.set_promiscuous_result = ESP_OK;
  g_esp_wifi_mock.set_protocols_result = ESP_OK;
  g_esp_wifi_mock.set_band_mode_result = ESP_OK;
  g_esp_wifi_mock.get_band_mode_result = ESP_OK;
  // Matches the ESP-IDF default on a 5 GHz-capable part, so the tests start
  // from the state real hardware boots into.
  g_esp_wifi_mock.band_mode = WIFI_BAND_MODE_AUTO;
}

namespace {
struct EspWifiMockResetInitializer {
  EspWifiMockResetInitializer() { esp_wifi_mock_reset(); }
} g_esp_wifi_mock_reset_initializer;
}  // namespace
