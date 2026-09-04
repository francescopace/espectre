/*
 * ESPectre - Mock esp_mac.h
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <cstring>

#include "esp_err.h"

enum esp_mac_type_t { ESP_MAC_WIFI_STA = 0 };

struct esp_mac_mock_state_t {
  esp_err_t result{ESP_OK};
  uint8_t mac[6]{0x7c, 0x2c, 0x67, 0x42, 0xbb, 0xac};
  int call_count{0};
};

inline esp_mac_mock_state_t g_esp_mac_mock{};

inline esp_err_t esp_read_mac(uint8_t* mac, esp_mac_type_t type) {
  (void)type;
  g_esp_mac_mock.call_count++;
  if (g_esp_mac_mock.result == ESP_OK && mac != nullptr) {
    std::memcpy(mac, g_esp_mac_mock.mac, sizeof(g_esp_mac_mock.mac));
  }
  return g_esp_mac_mock.result;
}
