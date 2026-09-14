/*
 * ESPectre - Mock private Wi-Fi rate control
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esp_wifi.h"

struct EspWifiFixedRateMock {
  esp_err_t enable_result{ESP_OK};
  esp_err_t disable_result{ESP_OK};
  unsigned enable_calls{0U};
  unsigned disable_calls{0U};
  bool enabled{false};
  wifi_interface_t interface{WIFI_IF_STA};
  wifi_phy_rate_t rate{WIFI_PHY_RATE_6M};
};

inline EspWifiFixedRateMock g_esp_wifi_fixed_rate_mock;

inline esp_err_t esp_wifi_internal_set_fix_rate(wifi_interface_t interface,
                                                bool enabled, wifi_phy_rate_t rate) {
  auto &mock = g_esp_wifi_fixed_rate_mock;
  mock.interface = interface;
  mock.rate = rate;
  if (enabled) ++mock.enable_calls;
  else ++mock.disable_calls;
  const esp_err_t result = enabled ? mock.enable_result : mock.disable_result;
  if (result == ESP_OK) mock.enabled = enabled;
  return result;
}
