/*
 * ESPectre - Mock esp_netif_mock.cpp
 *
 * Host-side mock of esp_netif_mock.cpp for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "esp_netif.h"

esp_netif_mock_state_t g_esp_netif_mock{};

void esp_netif_mock_reset(void) {
  g_esp_netif_mock = {};
  g_esp_netif_mock.get_ip_info_result = ESP_OK;
  g_esp_netif_mock.ip_addr =
      ((uint32_t)192U << 0U) | ((uint32_t)168U << 8U) | ((uint32_t)1U << 16U) | ((uint32_t)100U << 24U);
  g_esp_netif_mock.netmask_addr =
      ((uint32_t)255U << 0U) | ((uint32_t)255U << 8U) | ((uint32_t)255U << 16U);
  g_esp_netif_mock.gw_addr =
      ((uint32_t)192U << 0U) | ((uint32_t)168U << 8U) | ((uint32_t)1U << 16U) | ((uint32_t)1U << 24U);
}

namespace {
struct EspNetifMockResetInitializer {
  EspNetifMockResetInitializer() { esp_netif_mock_reset(); }
} g_esp_netif_mock_reset_initializer;
}  // namespace
