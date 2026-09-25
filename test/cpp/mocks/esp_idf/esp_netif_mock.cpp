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

namespace {
esp_netif_t station_netif = nullptr;
esp_netif_t created_netifs[8] = {};
int created_netif_count = 0;
}  // namespace

esp_netif_t *esp_netif_get_handle_from_ifkey(const char *) {
  g_esp_netif_mock.get_handle_call_count++;
  return g_esp_netif_mock.handle_available ? &station_netif : nullptr;
}

extern "C" esp_netif_t *__real_esp_netif_new(const esp_netif_config_t *) {
  if (created_netif_count >= static_cast<int>(sizeof(created_netifs) / sizeof(created_netifs[0]))) {
    return nullptr;
  }
  return &created_netifs[created_netif_count++];
}

extern "C" void __real_esp_netif_destroy(esp_netif_t *) {}

extern "C" esp_err_t __real_esp_netif_receive(esp_netif_t *netif, void *buffer, size_t len, void *eb) {
  g_esp_netif_mock.receive_call_count++;
  g_esp_netif_mock.last_netif = netif;
  g_esp_netif_mock.last_buffer = buffer;
  g_esp_netif_mock.last_len = len;
  g_esp_netif_mock.last_extra = eb;
  return g_esp_netif_mock.receive_result;
}

extern "C" esp_err_t __real_esp_netif_transmit_wrap(esp_netif_t *netif, void *buffer, size_t len, void *extra) {
  g_esp_netif_mock.transmit_call_count++;
  g_esp_netif_mock.last_netif = netif;
  g_esp_netif_mock.last_buffer = buffer;
  g_esp_netif_mock.last_len = len;
  g_esp_netif_mock.last_extra = extra;
  return g_esp_netif_mock.transmit_result;
}

void esp_netif_mock_reset(void) {
  g_esp_netif_mock = {};
  created_netif_count = 0;
  g_esp_netif_mock.get_ip_info_result = ESP_OK;
  g_esp_netif_mock.handle_available = true;
  g_esp_netif_mock.impl_index = 0;
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
