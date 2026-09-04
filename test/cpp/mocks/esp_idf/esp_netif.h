/*
 * ESPectre - Mock esp_netif.h
 *
 * Host-side mock of esp_netif.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#ifndef ESP_NETIF_H
#define ESP_NETIF_H

#include "esp_err.h"
#include "lwip/ip_addr.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Network interface handle
typedef void *esp_netif_t;

typedef ip4_addr_t esp_ip4_addr_t;
typedef ip4_addr_t esp_netif_ip4_addr_t;

typedef struct {
  esp_err_t get_ip_info_result;
  uint32_t ip_addr;
  uint32_t netmask_addr;
  uint32_t gw_addr;
  int get_ip_info_call_count;
  int get_handle_call_count;
  int handle_available;
  int impl_index;
} esp_netif_mock_state_t;

extern esp_netif_mock_state_t g_esp_netif_mock;

void esp_netif_mock_reset(void);

// IP info structure
typedef struct {
  esp_netif_ip4_addr_t ip;
  esp_netif_ip4_addr_t netmask;
  esp_netif_ip4_addr_t gw;
} esp_netif_ip_info_t;

// Mock functions
static inline esp_err_t esp_netif_init(void) { return ESP_OK; }

static inline esp_netif_t *esp_netif_create_default_wifi_sta(void) {
  static esp_netif_t dummy_netif = (esp_netif_t)0x2;
  return &dummy_netif;
}

static inline esp_netif_t *esp_netif_get_handle_from_ifkey(const char *ifkey) {
  (void)ifkey;
  g_esp_netif_mock.get_handle_call_count++;
  if (!g_esp_netif_mock.handle_available) {
    return nullptr;
  }
  static esp_netif_t dummy_netif = (esp_netif_t)0x1;
  return &dummy_netif;
}

static inline esp_err_t esp_netif_get_ip_info(esp_netif_t *netif, esp_netif_ip_info_t *ip_info) {
  (void)netif;
  g_esp_netif_mock.get_ip_info_call_count++;
  if (g_esp_netif_mock.get_ip_info_result != ESP_OK) {
    return g_esp_netif_mock.get_ip_info_result;
  }
  if (ip_info) {
    ip_info->ip.addr = g_esp_netif_mock.ip_addr;
    ip_info->netmask.addr = g_esp_netif_mock.netmask_addr;
    ip_info->gw.addr = g_esp_netif_mock.gw_addr;
  }
  return g_esp_netif_mock.get_ip_info_result;
}

static inline int esp_netif_get_netif_impl_index(esp_netif_t *netif) {
  (void)netif;
  return g_esp_netif_mock.impl_index;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_NETIF_H
