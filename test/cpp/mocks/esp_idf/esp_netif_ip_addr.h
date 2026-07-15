/*
 * ESPectre - Mock esp_netif_ip_addr.h
 *
 * Host-side mock of esp_netif_ip_addr.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#ifndef ESP_NETIF_IP_ADDR_H
#define ESP_NETIF_IP_ADDR_H

#include "esp_netif.h"

#include <cstdio>

static inline char *esp_ip4addr_ntoa(const esp_ip4_addr_t *addr, char *buf, int buflen) {
  if (addr == nullptr || buf == nullptr || buflen <= 0) {
    return nullptr;
  }
  std::snprintf(buf,
                static_cast<size_t>(buflen),
                "%u.%u.%u.%u",
                static_cast<unsigned>(ip4_addr1(addr)),
                static_cast<unsigned>(ip4_addr2(addr)),
                static_cast<unsigned>(ip4_addr3(addr)),
                static_cast<unsigned>(ip4_addr4(addr)));
  return buf;
}

#endif  // ESP_NETIF_IP_ADDR_H
