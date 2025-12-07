#ifndef ESP_NETIF_H
#define ESP_NETIF_H

#include "esp_err.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Network interface handle
typedef void *esp_netif_t;

// IP address structure
typedef struct {
  uint32_t addr;
} esp_netif_ip4_addr_t;

// IP info structure
typedef struct {
  esp_netif_ip4_addr_t ip;
  esp_netif_ip4_addr_t netmask;
  esp_netif_ip4_addr_t gw;
} esp_netif_ip_info_t;

// Mock functions
static inline esp_netif_t *esp_netif_get_handle_from_ifkey(const char *ifkey) {
  (void)ifkey;
  // Return a non-null pointer for testing
  static esp_netif_t dummy_netif = (esp_netif_t)0x1;
  return &dummy_netif;
}

static inline esp_err_t esp_netif_get_ip_info(esp_netif_t *netif, esp_netif_ip_info_t *ip_info) {
  (void)netif;
  if (ip_info) {
    // Set default IP info for testing
    ip_info->ip.addr = 0xC0A80164;      // 192.168.1.100
    ip_info->netmask.addr = 0xFFFFFF00; // 255.255.255.0
    ip_info->gw.addr = 0xC0A80101;      // 192.168.1.1
  }
  return ESP_OK;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_NETIF_H

