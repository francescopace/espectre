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

static inline int esp_netif_get_netif_impl_index(esp_netif_t *netif) {
  (void)netif;
  return 0;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_NETIF_H
