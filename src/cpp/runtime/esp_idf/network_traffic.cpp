// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include "network_traffic.h"

#include <atomic>
#include <cstddef>
#include <cstring>
#include "esp_attr.h"
#include "esp_netif.h"

namespace {
std::atomic<uint32_t> tx_packets{0U};
std::atomic<uint32_t> rx_packets{0U};
// Pointer captured when WIFI_STA_DEF is created. The receive and transmit
// hooks run on the Wi-Fi RX and TX path, which LWIP_IRAM_OPTIMIZATION keeps in
// IRAM for speed, so they only compare this pointer instead of calling flash
// code for every packet. Resolving the handle on the runtime loop would take
// the lwIP core lock.
std::atomic<esp_netif_t *> station{nullptr};

void remember_station(esp_netif_t *netif, const esp_netif_config_t *config) {
  if (netif == nullptr || config == nullptr || config->base == nullptr || config->base->if_key == nullptr) {
    return;
  }
  if (std::strcmp(config->base->if_key, "WIFI_STA_DEF") == 0) {
    station.store(netif, std::memory_order_relaxed);
  }
}
}  // namespace

namespace espectre {
NetworkTrafficSnapshot read_network_traffic() {
  return {tx_packets.load(std::memory_order_relaxed), rx_packets.load(std::memory_order_relaxed)};
}
}  // namespace espectre

extern "C" esp_netif_t *__real_esp_netif_new(const esp_netif_config_t *);
extern "C" void __real_esp_netif_destroy(esp_netif_t *);
extern "C" esp_err_t __real_esp_netif_receive(esp_netif_t *, void *, size_t, void *);
extern "C" esp_err_t __real_esp_netif_transmit_wrap(esp_netif_t *, void *, size_t, void *);

extern "C" esp_netif_t *__wrap_esp_netif_new(const esp_netif_config_t *config) {
  esp_netif_t *netif = __real_esp_netif_new(config);
  remember_station(netif, config);
  return netif;
}

extern "C" void __wrap_esp_netif_destroy(esp_netif_t *netif) {
  esp_netif_t *expected = netif;
  station.compare_exchange_strong(expected, nullptr, std::memory_order_relaxed);
  __real_esp_netif_destroy(netif);
}

extern "C" esp_err_t IRAM_ATTR __wrap_esp_netif_receive(esp_netif_t *netif, void *buffer, size_t len, void *eb) {
  // Count delivery from the driver, including packets the stack later drops.
  if (netif != nullptr && netif == station.load(std::memory_order_relaxed)) {
    rx_packets.fetch_add(1U, std::memory_order_relaxed);
  }
  return __real_esp_netif_receive(netif, buffer, len, eb);
}

extern "C" esp_err_t IRAM_ATTR __wrap_esp_netif_transmit_wrap(esp_netif_t *netif, void *buffer, size_t len, void *netstack_buffer) {
  const esp_err_t result = __real_esp_netif_transmit_wrap(netif, buffer, len, netstack_buffer);
  if (result == ESP_OK && netif != nullptr && netif == station.load(std::memory_order_relaxed)) {
    tx_packets.fetch_add(1U, std::memory_order_relaxed);
  }
  return result;
}
