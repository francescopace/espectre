// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#include "network_traffic.h"

#include <atomic>
#include <cstddef>
#include "esp_attr.h"
#include "esp_netif.h"

namespace {
std::atomic<esp_netif_t *> station{nullptr};
std::atomic<uint32_t> tx_packets{0U};
std::atomic<uint32_t> rx_packets{0U};
}  // namespace

namespace espectre {
NetworkTrafficSnapshot read_network_traffic() {
  station.store(esp_netif_get_handle_from_ifkey("WIFI_STA_DEF"), std::memory_order_relaxed);
  return {tx_packets.load(std::memory_order_relaxed), rx_packets.load(std::memory_order_relaxed)};
}
}  // namespace espectre

extern "C" esp_err_t __real_esp_netif_receive(esp_netif_t *, void *, size_t, void *);
extern "C" esp_err_t __real_esp_netif_transmit_wrap(esp_netif_t *, void *, size_t, void *);

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
