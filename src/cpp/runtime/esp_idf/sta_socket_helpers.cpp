/*
 * ESPectre - Station Socket Helpers
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "sta_socket_helpers.h"

#include <cerrno>
#include <cinttypes>
#include <net/if.h>

#include "esp_netif.h"
#include "espectre_log.h"
#include "lwip/sockets.h"

namespace espectre {

bool bind_socket_to_sta_interface(int sock, const char *log_tag, const char *purpose) {
  esp_netif_t *netif = esp_netif_get_handle_from_ifkey("WIFI_STA_DEF");
  if (netif == nullptr) {
    ESPECTRE_LOGW(log_tag, "Failed to get STA netif for %s socket", purpose);
    return false;
  }

  const int if_index = esp_netif_get_netif_impl_index(netif);
  if (if_index <= 0) {
    ESPECTRE_LOGW(log_tag, "Invalid STA netif index for %s socket: %d", purpose, if_index);
    return false;
  }

  struct ifreq iface{};
  if (if_indextoname(static_cast<unsigned>(if_index), iface.ifr_name) == nullptr) {
    ESPECTRE_LOGW(log_tag, "Failed to resolve STA interface name for %s socket index %" PRIu32,
             purpose, static_cast<uint32_t>(if_index));
    return false;
  }

  if (setsockopt(sock, SOL_SOCKET, SO_BINDTODEVICE, &iface, sizeof(iface)) != 0) {
    ESPECTRE_LOGW(log_tag, "Failed to bind %s socket to %s (errno=%d)", purpose, iface.ifr_name, errno);
    return false;
  }
  return true;
}

}  // namespace espectre
