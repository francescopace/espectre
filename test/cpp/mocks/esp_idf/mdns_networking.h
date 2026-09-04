/*
 * ESPectre - Mock mdns_networking.h
 *
 * Host-side mock of mdns_networking.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "mdns.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t mdns_priv_if_write(mdns_if_t tcpip_if,
                          mdns_ip_protocol_t ip_protocol,
                          const esp_ip_addr_t *ip,
                          uint16_t port,
                          uint8_t *data,
                          size_t len);
void *mdns_priv_get_packet_data(mdns_rx_packet_t *packet);
size_t mdns_priv_get_packet_len(mdns_rx_packet_t *packet);

#ifdef __cplusplus
}
#endif
