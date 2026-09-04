/*
 * ESPectre - UDP Datagram Socket Boundary
 *
 * Platform-neutral boundary used by UDP traffic ingress.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace espectre {

struct UdpDatagramPeer {
  // Both fields use host byte order; platform adapters own any conversion.
  uint32_t ipv4_addr{0U};
  uint16_t port{0U};
};

enum class UdpReceiveResult {
  PACKET,
  EMPTY,
  ERROR,
};

class IUdpDatagramSocket {
 public:
  virtual ~IUdpDatagramSocket() = default;

  virtual bool open(uint16_t port, const char *multicast_group) = 0;
  virtual void close() = 0;
  virtual UdpReceiveResult receive(uint8_t *buffer,
                                   size_t buffer_len,
                                   size_t *received_len,
                                   UdpDatagramPeer *peer) = 0;
};

}  // namespace espectre
