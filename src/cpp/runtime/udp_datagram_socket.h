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

/** Sender of a UDP datagram. Both fields use host byte order. */
struct UdpDatagramPeer {
  uint32_t ipv4_addr{0U};
  uint16_t port{0U};
};

/** Outcome of IUdpDatagramSocket::receive(). */
enum class UdpReceiveResult {
  /** A datagram was received. */
  PACKET,
  /** Nothing is waiting. */
  EMPTY,
  /** The socket failed or is closed. */
  ERROR,
};

/** Non-blocking UDP socket, the platform seam behind the external traffic listener. */
class IUdpDatagramSocket {
 public:
  virtual ~IUdpDatagramSocket() = default;

  /**
   * Bind to `port` on every interface and join `multicast_group` unless it is
   * `nullptr` or empty. Returns false when either step fails.
   */
  virtual bool open(uint16_t port, const char *multicast_group) = 0;
  /** Close the socket. Safe to repeat. */
  virtual void close() = 0;
  /** Receive one datagram into `buffer` without blocking. */
  virtual UdpReceiveResult receive(uint8_t *buffer,
                                   size_t buffer_len,
                                   size_t *received_len,
                                   UdpDatagramPeer *peer) = 0;
};

}  // namespace espectre
