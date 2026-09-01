/*
 * ESPectre - ESP-IDF UDP Datagram Socket
 *
 * lwIP implementation of the shared UDP datagram boundary.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "udp_datagram_socket.h"

namespace espectre {

class UdpDatagramSocketEspIdf : public IUdpDatagramSocket {
 public:
  bool open(uint16_t port, const char *multicast_group) override;
  void close() override;
  UdpReceiveResult receive(uint8_t *buffer,
                           size_t buffer_len,
                           size_t *received_len,
                           UdpDatagramPeer *peer) override;

 private:
  int socket_fd_{-1};
};

}  // namespace espectre
