/*
 * ESPectre - ESP-IDF UDP Datagram Socket
 *
 * lwIP implementation of the shared UDP datagram boundary.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "udp_datagram_socket_esp_idf.h"

#include <cerrno>
#include <fcntl.h>

#include "espectre_log.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"

namespace espectre {

namespace {

constexpr const char *kTag = "UdpSocket";

}  // namespace

bool UdpDatagramSocketEspIdf::open(uint16_t port, const char *multicast_group) {
  close();
  socket_fd_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (socket_fd_ < 0) {
    ESPECTRE_LOGE(kTag, "Failed to create UDP socket: errno %d", errno);
    return false;
  }

  const int flags = fcntl(socket_fd_, F_GETFL, 0);
  if (flags < 0 || fcntl(socket_fd_, F_SETFL, flags | O_NONBLOCK) < 0) {
    ESPECTRE_LOGE(kTag, "Failed to set UDP socket non-blocking: errno %d", errno);
    close();
    return false;
  }

  int reuse = 1;
  if (setsockopt(socket_fd_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    ESPECTRE_LOGW(kTag, "Failed to set SO_REUSEADDR");
  }

  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(port);
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  if (bind(socket_fd_, reinterpret_cast<sockaddr *>(&address), sizeof(address)) < 0) {
    ESPECTRE_LOGE(kTag, "Failed to bind UDP socket to port %u: errno %d", port, errno);
    close();
    return false;
  }

  if (multicast_group == nullptr || multicast_group[0] == '\0') {
    return true;
  }

  ip_mreq membership{};
  if (inet_aton(multicast_group, &membership.imr_multiaddr) == 0 ||
      !IN_MULTICAST(ntohl(membership.imr_multiaddr.s_addr))) {
    ESPECTRE_LOGE(kTag, "Invalid multicast group: %s", multicast_group);
    close();
    return false;
  }
  membership.imr_interface.s_addr = htonl(INADDR_ANY);
  if (setsockopt(socket_fd_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &membership,
                 sizeof(membership)) < 0) {
    ESPECTRE_LOGE(kTag, "Failed to join multicast group %s: errno %d",
                  multicast_group, errno);
    close();
    return false;
  }
  return true;
}

void UdpDatagramSocketEspIdf::close() {
  if (socket_fd_ >= 0) {
    ::close(socket_fd_);
    socket_fd_ = -1;
  }
}

UdpReceiveResult UdpDatagramSocketEspIdf::receive(uint8_t *buffer,
                                                  size_t buffer_len,
                                                  size_t *received_len,
                                                  UdpDatagramPeer *peer) {
  if (socket_fd_ < 0 || buffer == nullptr || received_len == nullptr || peer == nullptr) {
    return UdpReceiveResult::ERROR;
  }

  sockaddr_in source{};
  socklen_t source_len = sizeof(source);
  const ssize_t result = recvfrom(socket_fd_, buffer, buffer_len, MSG_DONTWAIT,
                                  reinterpret_cast<sockaddr *>(&source), &source_len);
  if (result < 0) {
    if (errno == EAGAIN || errno == EWOULDBLOCK) {
      return UdpReceiveResult::EMPTY;
    }
    ESPECTRE_LOGW(kTag, "recvfrom failed: errno %d", errno);
    return UdpReceiveResult::ERROR;
  }

  *received_len = static_cast<size_t>(result);
  peer->ipv4_addr = ntohl(source.sin_addr.s_addr);
  peer->port = ntohs(source.sin_port);
  return UdpReceiveResult::PACKET;
}

}  // namespace espectre
