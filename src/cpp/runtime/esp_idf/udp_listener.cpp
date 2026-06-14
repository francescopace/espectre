/*
 * ESPectre - UDP Listener Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "udp_listener.h"
#include "espectre_log.h"
#include "stimulus_protocol.h"
#include "lwip/inet.h"
#include "lwip/sockets.h"
#include "lwip/netdb.h"
#include <cstring>
#include <fcntl.h>

namespace esphome {
namespace espectre {

static const char *UDP_LISTENER_TAG = "UDPListener";

void UDPListener::init(uint16_t port) {
  port_ = port;
  running_ = false;
  sock_ = -1;
  raw_packets_received_ = 0U;
  packets_received_ = 0U;
  last_sender_ipv4_.store(0U, std::memory_order_relaxed);
  last_sender_port_.store(0U, std::memory_order_relaxed);
  ESP_LOGD(UDP_LISTENER_TAG, "UDP Listener initialized (port: %u)", port_);
}

void UDPListener::set_multicast_group(const char *group) {
  if (group == nullptr) {
    multicast_group_[0] = '\0';
    return;
  }

  std::strncpy(multicast_group_, group, sizeof(multicast_group_) - 1);
  multicast_group_[sizeof(multicast_group_) - 1] = '\0';
}

bool UDPListener::start() {
  if (running_) {
    ESP_LOGW(UDP_LISTENER_TAG, "UDP Listener already running");
    return true;
  }
  
  // Create UDP socket
  sock_ = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
  if (sock_ < 0) {
    ESP_LOGE(UDP_LISTENER_TAG, "Failed to create UDP socket: errno %d", errno);
    return false;
  }
  
  // Set socket to non-blocking
  int flags = fcntl(sock_, F_GETFL, 0);
  if (fcntl(sock_, F_SETFL, flags | O_NONBLOCK) < 0) {
    ESP_LOGW(UDP_LISTENER_TAG, "Failed to set socket non-blocking");
  }
  
  // Allow reuse of address
  int reuse = 1;
  if (setsockopt(sock_, SOL_SOCKET, SO_REUSEADDR, &reuse, sizeof(reuse)) < 0) {
    ESP_LOGW(UDP_LISTENER_TAG, "Failed to set SO_REUSEADDR");
  }
  
  // Bind to port
  struct sockaddr_in addr;
  memset(&addr, 0, sizeof(addr));
  addr.sin_family = AF_INET;
  addr.sin_port = htons(port_);
  addr.sin_addr.s_addr = htonl(INADDR_ANY);
  
  if (bind(sock_, (struct sockaddr *)&addr, sizeof(addr)) < 0) {
    ESP_LOGE(UDP_LISTENER_TAG, "Failed to bind UDP socket to port %u: errno %d", port_, errno);
    close(sock_);
    sock_ = -1;
    return false;
  }

  if (multicast_group_[0] != '\0') {
    ip_mreq membership{};
    if (inet_aton(multicast_group_, &membership.imr_multiaddr) == 0 ||
        !IN_MULTICAST(ntohl(membership.imr_multiaddr.s_addr))) {
      ESP_LOGE(UDP_LISTENER_TAG, "Invalid multicast group: %s", multicast_group_);
      close(sock_);
      sock_ = -1;
      return false;
    }
    membership.imr_interface.s_addr = htonl(INADDR_ANY);
    if (setsockopt(sock_, IPPROTO_IP, IP_ADD_MEMBERSHIP, &membership, sizeof(membership)) < 0) {
      ESP_LOGE(UDP_LISTENER_TAG, "Failed to join multicast group %s: errno %d", multicast_group_, errno);
      close(sock_);
      sock_ = -1;
      return false;
    }
  }
  
  running_ = true;
  ESP_LOGI(UDP_LISTENER_TAG,
           "UDP Listener started on port %u%s%s",
           port_,
           multicast_group_[0] != '\0' ? " mcast=" : "",
           multicast_group_[0] != '\0' ? multicast_group_ : "");
  
  return true;
}

void UDPListener::stop() {
  if (!running_) {
    return;
  }
  
  if (sock_ >= 0) {
    close(sock_);
    sock_ = -1;
  }
  
  running_ = false;
  ESP_LOGI(UDP_LISTENER_TAG, "UDP Listener stopped");
}

bool UDPListener::get_last_sender(sockaddr_in *out_addr) const {
  if (out_addr == nullptr) {
    return false;
  }

  const uint32_t ipv4 = last_sender_ipv4_.load(std::memory_order_relaxed);
  if (ipv4 == 0U) {
    return false;
  }

  std::memset(out_addr, 0, sizeof(*out_addr));
  out_addr->sin_family = AF_INET;
  out_addr->sin_addr.s_addr = ipv4;
  out_addr->sin_port = last_sender_port_.load(std::memory_order_relaxed);
  return true;
}

void UDPListener::loop() {
  if (!running_ || sock_ < 0) {
    return;
  }
  
  // Non-blocking receive - just drain any pending packets
  // The CSI callback is triggered by the WiFi driver when packets arrive,
  // we just need to consume them so the socket buffer doesn't fill up
  char buf[64];
  struct sockaddr_in src_addr;
  
  // Drain all pending packets (non-blocking)
  while (true) {
    socklen_t addr_len = sizeof(src_addr);
    ssize_t len = recvfrom(sock_, buf, sizeof(buf), 0, 
                           (struct sockaddr *)&src_addr, &addr_len);
    if (len < 0) {
      // EAGAIN/EWOULDBLOCK means no more data available
      if (errno == EAGAIN || errno == EWOULDBLOCK) {
        break;
      }
      // Other error
      ESP_LOGW(UDP_LISTENER_TAG, "recvfrom error: errno %d", errno);
      break;
    }
    raw_packets_received_++;
    StimulusMetadata metadata{};
    if (!parse_stimulus_datagram(reinterpret_cast<const uint8_t *>(buf), static_cast<size_t>(len), &metadata)) {
      continue;
    }

    packets_received_++;
    last_sender_ipv4_.store(src_addr.sin_addr.s_addr, std::memory_order_relaxed);
    last_sender_port_.store(src_addr.sin_port, std::memory_order_relaxed);
  }
}

}  // namespace espectre
}  // namespace esphome

