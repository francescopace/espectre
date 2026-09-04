/*
 * ESPectre - UDP Listener
 *
 * Non-blocking UDP listener used for external CSI traffic ingress.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "udp_listener.h"
#include "espectre_log.h"
#include <cinttypes>
#include <cstring>

namespace espectre {

static const char *UDP_LISTENER_TAG = "UDPListener";
static constexpr uint16_t UDP_LISTENER_MAX_PACKETS_PER_LOOP = 64;

void UDPListener::init(uint16_t port) {
  port_ = port;
  running_ = false;
  socket_->close();
  packets_received_ = 0U;
  expected_payload_len_ = 0U;
  expected_payload_.fill(0U);
  last_sender_ipv4_.store(0U, std::memory_order_relaxed);
  last_sender_port_.store(0U, std::memory_order_relaxed);
  ESPECTRE_LOGD(UDP_LISTENER_TAG, "UDP listener initialized: port=%u", port_);
}

void UDPListener::set_multicast_group(const char *group) {
  if (group == nullptr) {
    multicast_group_[0] = '\0';
    return;
  }

  std::strncpy(multicast_group_, group, sizeof(multicast_group_) - 1);
  multicast_group_[sizeof(multicast_group_) - 1] = '\0';
}

void UDPListener::set_expected_payload(const uint8_t *payload, size_t len) {
  if (payload == nullptr || len == 0U) {
    expected_payload_len_ = 0U;
    expected_payload_.fill(0U);
    return;
  }

  if (len > expected_payload_.size()) {
    len = expected_payload_.size();
  }
  std::memcpy(expected_payload_.data(), payload, len);
  expected_payload_len_ = len;
}

bool UDPListener::start() {
  if (running_) {
    ESPECTRE_LOGW(UDP_LISTENER_TAG, "UDP listener already running");
    return true;
  }
  
  if (!socket_->open(port_, multicast_group_)) {
    return false;
  }
  
  running_ = true;
  ESPECTRE_LOGI(UDP_LISTENER_TAG,
           "UDP listener started: port=%u%s%s",
           port_,
           multicast_group_[0] != '\0' ? " mcast=" : "",
           multicast_group_[0] != '\0' ? multicast_group_ : "");
  
  return true;
}

void UDPListener::stop() {
  if (!running_) {
    return;
  }
  
  socket_->close();
  
  running_ = false;
  ESPECTRE_LOGI(UDP_LISTENER_TAG, "UDP listener stopped");
}

bool UDPListener::get_last_sender(UdpDatagramPeer *out_peer) const {
  if (out_peer == nullptr) {
    return false;
  }

  const uint32_t ipv4 = last_sender_ipv4_.load(std::memory_order_relaxed);
  if (ipv4 == 0U) {
    return false;
  }

  out_peer->ipv4_addr = ipv4;
  out_peer->port = last_sender_port_.load(std::memory_order_relaxed);
  return true;
}

void UDPListener::loop() {
  if (!running_) {
    return;
  }
  
  // Non-blocking receive - just drain any pending packets
  // The CSI callback is triggered by the WiFi driver when packets arrive,
  // we just need to consume them so the socket buffer doesn't fill up
  uint8_t buffer[64];
  
  // Drain a bounded burst of pending packets (non-blocking). A slightly larger
  // per-loop budget helps slower chips keep up with 100 pps collector
  // traffic when the main loop is busy with Wi-Fi or telemetry work.
  for (uint16_t drained = 0; drained < UDP_LISTENER_MAX_PACKETS_PER_LOOP; drained++) {
    size_t received_len = 0U;
    UdpDatagramPeer peer{};
    const UdpReceiveResult result =
        socket_->receive(buffer, sizeof(buffer), &received_len, &peer);
    if (result != UdpReceiveResult::PACKET) {
      break;
    }
    if (expected_payload_len_ != 0U &&
        (received_len != expected_payload_len_ ||
         std::memcmp(buffer, expected_payload_.data(), expected_payload_len_) != 0)) {
      continue;
    }
    packets_received_++;
    last_sender_ipv4_.store(peer.ipv4_addr, std::memory_order_relaxed);
    last_sender_port_.store(peer.port, std::memory_order_relaxed);
    if (packet_callback_ != nullptr) {
      packet_callback_(packet_callback_context_, peer, packets_received_);
    }
  }
}

}  // namespace espectre
