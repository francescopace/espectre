/*
 * ESPectre - UDP Listener
 *
 * Non-blocking UDP listener used for pacing traffic and collector
 * discovery.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "lwip/sockets.h"

namespace espectre {

using udp_listener_packet_callback_t = void (*)(void *, const sockaddr_in &, uint64_t);

/**
 * UDP Listener for External Traffic Mode
 * 
 * Opens a UDP socket to receive packets from external sources.
 * The act of receiving packets triggers CSI callbacks in the WiFi driver.
 * No response is sent (fire-and-forget), minimizing network overhead.
 */
class UDPListener {
 public:
  /**
   * Initialize the UDP listener
   * 
   * @param port UDP port to listen on (default: 5555)
   */
  void init(uint16_t port = 5555);
  void set_multicast_group(const char *group);
  void set_expected_payload(const uint8_t *payload, size_t len);
  void set_packet_callback(udp_listener_packet_callback_t callback, void *context = nullptr) {
    packet_callback_ = callback;
    packet_callback_context_ = context;
  }
  
  /**
   * Start listening for UDP packets
   * 
   * Creates a non-blocking UDP socket bound to the configured port.
   * 
   * @return true if started successfully
   */
  bool start();
  
  /**
   * Stop listening and close socket
   */
  void stop();
  
  /**
   * Check if listener is running
   */
  bool is_running() const { return running_; }
  
  /**
   * Get the listening port
   */
  uint16_t get_port() const { return port_; }
  uint64_t get_packets_received() const { return packets_received_; }
  bool get_last_sender(sockaddr_in *out_addr) const;
  
  /**
   * Process incoming packets (call from loop)
   * 
   * Non-blocking read of any pending UDP packets.
   * Packets are discarded after reading - we only need the WiFi CSI callback.
   */
  void loop();

 private:
  int sock_{-1};
  uint16_t port_{5555};
  bool running_{false};
  uint64_t packets_received_{0U};
  char multicast_group_[16]{};
  static constexpr size_t kMaxExpectedPayloadLen = 16U;
  std::array<uint8_t, kMaxExpectedPayloadLen> expected_payload_{};
  size_t expected_payload_len_{0U};
  std::atomic<uint32_t> last_sender_ipv4_{0U};
  std::atomic<uint16_t> last_sender_port_{0U};
  udp_listener_packet_callback_t packet_callback_{nullptr};
  void *packet_callback_context_{nullptr};
};

}  // namespace espectre

