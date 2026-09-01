/*
 * ESPectre - UDP Listener
 *
 * Non-blocking UDP listener used for external CSI traffic ingress.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "csi_traffic_service.h"
#include "runtime_sensing_schema.h"
#include "udp_datagram_socket_esp_idf.h"

namespace espectre {

/**
 * UDP Listener for External Traffic Mode
 * 
 * Opens a UDP socket to receive packets from external sources.
 * The act of receiving packets triggers CSI callbacks in the WiFi driver.
 * No response is sent (fire-and-forget), minimizing network overhead.
 */
class UDPListener : public ICsiTrafficIngress {
 public:
  UDPListener() = default;
  explicit UDPListener(IUdpDatagramSocket &socket) : socket_(&socket) {}

  /**
   * Initialize the UDP listener
   * 
   * @param port UDP port to listen on (default: 5555)
   */
  void init(uint16_t port = 5555) override;
  void set_multicast_group(const char *group) override;
  void set_expected_payload(const uint8_t *payload, size_t len) override;
  void set_packet_callback(csi_traffic_packet_callback_t callback,
                           void *context = nullptr) override {
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
  bool start() override;
  
  /**
   * Stop listening and close socket
   */
  void stop() override;
  
  /**
   * Check if listener is running
   */
  bool is_running() const override { return running_; }
  
  /**
   * Get the listening port
   */
  uint16_t get_port() const { return port_; }
  uint64_t get_packets_received() const override { return packets_received_; }
  bool get_last_sender(UdpDatagramPeer *out_peer) const override;
  
  /**
   * Process incoming packets (call from loop)
   * 
   * Non-blocking read of any pending UDP packets.
   * Packets are discarded after reading - we only need the WiFi CSI callback.
   */
  void loop() override;

 private:
  UdpDatagramSocketEspIdf default_socket_{};
  IUdpDatagramSocket *socket_{&default_socket_};
  uint16_t port_{5555};
  bool running_{false};
  uint64_t packets_received_{0U};
  char multicast_group_[16]{};
  static constexpr size_t kMaxExpectedPayloadLen = RUNTIME_CSI_TRAFFIC_EXPECTED_PAYLOAD_MAX;
  std::array<uint8_t, kMaxExpectedPayloadLen> expected_payload_{};
  size_t expected_payload_len_{0U};
  std::atomic<uint32_t> last_sender_ipv4_{0U};
  std::atomic<uint16_t> last_sender_port_{0U};
  csi_traffic_packet_callback_t packet_callback_{nullptr};
  void *packet_callback_context_{nullptr};
};

}  // namespace espectre
