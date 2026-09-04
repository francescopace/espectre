/*
 * ESPectre - CSI Traffic Test Fakes
 *
 * Deterministic in-memory adapters for shared traffic policy and UDP ingress.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "csi_traffic_service.h"
#include "udp_datagram_socket.h"

namespace espectre::test {

class FakeCsiTrafficGenerator : public ICsiTrafficGenerator {
 public:
  void init(uint32_t target_pps, RuntimeTrafficMode traffic_mode) override {
    rate_pps = target_pps;
    mode = traffic_mode;
    running = false;
    send_successes = 0U;
  }

  bool start(uint32_t gateway) override {
    start_calls++;
    gateway_addr = gateway;
    running = start_result;
    return start_result;
  }
  void stop() override {
    stop_calls++;
    running = false;
  }
  void loop() override { loop_calls++; }
  bool is_running() const override { return running; }
  uint32_t send_success_count() const override { return send_successes; }
  uint16_t icmp_identifier() const override { return identifier; }

  uint32_t rate_pps{0U};
  RuntimeTrafficMode mode{RuntimeTrafficMode::PING};
  uint32_t gateway_addr{0U};
  uint32_t send_successes{0U};
  uint16_t identifier{0x1234U};
  bool start_result{true};
  bool running{false};
  uint32_t start_calls{0U};
  uint32_t stop_calls{0U};
  uint32_t loop_calls{0U};
};

class FakeCsiTrafficIngress : public ICsiTrafficIngress {
 public:
  void init(uint16_t configured_port) override {
    port = configured_port;
    running = false;
    packets_received = 0U;
    last_sender = {};
  }
  void set_multicast_group(const char *group) override {
    multicast_group = group == nullptr ? "" : group;
  }
  void set_expected_payload(const uint8_t *payload, size_t len) override {
    expected_payload.assign(payload, payload + len);
  }
  void set_packet_callback(csi_traffic_packet_callback_t packet_callback,
                           void *context = nullptr) override {
    callback = packet_callback;
    callback_context = context;
  }
  bool start() override {
    start_calls++;
    running = start_result;
    return start_result;
  }
  void stop() override {
    stop_calls++;
    running = false;
  }
  void loop() override { loop_calls++; }
  bool is_running() const override { return running; }
  uint64_t get_packets_received() const override { return packets_received; }
  bool get_last_sender(UdpDatagramPeer *out_peer) const override {
    if (out_peer == nullptr || last_sender.ipv4_addr == 0U) {
      return false;
    }
    *out_peer = last_sender;
    return true;
  }

  uint16_t port{0U};
  std::string multicast_group;
  std::vector<uint8_t> expected_payload;
  UdpDatagramPeer last_sender{};
  uint64_t packets_received{0U};
  bool start_result{true};
  bool running{false};
  uint32_t start_calls{0U};
  uint32_t stop_calls{0U};
  uint32_t loop_calls{0U};
  csi_traffic_packet_callback_t callback{nullptr};
  void *callback_context{nullptr};
};

class FakeUdpDatagramSocket : public IUdpDatagramSocket {
 public:
  struct Datagram {
    std::vector<uint8_t> payload;
    UdpDatagramPeer peer;
  };

  bool open(uint16_t configured_port, const char *group) override {
    open_calls++;
    port = configured_port;
    multicast_group = group == nullptr ? "" : group;
    opened = open_result;
    return open_result;
  }
  void close() override {
    close_calls++;
    opened = false;
  }
  UdpReceiveResult receive(uint8_t *buffer,
                           size_t buffer_len,
                           size_t *received_len,
                           UdpDatagramPeer *peer) override {
    if (datagrams.empty()) {
      return UdpReceiveResult::EMPTY;
    }
    Datagram datagram = std::move(datagrams.front());
    datagrams.pop_front();
    const size_t copied = std::min(buffer_len, datagram.payload.size());
    std::memcpy(buffer, datagram.payload.data(), copied);
    *received_len = copied;
    *peer = datagram.peer;
    return UdpReceiveResult::PACKET;
  }

  void enqueue(const uint8_t *payload, size_t len, UdpDatagramPeer peer) {
    datagrams.push_back({std::vector<uint8_t>(payload, payload + len), peer});
  }

  bool open_result{true};
  bool opened{false};
  uint16_t port{0U};
  std::string multicast_group;
  uint32_t open_calls{0U};
  uint32_t close_calls{0U};
  std::deque<Datagram> datagrams;
};

}  // namespace espectre::test
