/*
 * ESPectre - CSI Traffic Service
 *
 * Shared policy for internal CSI traffic generation and external ingress.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "csi_traffic_types.h"
#include "runtime_interface.h"
#include "udp_datagram_socket.h"

namespace espectre {

using csi_traffic_packet_callback_t = void (*)(void *, const UdpDatagramPeer &, uint64_t);

struct CsiTrafficServiceConfig {
  CsiTrafficMode mode{CsiTrafficMode::INTERNAL};
  uint32_t rate_pps{100U};
  RuntimeTrafficMode traffic_mode{RuntimeTrafficMode::PING};
  uint16_t udp_port{5555U};
  std::string multicast_group;
};

class ICsiTrafficGenerator {
 public:
  virtual ~ICsiTrafficGenerator() = default;

  virtual void init(uint32_t target_pps, RuntimeTrafficMode mode) = 0;
  virtual bool start(uint32_t gateway_addr) = 0;
  virtual void stop() = 0;
  virtual void loop() = 0;
  virtual bool is_running() const = 0;
  virtual uint32_t send_success_count() const = 0;
  virtual uint16_t icmp_identifier() const = 0;
};

class ICsiTrafficIngress {
 public:
  virtual ~ICsiTrafficIngress() = default;

  virtual void init(uint16_t port) = 0;
  virtual void set_multicast_group(const char *group) = 0;
  virtual void set_expected_payload(const uint8_t *payload, size_t len) = 0;
  virtual void set_packet_callback(csi_traffic_packet_callback_t callback,
                                   void *context = nullptr) = 0;
  virtual bool start() = 0;
  virtual void stop() = 0;
  virtual void loop() = 0;
  virtual bool is_running() const = 0;
  virtual uint64_t get_packets_received() const = 0;
  virtual bool get_last_sender(UdpDatagramPeer *out_peer) const = 0;
};

/** Project runtime configuration onto transport-independent CSI traffic policy. */
CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config);

class CsiTrafficService {
 public:
  CsiTrafficService(ICsiTrafficGenerator &traffic_generator,
                    ICsiTrafficIngress &traffic_ingress)
      : traffic_generator_(traffic_generator), traffic_ingress_(traffic_ingress) {}

  void init(const CsiTrafficServiceConfig &config);
  bool start(uint32_t gateway_addr = 0U);
  void stop();
  void loop();
  void set_packet_callback(csi_traffic_packet_callback_t callback,
                           void *context = nullptr);

  bool is_running() const;
  bool get_last_sender(UdpDatagramPeer *out_peer) const;
  uint64_t get_packets_received() const;
  uint64_t get_traffic_packets_total() const;
  uint16_t internal_icmp_identifier() const;
  CsiTrafficMode mode() const { return mode_; }

 private:
  CsiTrafficMode mode_{CsiTrafficMode::INTERNAL};
  ICsiTrafficGenerator &traffic_generator_;
  ICsiTrafficIngress &traffic_ingress_;
};

}  // namespace espectre
