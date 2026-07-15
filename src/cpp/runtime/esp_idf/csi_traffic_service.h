/*
 * ESPectre - CSI Traffic Service
 *
 * Owns CSI pacing traffic generation and external UDP pacing listeners.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <string>

#include "csi_traffic_types.h"
#include "traffic_generator_manager.h"
#include "udp_listener.h"

namespace espectre {

struct CsiTrafficServiceConfig {
  CsiTrafficMode mode{CsiTrafficMode::INTERNAL};
  uint32_t rate_pps{100U};
  bool adaptive{true};
  TrafficGeneratorMode traffic_mode{TrafficGeneratorMode::PING};
  uint16_t udp_port{5555U};
  std::string multicast_group;
  std::string expected_payload;
};

class CsiTrafficService {
 public:
  void init(const CsiTrafficServiceConfig &config);
  bool start(uint32_t gateway_addr = 0U);
  void stop();
  void loop();

  bool is_running() const;
  bool get_last_sender(sockaddr_in *out_addr) const;
  uint64_t get_packets_received() const;
  void observe_accepted_csi(uint64_t accepted_csi_total);

 private:
  CsiTrafficMode mode_{CsiTrafficMode::INTERNAL};
  TrafficGeneratorManager traffic_generator_;
  UDPListener udp_listener_;
};

}  // namespace espectre
