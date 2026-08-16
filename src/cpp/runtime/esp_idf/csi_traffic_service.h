/*
 * ESPectre - CSI Traffic Service
 *
 * Owns CSI pacing traffic generation and external UDP pacing listeners.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

#include "csi_traffic_types.h"
#include "runtime_interface.h"
#include "traffic_generator_manager.h"
#include "udp_listener.h"

namespace espectre {

struct CsiTrafficServiceConfig {
  CsiTrafficMode mode{CsiTrafficMode::INTERNAL};
  uint32_t rate_pps{100U};
  TrafficGeneratorMode traffic_mode{TrafficGeneratorMode::PING};
  uint16_t udp_port{5555U};
  std::string multicast_group;
  std::string expected_payload;
};

TrafficGeneratorMode to_traffic_generator_mode(RuntimeTrafficMode mode);

/**
 * Project a runtime config onto the traffic service config.
 *
 * Traffic source ownership comes exclusively from `csi_traffic_mode`; the
 * positive `csi_target_pps` value never enables or disables the service.
 */
CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config);

class CsiTrafficService {
 public:
  void init(const CsiTrafficServiceConfig &config);
  bool start(uint32_t gateway_addr = 0U);
  void stop();
  void loop();
  void set_packet_callback(udp_listener_packet_callback_t callback, void *context = nullptr);

  bool is_running() const;
  bool get_last_sender(sockaddr_in *out_addr) const;
  uint64_t get_packets_received() const;
  uint64_t get_pacing_total() const;

 private:
  CsiTrafficMode mode_{CsiTrafficMode::INTERNAL};
  TrafficGeneratorManager traffic_generator_;
  UDPListener udp_listener_;
};

}  // namespace espectre
