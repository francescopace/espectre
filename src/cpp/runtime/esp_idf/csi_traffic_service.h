#pragma once

#include <cstdint>
#include <string>

#include "traffic_generator_manager.h"
#include "udp_listener.h"

namespace esphome {
namespace espectre {

enum class CsiTrafficMode {
  INTERNAL,
  EXTERNAL,
  PACING,
  DISABLED,
};

struct CsiTrafficServiceConfig {
  CsiTrafficMode mode{CsiTrafficMode::INTERNAL};
  uint32_t rate_pps{100U};
  TrafficGeneratorMode traffic_mode{TrafficGeneratorMode::PING};
  uint16_t udp_port{5555U};
  std::string multicast_group;
  std::string expected_payload;
};

class CsiTrafficService {
 public:
  void init(const CsiTrafficServiceConfig &config);
  bool start();
  void stop();
  void loop();

  bool is_running() const;
  bool get_last_sender(sockaddr_in *out_addr) const;
  uint64_t get_packets_received() const;
  const CsiTrafficServiceConfig &config() const { return config_; }

 private:
  CsiTrafficServiceConfig config_{};
  TrafficGeneratorManager traffic_generator_;
  UDPListener udp_listener_;
};

}  // namespace espectre
}  // namespace esphome
