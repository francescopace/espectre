#pragma once

#include <cstdint>
#include <string>

#include "traffic_generator_manager.h"
#include "udp_listener.h"

namespace esphome {
namespace espectre {

enum class StimulusMode {
  INTERNAL,
  EXTERNAL,
  DISABLED,
};

struct StimulusServiceConfig {
  StimulusMode mode{StimulusMode::INTERNAL};
  uint32_t rate_pps{100U};
  TrafficGeneratorMode traffic_mode{TrafficGeneratorMode::PING};
  uint16_t udp_port{5555U};
  std::string multicast_group;
};

class StimulusService {
 public:
  void init(const StimulusServiceConfig &config);
  bool start();
  void stop();
  void loop();

  bool is_running() const;
  bool get_last_sender(sockaddr_in *out_addr) const;
  uint64_t get_raw_packets_received() const;
  uint64_t get_packets_received() const;
  const StimulusServiceConfig &config() const { return config_; }

 private:
  StimulusServiceConfig config_{};
  TrafficGeneratorManager traffic_generator_;
  UDPListener udp_listener_;
};

}  // namespace espectre
}  // namespace esphome
