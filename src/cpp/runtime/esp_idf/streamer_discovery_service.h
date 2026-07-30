/*
 * ESPectre - Streamer Discovery Service
 *
 * Advertises streamer pacing endpoints through mDNS/DNS-SD.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <string>

#include "standalone_wifi_service.h"

namespace espectre {

struct StreamerDiscoveryServiceConfig {
  uint64_t device_id{0U};
  std::string chip{"unknown"};
  uint16_t traffic_port{0U};
  uint16_t collector_port{0U};
};

class StreamerDiscoveryService {
 public:
  bool setup(const StreamerDiscoveryServiceConfig &config);
  void on_wifi_connected(const StandaloneWifiInfo &wifi_info);
  void on_wifi_disconnected();
  void shutdown();

 private:
  bool initialize_mdns_();
  bool configure_identity_();
  bool configure_service_();
  bool set_service_txt_();
  void apply_netif_action_(int action);

  StreamerDiscoveryServiceConfig config_{};
  std::string hostname_;
  std::string instance_name_;
  std::string device_id_text_;
  bool mdns_initialized_{false};
  bool service_added_{false};
  bool service_enabled_{false};
};

}  // namespace espectre
