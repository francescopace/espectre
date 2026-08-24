/*
 * ESPectre - Streamer Discovery Service
 *
 * Advertises the Streamer Direct WebSocket through mDNS/DNS-SD.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

#include "mdns_discovery_service.h"
#include "standalone_wifi_service.h"

namespace espectre {

struct StreamerDiscoveryServiceConfig {
  uint64_t device_id{0U};
  std::string chip{"unknown"};
  uint16_t direct_port{0U};
  uint16_t traffic_port{0U};
};

class StreamerDiscoveryService {
 public:
  bool setup(const StreamerDiscoveryServiceConfig &config);
  void on_wifi_connected(const StandaloneWifiInfo &wifi_info);
  void on_wifi_disconnected();
  void shutdown();

 private:
  MdnsTxtRecords txt_records_() const;

  StreamerDiscoveryServiceConfig config_{};
  std::string hostname_;
  std::string instance_name_;
  std::string device_id_text_;
  MdnsDiscoveryService discovery_;
};

}  // namespace espectre
