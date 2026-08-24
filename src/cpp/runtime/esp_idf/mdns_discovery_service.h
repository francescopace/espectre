/*
 * ESPectre - Shared mDNS Discovery Service
 *
 * Owns the common ESP-IDF mDNS lifecycle used by firmware frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace espectre {

using MdnsTxtRecords = std::vector<std::pair<std::string, std::string>>;

enum class MdnsResponderMode : uint8_t {
  OWN_RESPONDER = 0,
  USE_EXISTING_RESPONDER,
};

struct MdnsDiscoveryServiceConfig {
  std::string hostname;
  std::string instance_name;
  std::string service_type;
  std::string service_protocol;
  uint16_t port{0U};
  MdnsTxtRecords txt_records;
  MdnsResponderMode responder_mode{MdnsResponderMode::OWN_RESPONDER};
};

class MdnsDiscoveryService {
 public:
  bool setup(const MdnsDiscoveryServiceConfig &config);
  bool update_txt(const MdnsTxtRecords &txt_records);
  void on_wifi_connected();
  void on_wifi_disconnected();
  void shutdown();

  bool initialized() const { return mdns_initialized_; }
  bool service_enabled() const { return service_enabled_; }

 private:
  bool set_service_txt_();
  void apply_netif_action_(int action);

  MdnsDiscoveryServiceConfig config_{};
  bool mdns_initialized_{false};
  bool owns_mdns_{false};
  bool service_added_{false};
  bool service_enabled_{false};
};

}  // namespace espectre
