/*
 * ESPectre - Streamer Discovery Service
 *
 * Advertises the Streamer Direct WebSocket through mDNS/DNS-SD.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "streamer_discovery_service.h"

#include <cstdio>

#include "direct_websocket_protocol.h"
#include "espectre_protocol.h"
#include "firmware_version.h"

namespace espectre {

namespace {

static constexpr const char *kServiceType = "_espectre";
static constexpr const char *kServiceProto = "_tcp";

std::string build_device_suffix(uint64_t device_id) {
  char buffer[17];
  std::snprintf(buffer, sizeof(buffer), "%016llx", static_cast<unsigned long long>(device_id));
  return std::string(buffer);
}

}  // namespace

bool StreamerDiscoveryService::setup(const StreamerDiscoveryServiceConfig &config) {
  config_ = config;
  device_id_text_ = format_espectre_device_id(config_.device_id);
  const std::string suffix = build_device_suffix(config_.device_id);
  hostname_ = "espectre-" + suffix;
  instance_name_ = "ESPectre Streamer " + device_id_text_;

  return discovery_.setup(MdnsDiscoveryServiceConfig{
      hostname_, instance_name_, kServiceType, kServiceProto, config_.direct_port, txt_records_()});
}

void StreamerDiscoveryService::on_wifi_connected(const StandaloneWifiInfo &wifi_info) {
  (void) wifi_info;
  (void) discovery_.update_txt(txt_records_());
  discovery_.on_wifi_connected();
}

void StreamerDiscoveryService::on_wifi_disconnected() { discovery_.on_wifi_disconnected(); }

void StreamerDiscoveryService::shutdown() { discovery_.shutdown(); }

MdnsTxtRecords StreamerDiscoveryService::txt_records_() const {
  return {
      {"device_id", device_id_text_},
      {"name", instance_name_},
      {"frontend", "streamer"},
      {"txtvers", "1"},
      {"protovers", "1"},
      {"path", ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT},
      {"firmware", espectre_firmware_version()},
      {"chip", config_.chip},
      {"tls", "0"},
      {"capabilities", "collect,monitor"},
      {"traffic_port", std::to_string(config_.traffic_port)},
  };
}

}  // namespace espectre
