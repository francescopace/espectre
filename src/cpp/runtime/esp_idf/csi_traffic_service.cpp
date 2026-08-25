/*
 * ESPectre - CSI Traffic Service
 *
 * Owns internal CSI traffic generation and the external UDP listener.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "csi_traffic_service.h"

namespace espectre {

TrafficGeneratorMode to_traffic_generator_mode(RuntimeTrafficMode mode) {
  return mode == RuntimeTrafficMode::PING ? TrafficGeneratorMode::PING : TrafficGeneratorMode::DNS;
}

CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config) {
  CsiTrafficServiceConfig csi_traffic_config;
  csi_traffic_config.mode = config.csi_traffic_mode;
  csi_traffic_config.rate_pps = config.csi_target_pps;
  csi_traffic_config.traffic_mode = to_traffic_generator_mode(config.traffic_generator_mode);
  csi_traffic_config.udp_port = config.csi_traffic_udp_port;
  csi_traffic_config.multicast_group = config.csi_traffic_multicast_group;
  return csi_traffic_config;
}

void CsiTrafficService::init(const CsiTrafficServiceConfig &config) {
  mode_ = config.mode;
  traffic_generator_.init(config.rate_pps, config.traffic_mode);
  udp_listener_.init(config.udp_port);
  if (!config.multicast_group.empty()) {
    udp_listener_.set_multicast_group(config.multicast_group.c_str());
  } else {
    udp_listener_.set_multicast_group(nullptr);
  }
  udp_listener_.set_expected_payload(RUNTIME_CSI_TRAFFIC_MARKER_BYTES,
                                     RUNTIME_CSI_TRAFFIC_MARKER_LENGTH);
}

bool CsiTrafficService::start(uint32_t gateway_addr) {
  switch (mode_) {
    case CsiTrafficMode::INTERNAL:
      return traffic_generator_.is_running() ? true : traffic_generator_.start(gateway_addr);
    case CsiTrafficMode::EXTERNAL:
      return udp_listener_.is_running() ? true : udp_listener_.start();
    default:
      return false;
  }
}

void CsiTrafficService::stop() {
  if (traffic_generator_.is_running()) {
    traffic_generator_.stop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.stop();
  }
}

void CsiTrafficService::loop() {
  if (traffic_generator_.is_running()) {
    traffic_generator_.loop();
  }
  if (udp_listener_.is_running()) {
    udp_listener_.loop();
  }
}

void CsiTrafficService::set_packet_callback(udp_listener_packet_callback_t callback, void *context) {
  udp_listener_.set_packet_callback(callback, context);
}

bool CsiTrafficService::is_running() const {
  switch (mode_) {
    case CsiTrafficMode::INTERNAL:
      return traffic_generator_.is_running();
    case CsiTrafficMode::EXTERNAL:
      return udp_listener_.is_running();
    default:
      return false;
  }
}

bool CsiTrafficService::get_last_sender(sockaddr_in *out_addr) const { return udp_listener_.get_last_sender(out_addr); }

uint64_t CsiTrafficService::get_packets_received() const { return udp_listener_.get_packets_received(); }

uint64_t CsiTrafficService::get_traffic_packets_total() const {
  return mode_ == CsiTrafficMode::INTERNAL
             ? static_cast<uint64_t>(traffic_generator_.send_success_count())
             : udp_listener_.get_packets_received();
}

}  // namespace espectre
