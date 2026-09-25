/*
 * ESPectre - CSI Traffic Service
 *
 * Shared policy for internal CSI traffic generation and external ingress.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "csi_traffic_service.h"

namespace espectre {

CsiTrafficServiceConfig to_csi_traffic_config(const RuntimeConfig &config) {
  CsiTrafficServiceConfig traffic_config;
  traffic_config.mode = config.traffic_generator_mode;
  traffic_config.rate_pps = config.csi_target_pps;
  traffic_config.udp_port = config.csi_traffic_udp_port;
  traffic_config.multicast_group = config.csi_traffic_multicast_group;
  return traffic_config;
}

void CsiTrafficService::init(const CsiTrafficServiceConfig &config) {
  mode_ = config.mode;
  if (mode_ != TrafficGeneratorMode::EXTERNAL) {
    traffic_generator_.init(config.rate_pps, mode_);
  }
  traffic_ingress_.init(config.udp_port);
  traffic_ingress_.set_multicast_group(config.multicast_group.empty()
                                           ? nullptr
                                           : config.multicast_group.c_str());
  traffic_ingress_.set_expected_payload(RUNTIME_CSI_TRAFFIC_MARKER_BYTES,
                                        RUNTIME_CSI_TRAFFIC_MARKER_LENGTH);
}

bool CsiTrafficService::start(uint32_t target_addr) {
  if (mode_ == TrafficGeneratorMode::EXTERNAL) {
    return traffic_ingress_.is_running() || traffic_ingress_.start();
  }
  return traffic_generator_.is_running() || traffic_generator_.start(target_addr);
}

void CsiTrafficService::stop() {
  if (traffic_generator_.is_running()) {
    traffic_generator_.stop();
  }
  if (traffic_ingress_.is_running()) {
    traffic_ingress_.stop();
  }
}

void CsiTrafficService::hold_pending_restart(bool hold) {
  traffic_generator_.hold_pending_restart(hold);
}

void CsiTrafficService::loop() {
  // A stopped generator may still be finishing its stop.
  traffic_generator_.loop();
  if (traffic_ingress_.is_running()) {
    traffic_ingress_.loop();
  }
}

void CsiTrafficService::set_packet_callback(csi_traffic_packet_callback_t callback,
                                            void *context) {
  traffic_ingress_.set_packet_callback(callback, context);
}

bool CsiTrafficService::is_running() const {
  return mode_ == TrafficGeneratorMode::EXTERNAL ? traffic_ingress_.is_running()
                                                  : traffic_generator_.is_running();
}

bool CsiTrafficService::is_quiescent() const {
  return traffic_generator_.is_quiescent();
}

bool CsiTrafficService::generator_is_stopping() const {
  return !traffic_generator_.is_quiescent() && !traffic_generator_.has_live_worker();
}

bool CsiTrafficService::source_is_active() const {
  return mode_ == TrafficGeneratorMode::EXTERNAL ? traffic_ingress_.is_running()
                                                  : traffic_generator_.has_live_worker();
}

bool CsiTrafficService::consume_generator_start_failure() {
  return traffic_generator_.consume_start_failure();
}

bool CsiTrafficService::consume_generator_stop_timeout() {
  return traffic_generator_.consume_stop_timeout();
}

bool CsiTrafficService::get_last_sender(UdpDatagramPeer *out_peer) const {
  return traffic_ingress_.get_last_sender(out_peer);
}

uint64_t CsiTrafficService::get_packets_received() const {
  return traffic_ingress_.get_packets_received();
}

uint32_t CsiTrafficService::get_generator_packets_total() const {
  return mode_ != TrafficGeneratorMode::EXTERNAL
             ? traffic_generator_.send_success_count()
             : 0U;
}

uint16_t CsiTrafficService::internal_icmp_identifier() const {
  return traffic_generator_.icmp_identifier();
}

}  // namespace espectre
