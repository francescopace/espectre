/*
 * ESPectre - CSI Frame Identity
 *
 * Matches CSI frames against the local device identity when filtering
 * traffic.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "runtime/csi_capture_profile.h"
#include "esp_wifi.h"
#include "runtime/runtime_sensing_schema.h"

namespace espectre {

struct CsiFrameFilterConfig {
  TrafficGeneratorMode traffic_mode{TrafficGeneratorMode::PING};
  uint32_t local_ip_addr{0U};
  uint32_t internal_target_ip_addr{0U};
  uint32_t multicast_ip_addr{0U};
  uint16_t external_udp_port{RUNTIME_CSI_TRAFFIC_UDP_PORT_DEFAULT};
  uint16_t internal_icmp_identifier{0U};
  uint8_t local_mac_addr[6]{};
};

/**
 * One-shot hand-off from the traffic generator to the CSI callback.
 *
 * The generator arms the credit just before it injects a frame whose sample is
 * the AP's ACK. The callback admits at most one ACK per arming, within
 * kWindowUs. Every other ACK, such as one for a TCP segment of the CSI stream,
 * is rejected in every traffic mode: admitting it would let the device's own
 * transmissions add samples and feed more transmissions.
 */
class CsiAckCredit {
 public:
  /** Below one 100 pps interval, so a credit never survives into the next send. */
  static constexpr uint32_t kWindowUs = 5000U;

  /** Allow one ACK from now on; `now_us` comes from esp_timer_get_time(). */
  void arm(uint32_t now_us) {
    armed_at_us_.store(now_us == 0U ? 1U : now_us, std::memory_order_release);
  }
  /** Withdraw an unused credit, for example after a failed send. */
  void clear() { armed_at_us_.store(0U, std::memory_order_release); }
  /** Take the credit when it is armed and fresh; true at most once per arm(). */
  bool consume(uint32_t now_us) {
    uint32_t armed_at_us = armed_at_us_.load(std::memory_order_acquire);
    if (armed_at_us == 0U || now_us - armed_at_us > kWindowUs) return false;
    return armed_at_us_.compare_exchange_strong(armed_at_us, 0U, std::memory_order_acq_rel);
  }

 private:
  std::atomic<uint32_t> armed_at_us_{0U};
};

/** The station's single credit, shared by the generator and the CSI callback. */
CsiAckCredit &csi_station_ack_credit();

/**
 * Drops 802.11 retransmissions of the last admitted frame.
 *
 * An AP that misses the station's ACK sends the same frame again, and each copy
 * raises its own CSI callback. Only the CSI callback may use an instance.
 */
class CsiRetransmissionFilter {
 public:
  /** False for a same-length retry of the last admitted frame; otherwise record it and return true. */
  bool admit(const wifi_csi_info_t &info);

 private:
  bool has_last_frame_{false};
  uint16_t last_rx_seq_{0U};
  uint16_t last_length_{0U};
  uint8_t last_source_mac_[6]{};
};

/**
 * Match configured traffic, admitting each transmission once.
 *
 * In LLTF20, an 802.11 ACK to the local station matches only when it consumes
 * `ack_credit`. Any other frame must match the configured traffic and pass
 * `retransmissions`.
 */
bool csi_frame_matches_traffic(const wifi_csi_info_t *info,
                               const CsiFrameFilterConfig &config,
                               CsiCaptureProfile profile,
                               uint32_t now_us,
                               CsiAckCredit &ack_credit,
                               CsiRetransmissionFilter &retransmissions);

}  // namespace espectre
