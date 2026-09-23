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

/** Match configured traffic or, in LLTF20, an 802.11 ACK to the local station. */
bool csi_frame_matches_traffic(const wifi_csi_info_t *info,
                               const CsiFrameFilterConfig &config,
                               CsiCaptureProfile profile);

}  // namespace espectre
