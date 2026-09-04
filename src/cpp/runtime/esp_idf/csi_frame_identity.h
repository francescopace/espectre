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

#include "csi_traffic_types.h"
#include "esp_wifi.h"
#include "runtime_sensing_schema.h"

namespace espectre {

struct CsiFrameFilterConfig {
  CsiTrafficMode traffic_mode{CsiTrafficMode::INTERNAL};
  RuntimeTrafficMode internal_mode{RuntimeTrafficMode::PING};
  uint32_t local_ip_addr{0U};
  uint32_t gateway_ip_addr{0U};
  uint32_t multicast_ip_addr{0U};
  uint16_t external_udp_port{RUNTIME_CSI_TRAFFIC_UDP_PORT_DEFAULT};
  uint16_t internal_icmp_identifier{0U};
  uint8_t local_mac_addr[6]{};
};

/** Match a CSI callback to the configured ESPectre traffic source. */
bool csi_frame_matches_traffic(const wifi_csi_info_t *info,
                               const CsiFrameFilterConfig &config);

}  // namespace espectre
