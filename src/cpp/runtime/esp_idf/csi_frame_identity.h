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

#include "esp_wifi.h"

namespace espectre {

bool csi_frame_matches_local_identity(const wifi_csi_info_t *info,
                                      uint32_t local_ip_addr,
                                      const uint8_t *local_mac_addr);

}  // namespace espectre
