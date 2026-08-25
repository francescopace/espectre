/*
 * ESPectre - CSI PHY Filter
 *
 * HT20 sensing gate shared by ESP-IDF sensing runtimes (ESPHome, native,
 * Matter). Capture may still enable legacy LTF on original ESP32 for health;
 * detectors only consume HT20 frames.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "esp_wifi.h"
#include "sdkconfig.h"

namespace espectre {

/**
 * Return true when RX control metadata matches the HT20 sensing contract.
 *
 * Aligns with host-side `is_ht20_phy` (`phy_mode=ht`, `channel_width=20`) and
 * raw CSI PHY extraction for HT/20 MHz frames.
 */
inline bool csi_rx_is_ht20_sensing(const wifi_pkt_rx_ctrl_t &rx_ctrl) {
#if CONFIG_SOC_WIFI_HE_SUPPORT
  return rx_ctrl.cur_bb_format == RX_BB_FORMAT_HT && rx_ctrl.second == 0U;
#else
  return rx_ctrl.sig_mode == 1U && rx_ctrl.cwb == 0U;
#endif
}

inline bool csi_info_is_ht20_sensing(const wifi_csi_info_t *info) {
  return info != nullptr && csi_rx_is_ht20_sensing(info->rx_ctrl);
}

}  // namespace espectre
