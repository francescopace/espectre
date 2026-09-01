/*
 * ESPectre - CSI PHY Filter
 *
 * 20 MHz sensing gates shared by ESP-IDF sensing runtimes (ESPHome, native,
 * and Matter).
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
 * Aligns with host-side `assess_ht20_sensing_record` (`phy_mode=ht`, `channel_width=20`) and
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

/** Return true when RX control metadata matches a VHT/20 MHz frame. */
inline bool csi_rx_is_vht20_sensing(const wifi_pkt_rx_ctrl_t &rx_ctrl) {
#if CONFIG_SOC_WIFI_HE_SUPPORT
  return rx_ctrl.cur_bb_format == RX_BB_FORMAT_VHT && rx_ctrl.second == 0U;
#else
  (void) rx_ctrl;
  return false;
#endif
}

inline bool csi_info_is_vht20_sensing(const wifi_csi_info_t *info) {
  return info != nullptr && csi_rx_is_vht20_sensing(info->rx_ctrl);
}

}  // namespace espectre
