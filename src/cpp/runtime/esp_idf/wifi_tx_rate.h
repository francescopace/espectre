/*
 * ESPectre - Shared Wi-Fi TX rate policy
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file wifi_tx_rate.h
 * @brief Internal transmit-rate constants and raw-frame policy.
 *
 * The CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS build setting selects Auto, OFDM
 * 6 Mbps, or HT20 MCS0 with long GI. The public station entry point is
 * `apply_station_tx_rate()` in network_traffic.h.
 */

#include "sdkconfig.h"
#include "esp_wifi.h"
#include <string_view>

#ifndef CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS
#if CONFIG_IDF_TARGET_ESP32
#define CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS "6.5"
#else
#define CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS "0"
#endif
#endif

namespace espectre {

constexpr std::string_view WIFI_TX_RATE_SETTING = CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS;
constexpr float WIFI_TX_RATE_MBPS =
    WIFI_TX_RATE_SETTING == "0" ? 0.0f :
    WIFI_TX_RATE_SETTING == "6" ? 6.0f :
    WIFI_TX_RATE_SETTING == "6.5" ? 6.5f : -1.0f;
static_assert(WIFI_TX_RATE_MBPS >= 0.0f,
              "Wi-Fi TX rate must be 0 (auto), 6, or 6.5 Mbps");
// Raw Null Data retains legacy OFDM for ACK CSI under Auto and HT selection.
constexpr wifi_phy_rate_t WIFI_OFDM_TX_RATE = WIFI_PHY_RATE_6M;
constexpr wifi_phy_rate_t WIFI_STATION_TX_RATE =
    WIFI_TX_RATE_MBPS == 6.5f ? WIFI_PHY_RATE_MCS0_LGI : WIFI_OFDM_TX_RATE;

/** Configure raw-frame OFDM transmission for the associated AP's band. */
esp_err_t apply_raw_tx_rate(const wifi_ap_record_t &ap);

}  // namespace espectre
