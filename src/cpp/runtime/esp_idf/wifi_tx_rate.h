/*
 * ESPectre - Shared Wi-Fi TX rate policy
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file wifi_tx_rate.h
 * @brief Shared ESP-IDF station and raw-frame transmit-rate policy.
 *
 * Apply the station policy after association, including reassociation, before
 * starting CSI. The CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS build setting selects
 * Auto, OFDM 6 Mbps, or HT20 MCS0 with long GI. Driver errors reach the caller.
 */

#include "sdkconfig.h"
#include "esp_wifi.h"
#include "espectre_log.h"
#include <string_view>

#ifndef CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS
#if CONFIG_IDF_TARGET_ESP32
#define CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS "6.5"
#else
#define CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS "0"
#endif
#endif

#if !CONFIG_ESP_WIFI_AMPDU_TX_ENABLED
#include "esp_private/wifi.h"
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
inline esp_err_t apply_raw_tx_rate(const wifi_ap_record_t &ap) {
  // Raw injection defaults to DSSS on 2.4 GHz; its ACKs cannot supply LLTF CSI.
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
  wifi_tx_rate_config_t rate_config{};
  rate_config.phymode = ap.primary > 14U ? WIFI_PHY_MODE_11A : WIFI_PHY_MODE_11G;
  rate_config.rate = WIFI_OFDM_TX_RATE;
  const esp_err_t err = esp_wifi_config_80211_tx(WIFI_IF_STA, &rate_config);
#else
  (void)ap;
  // ESP-IDF 5.5.5 accepts the newer API but leaves raw TX at 1 Mbps on ESP32/S2/S3/C3.
  const esp_err_t err = esp_wifi_config_80211_tx_rate(WIFI_IF_STA, WIFI_OFDM_TX_RATE);
#endif
  if (err != ESP_OK) {
    ESPECTRE_LOGE("WiFiRate", "Failed to configure raw OFDM TX rate: %s", esp_err_to_name(err));
  }
  return err;
}

/** Apply station policy; Auto and TX A-MPDU builds leave driver rate selection unchanged. */
inline esp_err_t apply_station_tx_rate() {
#if !CONFIG_ESP_WIFI_AMPDU_TX_ENABLED
  if constexpr (WIFI_TX_RATE_MBPS == 0.0f) return ESP_OK;
  wifi_ap_record_t ap{};
  const esp_err_t ap_err = esp_wifi_sta_get_ap_info(&ap);
  if (ap_err != ESP_OK) {
    ESPECTRE_LOGE("WiFiRate", "Failed to inspect associated AP for station TX rate: %s",
                  esp_err_to_name(ap_err));
    return ap_err;
  }

  // Station TX policy belongs to the connection, independently of sensing
  // and traffic generation. Reevaluate it for every AP, including roaming
  // to an AP without the PHY required by the selected rate.
  const bool fixed_rate = WIFI_TX_RATE_MBPS == 6.5f
      ? ap.phy_11n : (ap.primary > 14U || ap.phy_11g || ap.phy_11n);
  const esp_err_t err =
      esp_wifi_internal_set_fix_rate(WIFI_IF_STA, fixed_rate, WIFI_STATION_TX_RATE);
  if (err != ESP_OK) {
    ESPECTRE_LOGE("WiFiRate", "Failed to configure station TX rate: %s",
                  esp_err_to_name(err));
    return err;
  }
  if (fixed_rate) {
    ESPECTRE_LOGI("WiFiRate", "Station %s %s Mbps TX rate enabled",
                  WIFI_TX_RATE_MBPS == 6.5f ? "HT20 MCS0 LGI" : "OFDM",
                  WIFI_TX_RATE_SETTING.data());
  } else {
    ESPECTRE_LOGI("WiFiRate", "Station TX rate remains automatic: AP lacks the selected PHY");
  }
#endif
  return ESP_OK;
}

}  // namespace espectre
