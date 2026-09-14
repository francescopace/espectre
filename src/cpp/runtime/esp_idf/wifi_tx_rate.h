/*
 * ESPectre - Shared Wi-Fi TX rate policy
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "sdkconfig.h"
#include "esp_wifi.h"
#include "espectre_log.h"

#ifndef CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS
#define CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS 6
#endif

#if CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS > 0 && !CONFIG_ESP_WIFI_AMPDU_TX_ENABLED
#include "esp_private/wifi.h"
#endif

namespace espectre {

constexpr unsigned WIFI_TX_RATE_MBPS = CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS;
static_assert(WIFI_TX_RATE_MBPS == 0U || WIFI_TX_RATE_MBPS == 6U ||
              WIFI_TX_RATE_MBPS == 12U || WIFI_TX_RATE_MBPS == 24U,
              "Wi-Fi TX rate must be 0 (auto), 6, 12, or 24 Mbps");
// Raw ACK CSI needs OFDM even when station rate selection is automatic.
constexpr wifi_phy_rate_t WIFI_OFDM_TX_RATE =
    WIFI_TX_RATE_MBPS == 24U ? WIFI_PHY_RATE_24M :
    WIFI_TX_RATE_MBPS == 12U ? WIFI_PHY_RATE_12M : WIFI_PHY_RATE_6M;

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

inline esp_err_t apply_station_tx_rate() {
#if CONFIG_ESPECTRE_WIFI_TX_RATE_MBPS > 0 && !CONFIG_ESP_WIFI_AMPDU_TX_ENABLED
  wifi_ap_record_t ap{};
  const esp_err_t ap_err = esp_wifi_sta_get_ap_info(&ap);
  if (ap_err != ESP_OK) {
    ESPECTRE_LOGE("WiFiRate", "Failed to inspect associated AP for station TX rate: %s",
                  esp_err_to_name(ap_err));
    return ap_err;
  }

  // Station TX policy belongs to the connection, independently of sensing
  // and traffic generation. Reevaluate it for every AP, including roaming
  // from an OFDM network to an 802.11b-only AP that requires automatic rates.
  const bool fixed_rate = ap.primary > 14U || ap.phy_11g || ap.phy_11n;
  const esp_err_t err =
      esp_wifi_internal_set_fix_rate(WIFI_IF_STA, fixed_rate, WIFI_OFDM_TX_RATE);
  if (err != ESP_OK) {
    ESPECTRE_LOGE("WiFiRate", "Failed to configure station TX rate: %s",
                  esp_err_to_name(err));
    return err;
  }
  if (fixed_rate) {
    ESPECTRE_LOGI("WiFiRate", "Station OFDM %u Mbps TX rate enabled", WIFI_TX_RATE_MBPS);
  } else {
    ESPECTRE_LOGI("WiFiRate", "Station TX rate remains automatic for 802.11b-only AP");
  }
#endif
  return ESP_OK;
}

}  // namespace espectre
