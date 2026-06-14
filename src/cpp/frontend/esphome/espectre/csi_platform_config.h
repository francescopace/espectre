/*
 * ESPectre - CSI Platform Configuration Helpers
 *
 * Shared helpers for configuring ESP-IDF CSI capture consistently across
 * frontends and runtime implementations.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_err.h"
#include "esp_wifi.h"
#include "wifi_csi_interface.h"


namespace esphome {
namespace espectre {

inline wifi_csi_config_t build_ht20_csi_config() {
#if CONFIG_IDF_TARGET_ESP32C5
  return wifi_csi_config_t{
      .enable = 1,
      .acquire_csi_legacy = 0,
      .acquire_csi_force_lltf = 0,
      .acquire_csi_ht20 = 1,
      .acquire_csi_ht40 = 0,
      .acquire_csi_vht = 0,
      .acquire_csi_su = 0,
      .acquire_csi_mu = 0,
      .acquire_csi_dcm = 0,
      .acquire_csi_beamformed = 0,
      .acquire_csi_he_stbc_mode = 0,
      .val_scale_cfg = 0,
      .dump_ack_en = 0,
  };
#elif CONFIG_IDF_TARGET_ESP32C6
  return wifi_csi_config_t{
      .enable = 1,
      .acquire_csi_legacy = 0,
      .acquire_csi_ht20 = 1,
      .acquire_csi_ht40 = 0,
      .acquire_csi_su = 0,
      .acquire_csi_mu = 0,
      .acquire_csi_dcm = 0,
      .acquire_csi_beamformed = 0,
      .acquire_csi_he_stbc = 0,
      .val_scale_cfg = 0,
      .dump_ack_en = 0,
  };
#else
  return wifi_csi_config_t{
      .lltf_en = false,
      .htltf_en = true,
      .stbc_htltf2_en = false,
      .ltf_merge_en = false,
      .channel_filter_en = false,
      .manu_scale = false,
      .shift = 0,
      .dump_ack_en = false,
  };
#endif
}

inline esp_err_t configure_ht20_csi(IWiFiCSI *wifi_csi) {
  if (wifi_csi == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }

  const wifi_csi_config_t csi_config = build_ht20_csi_config();
  return wifi_csi->set_csi_config(&csi_config);
}

}  // namespace espectre
}  // namespace esphome
