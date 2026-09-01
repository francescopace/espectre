/*
 * ESPectre - CSI Platform Configuration Helpers
 *
 * Selects and builds ESP-IDF CSI capture settings for the sensing pipeline.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esp_err.h"
#include "esp_wifi.h"
#include "csi_capture_profile.h"
#include "sdkconfig.h"
#include "wifi_csi_interface.h"

namespace espectre {

constexpr CsiCaptureProfile select_csi_capture_profile(uint8_t wifi_channel) {
#if defined(CONFIG_IDF_TARGET_ESP32) && CONFIG_IDF_TARGET_ESP32
  constexpr bool kOriginalEsp32 = true;
#else
  constexpr bool kOriginalEsp32 = false;
#endif
#if defined(CONFIG_IDF_TARGET_ESP32C5) && CONFIG_IDF_TARGET_ESP32C5
  constexpr bool kSupportsVht20 = true;
#else
  constexpr bool kSupportsVht20 = false;
#endif
  return resolve_csi_capture_profile(kOriginalEsp32, kSupportsVht20,
                                     wifi_channel);
}

wifi_csi_config_t build_csi_config(CsiCaptureProfile profile);
esp_err_t configure_csi(IWiFiCSI *wifi_csi, CsiCaptureProfile profile);

}  // namespace espectre
