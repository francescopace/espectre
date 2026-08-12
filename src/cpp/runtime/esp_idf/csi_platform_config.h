/*
 * ESPectre - CSI Platform Configuration Helpers
 *
 * Builds ESP-IDF CSI capture settings for the HT20 sensing pipeline.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esp_err.h"
#include "esp_wifi.h"
#include "wifi_csi_interface.h"

namespace espectre {

wifi_csi_config_t build_ht20_csi_config();
esp_err_t configure_ht20_csi(IWiFiCSI *wifi_csi);

}  // namespace espectre
