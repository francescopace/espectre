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

wifi_csi_config_t build_ht20_csi_config();
esp_err_t configure_ht20_csi(IWiFiCSI *wifi_csi);

}  // namespace espectre
}  // namespace esphome
