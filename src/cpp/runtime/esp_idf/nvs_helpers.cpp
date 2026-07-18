/*
 * ESPectre - NVS Helpers
 *
 * Shared NVS initialization helpers for ESP-IDF runtimes and firmware
 * entrypoints.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "nvs_helpers.h"

#include "nvs_flash.h"

namespace espectre {

esp_err_t nvs_init_with_erase_fallback() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    const esp_err_t erase_err = nvs_flash_erase();
    if (erase_err != ESP_OK) {
      return erase_err;
    }
    err = nvs_flash_init();
  }
  return err;
}

}  // namespace espectre
