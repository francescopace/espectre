/*
 * ESPectre - NVS Helpers
 *
 * Shared NVS namespace and helpers for ESP-IDF runtimes and firmware
 * entrypoints.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "nvs_helpers.h"

#include "nvs.h"
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

namespace detail {

esp_err_t erase_espectre_nvs_keys(std::initializer_list<const char *> keys) {
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(ESPECTRE_NVS_NAMESPACE, NVS_READWRITE, &handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }
  for (const char *key : keys) {
    err = nvs_erase_key(handle, key);
    if (err != ESP_OK && err != ESP_ERR_NVS_NOT_FOUND) {
      nvs_close(handle);
      return err;
    }
  }
  err = nvs_commit(handle);
  nvs_close(handle);
  return err;
}

}  // namespace detail

}  // namespace espectre
