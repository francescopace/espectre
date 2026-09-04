/*
 * ESPectre - Mock esp_https_ota.h
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esp_err.h"
#include "esp_http_client.h"

struct esp_https_ota_config_t {
  const esp_http_client_config_t* http_config{nullptr};
};

inline esp_err_t g_esp_https_ota_result = ESP_OK;
inline int g_esp_https_ota_calls = 0;

inline esp_err_t esp_https_ota(const esp_https_ota_config_t*) {
  g_esp_https_ota_calls++;
  return g_esp_https_ota_result;
}
