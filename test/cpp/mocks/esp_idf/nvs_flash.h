/*
 * ESPectre - Mock nvs_flash.h
 *
 * Host-side mock of nvs_flash.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#ifndef NVS_FLASH_H
#define NVS_FLASH_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

esp_err_t nvs_flash_init(void);
esp_err_t nvs_flash_erase(void);

#ifdef __cplusplus
}
#endif

#endif  // NVS_FLASH_H
