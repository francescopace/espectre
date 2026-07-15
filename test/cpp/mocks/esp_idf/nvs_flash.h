/*
 * ESPectre - Mock nvs_flash.h
 *
 * Host-side mock of nvs_flash.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
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
