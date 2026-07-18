/*
 * ESPectre - NVS Helpers
 *
 * Shared NVS initialization helpers for ESP-IDF runtimes and firmware
 * entrypoints.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "esp_err.h"

namespace espectre {

// Initializes NVS, erasing and retrying once when the partition has no
// free pages or holds data from a newer format version.
esp_err_t nvs_init_with_erase_fallback();

}  // namespace espectre
