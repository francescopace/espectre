/*
 * ESPectre - NVS Helpers
 *
 * Shared NVS initialization helpers for ESP-IDF runtimes and firmware
 * entrypoints.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esp_err.h"

namespace espectre {

// Initializes NVS, erasing and retrying once when the partition has no
// free pages or holds data from a newer format version.
esp_err_t nvs_init_with_erase_fallback();

}  // namespace espectre
