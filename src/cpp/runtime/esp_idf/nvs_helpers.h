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
#pragma once

#include "esp_err.h"

#include <initializer_list>

namespace espectre {

/**
 * NVS namespace that holds every setting ESPectre saves.
 *
 * Erase this namespace for a complete factory reset of SDK settings. Use
 * `RuntimeFrontendController::clear_persisted_overrides()` to remove only the
 * saved sensing controls.
 */
constexpr const char *ESPECTRE_NVS_NAMESPACE = "espectre";

/**
 * Initialize NVS, erasing it and retrying once when the partition has no free
 * pages or holds data from a newer format version.
 *
 * The erase discards every saved setting, including Wi-Fi credentials.
 *
 * @return The result of the final `nvs_flash_init()`, or the erase error.
 */
esp_err_t nvs_init_with_erase_fallback();

namespace detail {

// Erase `keys` from ESPECTRE_NVS_NAMESPACE and commit. A missing key or
// namespace is not an error. Internal to the SDK's own stores.
esp_err_t erase_espectre_nvs_keys(std::initializer_list<const char *> keys);

}  // namespace detail

}  // namespace espectre
