/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#pragma once

#include <cstdint>

#include "esp_err.h"

namespace espectre {

esp_err_t load_runtime_motion_hits(uint8_t *motion_on_hits, uint8_t *motion_off_hits, bool *has_saved_value);
esp_err_t save_runtime_motion_hits(uint8_t motion_on_hits, uint8_t motion_off_hits);
esp_err_t clear_runtime_motion_hits();

}  // namespace espectre
