/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#pragma once

#include "esp_err.h"
#include "runtime/runtime_config.h"

namespace espectre {

/** Load the saved traffic generator mode. */
esp_err_t load_runtime_traffic_generator_mode(TrafficGeneratorMode *mode, bool *has_saved_value);
esp_err_t save_runtime_traffic_generator_mode(TrafficGeneratorMode mode);
/** Erase the saved traffic generator mode. A missing value is not an error. */
esp_err_t clear_runtime_traffic_generator_mode();

}  // namespace espectre
