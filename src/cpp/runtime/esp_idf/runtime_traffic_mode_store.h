/*
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#pragma once

#include "esp_err.h"
#include "runtime_interface.h"

namespace espectre {

esp_err_t load_runtime_csi_traffic_mode(CsiTrafficMode *mode, bool *has_saved_value);
esp_err_t load_runtime_traffic_generator_mode(RuntimeTrafficMode *mode, bool *has_saved_value);
esp_err_t save_runtime_csi_traffic_mode(CsiTrafficMode mode);
esp_err_t save_runtime_traffic_generator_mode(RuntimeTrafficMode mode);

}  // namespace espectre
