/*
 * ESPectre - Runtime Config Validation
 *
 * Internal range checks shared by config validation and Kconfig parsing.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cmath>
#include <cstdint>

namespace espectre {

inline bool validate_runtime_float(float value, float min_value, float max_value) {
  return std::isfinite(value) && value >= min_value && value <= max_value;
}

inline bool validate_runtime_uint32(uint32_t value, uint32_t min_value, uint32_t max_value) {
  return value >= min_value && value <= max_value;
}

inline bool validate_runtime_uint8(uint8_t value, uint8_t min_value, uint8_t max_value) {
  return value >= min_value && value <= max_value;
}

}  // namespace espectre
