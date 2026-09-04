/*
 * ESPectre - Counter Helpers
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

inline uint64_t counter_delta(uint64_t current, uint64_t previous) {
  return current >= previous ? current - previous : current;
}

}  // namespace espectre
