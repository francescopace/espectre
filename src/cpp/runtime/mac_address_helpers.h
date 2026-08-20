/*
 * ESPectre - MAC Address Helpers
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>

namespace espectre {

inline bool is_zero_mac_address(const uint8_t *mac) {
  if (mac == nullptr) {
    return true;
  }
  for (size_t index = 0U; index < 6U; ++index) {
    if (mac[index] != 0U) {
      return false;
    }
  }
  return true;
}

}  // namespace espectre
