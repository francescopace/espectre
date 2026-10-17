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
#include <string_view>

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

// Strict colon-separated unicast address; no zero, broadcast, or group addresses.
inline bool parse_unicast_mac_address(std::string_view text, uint8_t *out) {
  if (text.size() != 17U || out == nullptr) return false;
  const auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  uint8_t parsed[6]{};
  for (size_t i = 0; i < 6U; ++i) {
    const int high = hex(text[i * 3U]);
    const int low = hex(text[i * 3U + 1U]);
    if (high < 0 || low < 0 || (i < 5U && text[i * 3U + 2U] != ':')) return false;
    parsed[i] = static_cast<uint8_t>((high << 4) | low);
  }
  if ((parsed[0] & 1U) != 0U || is_zero_mac_address(parsed)) return false;
  for (size_t i = 0; i < 6U; ++i) out[i] = parsed[i];
  return true;
}

}  // namespace espectre
