/*
 * ESPectre - Matter Surface Mapping
 *
 * Maps runtime snapshots and events to the standard Matter occupancy
 * surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "espectre_sdk.h"
#include <cstdint>
#include <string>

namespace espectre {

inline bool snapshot_to_motion_detected(const RuntimeSnapshot &snapshot) {
  return snapshot.motion_state == MotionState::MOTION;
}

/**
 * Basic Information serial number for builds without factory data: stable and
 * unique per chip, derived from the base MAC, and not secret.
 */
inline std::string matter_serial_number_from_mac(const uint8_t (&mac)[6]) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  std::string serial = "ESPECTRE-";
  for (uint8_t byte : mac) {
    serial.push_back(kHex[byte >> 4U]);
    serial.push_back(kHex[byte & 0x0FU]);
  }
  return serial;
}

}  // namespace espectre
