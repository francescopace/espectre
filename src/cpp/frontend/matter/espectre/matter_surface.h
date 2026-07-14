/*
 * ESPectre - Matter Surface Mapping
 *
 * Maps runtime snapshots and events to the standard Matter occupancy surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

#include "runtime_snapshot.h"
namespace espectre {

inline bool snapshot_to_motion_detected(const RuntimeSnapshot &snapshot) {
  return snapshot.motion_state == MotionState::MOTION;
}

}  // namespace espectre
