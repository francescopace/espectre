/*
 * ESPectre - Matter Surface Mapping
 *
 * Maps runtime snapshots and events to Matter clusters/attributes.
 * Standard clusters cover occupancy; vendor-specific attributes carry
 * diagnostics and runtime controls not modeled cleanly by Matter today.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

// Vendor cluster for ESPectre diagnostics and runtime controls.
// Manufacturer prefix 0xFFF1 + suffix 0xFC01 (Matter MEI convention).
constexpr uint32_t ESPECTRE_MATTER_VENDOR_CLUSTER_ID = 0xFFF1FC01;

constexpr uint32_t ESPECTRE_MATTER_ATTR_MOVEMENT_METRIC = 0x0000;
constexpr uint32_t ESPECTRE_MATTER_ATTR_THRESHOLD = 0x0001;
constexpr uint32_t ESPECTRE_MATTER_ATTR_CALIBRATING = 0x0002;
constexpr uint32_t ESPECTRE_MATTER_ATTR_READY_TO_PUBLISH = 0x0003;
constexpr uint32_t ESPECTRE_MATTER_ATTR_BEST_PXX = 0x0004;
constexpr uint32_t ESPECTRE_MATTER_ATTR_GAIN_LOCKED = 0x0005;
constexpr uint32_t ESPECTRE_MATTER_ATTR_REQUEST_RECALIBRATE = 0x0006;

struct MatterPeriodicState {
  float movement_metric{0.0f};
  float threshold{SEGMENTATION_DEFAULT_THRESHOLD};
  float best_pxx{0.0f};
  bool ready_to_publish{false};
  bool gain_locked{false};
  bool calibrating{false};
  const char *detector_name{"unknown"};
  uint32_t packets_received{0};
};

inline bool snapshot_to_motion_detected(const RuntimeSnapshot &snapshot) {
  return snapshot.motion_state == MotionState::MOTION;
}

MatterPeriodicState snapshot_to_periodic_state(const RuntimeSnapshot &snapshot, uint32_t packets_received);

bool validate_matter_threshold(float threshold);

}  // namespace espectre
}  // namespace esphome
