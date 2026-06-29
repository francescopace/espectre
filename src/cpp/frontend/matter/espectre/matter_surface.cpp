/*
 * ESPectre - Matter Surface Mapping
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "matter_surface.h"

#include "runtime_config_utils.h"

namespace esphome {
namespace espectre {

MatterPeriodicState snapshot_to_periodic_state(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  MatterPeriodicState state{};
  state.movement_metric = snapshot.movement_metric;
  state.threshold = snapshot.threshold;
  state.best_pxx = snapshot.best_pxx;
  state.ready_to_publish = snapshot.ready_to_publish;
  state.gain_locked = snapshot.gain_locked;
  state.calibrating = snapshot.calibrating;
  state.detector_name = snapshot.detector_name;
  state.packets_received = packets_received;
  return state;
}

bool validate_matter_threshold(float threshold) {
  return validate_runtime_threshold(threshold);
}

}  // namespace espectre
}  // namespace esphome
