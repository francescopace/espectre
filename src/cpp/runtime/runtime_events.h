#pragma once

#include <cstdint>

#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

class IRuntimeListener {
 public:
  virtual ~IRuntimeListener() = default;

  virtual void on_motion_state_changed(const RuntimeSnapshot &snapshot) {}
  virtual void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {}
  virtual void on_threshold_changed(const RuntimeSnapshot &snapshot) {}
  virtual void on_calibration_started(const RuntimeSnapshot &snapshot) {}
  virtual void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {}
  virtual void on_live_telemetry(float movement, float threshold) {}
  virtual void on_runtime_fault(const char *message) {}
};

}  // namespace espectre
}  // namespace esphome
