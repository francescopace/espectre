/*
 * ESPectre - Matter Bindings Interface
 *
 * Thin boundary between the frontend adapter and the esp-matter stack.
 * Host-side tests provide a mock implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

#include "matter_surface.h"

namespace esphome {
namespace espectre {

class IMatterBindings {
 public:
  virtual ~IMatterBindings() = default;

  virtual void publish_motion(uint16_t endpoint_id, bool motion_detected) = 0;
  virtual void publish_periodic_state(uint16_t endpoint_id, const MatterPeriodicState &state) = 0;
  virtual void publish_threshold(uint16_t endpoint_id, float threshold) = 0;
  virtual void publish_calibrating(uint16_t endpoint_id, bool calibrating) = 0;
  virtual void report_fault(const char *message) = 0;
};

}  // namespace espectre
}  // namespace esphome
