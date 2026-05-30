/*
 * ESPectre - esp-matter Bindings
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

#include "matter_bindings.h"

namespace esphome {
namespace espectre {

class MatterEspBindings : public IMatterBindings {
 public:
  void publish_motion(uint16_t endpoint_id, bool motion_detected) override;
  void publish_periodic_state(uint16_t endpoint_id, const MatterPeriodicState &state) override;
  void publish_threshold(uint16_t endpoint_id, float threshold) override;
  void publish_calibrating(uint16_t endpoint_id, bool calibrating) override;
  void report_fault(const char *message) override;
};

}  // namespace espectre
}  // namespace esphome
