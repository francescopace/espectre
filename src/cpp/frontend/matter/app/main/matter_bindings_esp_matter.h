/*
 * ESPectre - esp-matter Bindings
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>

#include "matter_bindings.h"

namespace espectre {

class MatterEspBindings : public IMatterBindings {
 public:
  void publish_motion(uint16_t endpoint_id, bool motion_detected) override;
  void report_fault(const char *message) override;
};

}  // namespace espectre
