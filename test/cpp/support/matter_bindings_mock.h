/*
 * ESPectre - Matter Bindings Mock
 *
 * Test double for the Matter bindings boundary used by Matter frontend
 * tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <string>
#include <vector>

#include "matter_bindings.h"

namespace espectre {
namespace matter_bindings_mock {

struct MotionPublish {
  uint16_t endpoint_id{0};
  bool motion_detected{false};
};

struct State {
  std::vector<MotionPublish> motion_events;
  std::vector<std::string> faults;
};

extern State state;

void reset();

class MockMatterBindings : public IMatterBindings {
 public:
  void publish_motion(uint16_t endpoint_id, bool motion_detected) override;
  void report_fault(const char *message) override;
};

}  // namespace matter_bindings_mock
}  // namespace espectre
