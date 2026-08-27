/*
 * ESPectre - Matter Bindings Mock
 *
 * Test double for the Matter bindings boundary used by Matter frontend
 * tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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
  std::string node_label{};
};

extern State state;

void reset();

class MockMatterBindings : public IMatterBindings {
 public:
  void publish_motion(uint16_t endpoint_id, bool motion_detected) override;
  void report_fault(const char *message) override;
  bool get_node_label(std::string *label) override;
  bool set_node_label(const std::string &label) override;
};

}  // namespace matter_bindings_mock
}  // namespace espectre
