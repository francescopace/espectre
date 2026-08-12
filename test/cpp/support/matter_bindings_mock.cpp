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
#include "matter_bindings_mock.h"

namespace espectre {
namespace matter_bindings_mock {

State state{};

void reset() { state = State{}; }

void MockMatterBindings::publish_motion(uint16_t endpoint_id, bool motion_detected) {
  state.motion_events.push_back(MotionPublish{endpoint_id, motion_detected});
}

void MockMatterBindings::report_fault(const char *message) {
  if (message != nullptr) {
    state.faults.emplace_back(message);
  }
}

}  // namespace matter_bindings_mock
}  // namespace espectre
