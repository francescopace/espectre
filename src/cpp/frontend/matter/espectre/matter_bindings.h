/*
 * ESPectre - Matter Bindings Interface
 *
 * Thin boundary between the frontend adapter and the esp-matter stack.
 * Host-side tests provide a mock implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

class IMatterBindings {
 public:
  virtual ~IMatterBindings() = default;

  virtual void publish_motion(uint16_t endpoint_id, bool motion_detected) = 0;
  virtual void report_fault(const char *message) = 0;
};

}  // namespace espectre
