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
#include <string>

namespace espectre {

class IMatterBindings {
 public:
  virtual ~IMatterBindings() = default;

  virtual void publish_motion(uint16_t endpoint_id, bool motion_detected) = 0;
  /** Retry any transport work that could not be scheduled immediately. */
  virtual void flush_pending() {}
  virtual void report_fault(const char *message) = 0;
  /** Read the persisted Basic Information NodeLabel. */
  virtual bool get_node_label(std::string *label) {
    (void) label;
    return false;
  }
  /** Update the persisted Basic Information NodeLabel. */
  virtual bool set_node_label(const std::string &label) {
    (void) label;
    return false;
  }
};

}  // namespace espectre
