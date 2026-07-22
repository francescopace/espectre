/*
 * ESPectre - Wrap-aware serial sequence tracking
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

namespace espectre {

/** Track a wrapping 32-bit serial and reject duplicates or older values. */
class SerialSequenceTracker {
 public:
  bool accept(uint32_t value) {
    if (!initialized_) {
      last_ = value;
      initialized_ = true;
      return true;
    }

    const uint32_t delta = value - last_;
    if (delta == 0U || delta >= 0x80000000U) {
      return false;
    }
    last_ = value;
    return true;
  }

  void reset() {
    initialized_ = false;
    last_ = 0U;
  }

 private:
  bool initialized_{false};
  uint32_t last_{0U};
};

}  // namespace espectre
