/*
 * ESPectre - Wrap-aware serial sequence tracking
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "core/wrapping_timestamp_order.h"

namespace espectre {

/**
 * Track a wrapping 32-bit serial and reject duplicates or older values.
 *
 * Older values that keep advancing for the resync span restart the sequence,
 * so a silence longer than half the range does not reject every later value.
 */
class SerialSequenceTracker {
 public:
  /** Construct with the span, in serial units, that resyncs after a long silence. */
  explicit SerialSequenceTracker(uint32_t resync_span) : order_(resync_span) {}

  bool accept(uint32_t value) {
    const WrappingTimestampOrderResult result = order_.observe(value);
    return result != WrappingTimestampOrderResult::DUPLICATE &&
           result != WrappingTimestampOrderResult::BACKWARD;
  }

  void reset() { order_.reset(); }

 private:
  WrappingTimestampOrder order_;
};

}  // namespace espectre
