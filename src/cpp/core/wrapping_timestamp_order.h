/*
 * ESPectre - Wrapping Timestamp Order
 *
 * Wrap-safe ordering of 32-bit timestamps, shared by temporal CSI admission
 * and the capture-level duplicate filter. Mirrored in
 * tools/lib/temporal_csi_sampler.py.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

/** Half the 32-bit range, in timestamp units: a larger forward delta reads as backward. */
constexpr uint32_t WRAPPING_TIMESTAMP_HALF_RANGE = 0x80000000U;

/** How a timestamp relates to the last accepted one. */
enum class WrappingTimestampOrderResult : uint8_t {
  FIRST = 0U,      ///< No earlier timestamp; this one starts the sequence.
  FORWARD = 1U,    ///< Later than the last accepted one; see delta().
  DUPLICATE = 2U,  ///< Equal to the last accepted one, or to the latest value of a backward run.
  BACKWARD = 3U,   ///< Earlier than the last accepted one; rejected.
  RESYNC = 4U,     ///< Restarts the sequence after a run of advancing backward values.
};

/**
 * Order wrapping 32-bit timestamps without stalling after a long silence.
 *
 * A forward delta of half the range or more reads as backward. A clock that
 * wraps every 2^32 units therefore makes a silence between half and one full
 * range look like a step back, and plain rejection would refuse every later
 * value until the clock caught up. Here, rejected values that keep advancing
 * among themselves for at least the resync span start a new sequence instead.
 * An isolated late value never forms such a run and stays rejected, and a
 * repeated value inside a run is a duplicate that leaves the run intact.
 */
class WrappingTimestampOrder {
 public:
  /**
   * Construct with the span, in timestamp units, that a run of backward
   * values must cover to resync; zero means one unit.
   */
  explicit WrappingTimestampOrder(uint32_t resync_span) { set_resync_span(resync_span); }

  /** Change the resync span; the current sequence is kept. */
  void set_resync_span(uint32_t resync_span) {
    resync_span_ = resync_span > 0U ? resync_span : 1U;
  }
  /** Span a run of backward values must cover to resync. */
  uint32_t resync_span() const { return resync_span_; }

  /** Forget the sequence and any backward run. */
  void reset() {
    has_last_ = false;
    last_ = 0U;
    delta_ = 0U;
    has_run_ = false;
    run_origin_ = 0U;
    run_last_ = 0U;
  }

  /** Classify `value` and accept it unless it is a duplicate or backward. */
  WrappingTimestampOrderResult observe(uint32_t value) {
    delta_ = 0U;
    if (!has_last_) {
      has_last_ = true;
      last_ = value;
      has_run_ = false;
      return WrappingTimestampOrderResult::FIRST;
    }
    const uint32_t delta = value - last_;
    if (delta == 0U) {
      return WrappingTimestampOrderResult::DUPLICATE;
    }
    if (delta < WRAPPING_TIMESTAMP_HALF_RANGE) {
      last_ = value;
      delta_ = delta;
      has_run_ = false;
      return WrappingTimestampOrderResult::FORWARD;
    }
    if (has_run_) {
      const uint32_t run_step = value - run_last_;
      // A repeated value neither advances nor breaks the run.
      if (run_step == 0U) {
        return WrappingTimestampOrderResult::DUPLICATE;
      }
      if (run_step < WRAPPING_TIMESTAMP_HALF_RANGE) {
        run_last_ = value;
        if (value - run_origin_ < resync_span_) {
          return WrappingTimestampOrderResult::BACKWARD;
        }
        last_ = value;
        has_run_ = false;
        return WrappingTimestampOrderResult::RESYNC;
      }
    }
    // Start a run, or restart it when it steps back.
    has_run_ = true;
    run_origin_ = value;
    run_last_ = value;
    return WrappingTimestampOrderResult::BACKWARD;
  }

  /** Whether a value was accepted since the last reset. */
  bool has_last() const { return has_last_; }
  /** Last accepted value; meaningful only when has_last() is true. */
  uint32_t last() const { return last_; }
  /** Forward step of the latest observe() that returned FORWARD; zero otherwise. */
  uint32_t delta() const { return delta_; }

 private:
  uint32_t resync_span_{1000000U};
  bool has_last_{false};
  uint32_t last_{0U};
  uint32_t delta_{0U};
  bool has_run_{false};
  uint32_t run_origin_{0U};
  uint32_t run_last_{0U};
};

}  // namespace espectre
