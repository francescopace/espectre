/*
 * ESPectre - Evaluation Cadence
 *
 * Decides when the detector is due for an evaluation, from the packets' own
 * arrival timestamps rather than from a packet count.
 *
 * Owned in one place because two callers need the same answer: the steady-state
 * detection path, and the startup calibration interceptor. They used to decide
 * separately, and calibration always counted packets while detection counted
 * elapsed time, so on an off-nominal stream the threshold was fitted at a
 * different feature resolution than the one it was applied at.
 *
 * Keep aligned with RuntimeMotionPolicy in
 * src/python/micro_espectre/runtime_policy.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "detector_timing.h"

namespace espectre {

class EvaluationCadence {
 public:
  /** Set the elapsed-time evaluation cadence. */
  void set_interval_ms(uint32_t interval_ms) {
    evaluation_interval_us_ = (interval_ms > 0U ? interval_ms : 1U) * 1000U;
  }

  /** Set the detector window duration used for gap handling and sizing. */
  void set_window_size_ms(uint32_t window_size_ms) {
    window_size_ms_ = window_size_ms > 0U ? window_size_ms : 1U;
    window_duration_us_ = window_size_ms_ * 1000U;
  }

  /**
   * Record one accepted packet and report whether an evaluation is due.
   *
   * The cadence advances on the MAC receive timestamp, not the loop clock. The
   * loop clock measures how fast packets are *processed*, which matches arrival
   * on hardware but not on synchronous replay paths, and it would make the
   * cadence depend on host scheduling. The arrival timestamp is an input, so a
   * caller that supplies it gets the same cadence every run.
   *
   * @param arrival_us MAC receive timestamp; zero is valid after counter wrap
   */
  bool observe(uint32_t arrival_us) {
    packets_since_evaluation_++;
    if (has_last_packet_) {
      const uint32_t delta_us = elapsed_since_timestamp_us(arrival_us, last_packet_us_);
      if (delta_us > 0U && delta_us < window_duration_us_) {
        elapsed_since_evaluation_us_ += delta_us;
      } else if (delta_us >= window_duration_us_) {
        // A hole longer than one window leaves the detector holding stale
        // history, so drop the accumulated coverage rather than counting it.
        elapsed_since_evaluation_us_ = 0U;
      }
    }
    last_packet_us_ = arrival_us;
    has_last_packet_ = true;

    return elapsed_since_evaluation_us_ >= evaluation_interval_us_;
  }

  /** Packets represented by the evaluation that is about to run. */
  uint32_t packets_since_evaluation() const { return packets_since_evaluation_; }

  /** Close the current evaluation window. */
  void after_evaluation() {
    packets_since_evaluation_ = 0U;
    elapsed_since_evaluation_us_ = 0U;
  }

  /** Forget the accumulated coverage. */
  void reset_window() {
    packets_since_evaluation_ = 0U;
    elapsed_since_evaluation_us_ = 0U;
  }

  /** Forget all timing state. */
  void reset() {
    last_packet_us_ = 0U;
    has_last_packet_ = false;
    reset_window();
  }

 private:
  uint32_t window_size_ms_{DETECTOR_WINDOW_SIZE_MS_DEFAULT};
  uint32_t window_duration_us_{DETECTOR_WINDOW_SIZE_MS_DEFAULT * 1000U};
  uint32_t evaluation_interval_us_{EVALUATION_INTERVAL_US};
  uint32_t last_packet_us_{0U};
  bool has_last_packet_{false};
  uint32_t elapsed_since_evaluation_us_{0U};
  uint32_t packets_since_evaluation_{0U};
};

}  // namespace espectre
