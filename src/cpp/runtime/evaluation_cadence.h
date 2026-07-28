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
 * License: GPLv3
 */
#pragma once

#include <cstdint>

#include "detector_limits.h"
#include "detector_timing.h"

namespace espectre {

class EvaluationCadence {
 public:
  /** Set the packet-count fallback used before the rate estimate is trusted. */
  void set_packet_interval(uint32_t interval) {
    packet_interval_ = interval > 0U ? interval : 1U;
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
   * @param arrival_us MAC receive timestamp, or 0 when the source has none
   */
  bool observe(uint32_t arrival_us) {
    packets_since_evaluation_++;
    if (arrival_us != 0U && last_packet_us_ != 0U) {
      const uint32_t delta_us = elapsed_since_timestamp_us(arrival_us, last_packet_us_);
      if (delta_us > 0U && delta_us < SEG_WINDOW_US) {
        packet_rate_.observe_interval(delta_us);
        elapsed_since_evaluation_us_ += delta_us;
      } else if (delta_us >= SEG_WINDOW_US) {
        // A hole longer than one window leaves the detector holding stale
        // history, so drop the accumulated coverage rather than counting it.
        elapsed_since_evaluation_us_ = 0U;
      }
    }
    if (arrival_us != 0U) {
      last_packet_us_ = arrival_us;
    }

    // Elapsed time is authoritative once the stream has shown a plausible
    // cadence. Until then, and on sources that carry no arrival timestamp, the
    // packet counter is the fallback.
    return packet_rate_.ready()
               ? elapsed_since_evaluation_us_ >= EVALUATION_INTERVAL_US
               : packets_since_evaluation_ >= packet_interval_;
  }

  /** Packets represented by the evaluation that is about to run. */
  uint32_t packets_since_evaluation() const { return packets_since_evaluation_; }

  /** Close the current evaluation window. */
  void after_evaluation() {
    packets_since_evaluation_ = 0U;
    elapsed_since_evaluation_us_ = 0U;
  }

  /** True once the measured cadence can be trusted to size the detector. */
  bool rate_ready() const { return packet_rate_.ready(); }

  /** Measured inter-packet interval, or the nominal one until ready. */
  uint32_t interval_us() const { return packet_rate_.interval_us(); }

  /** Forget the accumulated coverage; keeps the rate estimate. */
  void reset_window() {
    packets_since_evaluation_ = 0U;
    elapsed_since_evaluation_us_ = 0U;
  }

  /** Forget everything, including the measured cadence. */
  void reset() {
    packet_rate_.reset();
    last_packet_us_ = 0U;
    reset_window();
  }

 private:
  // Nominal-rate fallback, resolved from the same duration contract the
  // time-based path uses, so the two agree at 100 pps by construction.
  static constexpr uint32_t kNominalPacketInterval = packets_for_duration(
      EVALUATION_INTERVAL_US, nominal_packet_interval_us(DETECTOR_DEFAULT_WINDOW_SIZE), 1U);

  PacketRateEstimator packet_rate_{};
  uint32_t packet_interval_{kNominalPacketInterval};
  uint32_t last_packet_us_{0U};
  uint32_t elapsed_since_evaluation_us_{0U};
  uint32_t packets_since_evaluation_{0U};
};

}  // namespace espectre
