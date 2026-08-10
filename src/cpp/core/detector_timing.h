/*
 * ESPectre - Detector Timing
 *
 * Effective packet-cadence estimation and the shared duration-to-packet-count
 * contract used by the detectors. Counterpart of the same helpers in
 * src/python/micro_espectre/runtime_policy.py; keep the two aligned, because a
 * detector fitted under one resolution and run under another is measuring a
 * different feature.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <cstdlib>

#include "detector_limits.h"

namespace espectre {

// One second of packets at the nominal rate is enough to be stable without
// making the estimate slow to follow a genuine rate change.
constexpr uint8_t RATE_ESTIMATOR_SAMPLES = 64U;
// Samples required before a rate-derived rule may fire. Until then the
// estimator has not seen enough of the stream to tell a slower cadence from
// packet loss, and guessing either way is worse than not acting.
constexpr uint8_t RATE_ESTIMATOR_WARMUP = 16U;
// Medians are refreshed on this stride rather than on every packet: sorting in
// the hot path would cost more than the estimate can move in that time.
constexpr uint8_t RATE_ESTIMATOR_REFRESH_STRIDE = 16U;

/**
 * Resolved detector timing for one measured cadence.
 */
struct DetectorTiming {
  uint32_t interval_us;
  uint16_t window_packets;
  uint16_t lag;
  uint16_t autocorr_lag;
};

constexpr uint32_t nominal_packet_interval_us(uint16_t window_packets) {
  return window_packets > 0U ? (1000000U / window_packets) : 1000000U;
}

/**
 * Elapsed microseconds between two 32-bit arrival timestamps.
 *
 * The MAC receive timestamp wraps roughly every 71.6 minutes, so the delta is
 * taken modulo the counter width. A result past half the range is a counter
 * that went backwards rather than a very long gap, and is reported as zero so
 * callers ignore it instead of inventing an hour of coverage.
 */
inline uint32_t elapsed_since_timestamp_us(uint32_t now_us, uint32_t previous_us) {
  const uint32_t delta = now_us - previous_us;
  return delta < 0x80000000U ? delta : 0U;
}

constexpr uint32_t packets_for_duration(uint32_t duration_us, uint32_t interval_us,
                                        uint32_t minimum) {
  const uint32_t interval = interval_us > 0U ? interval_us : 1U;
  const uint32_t packets = (duration_us + interval / 2U) / interval;
  return packets < minimum ? minimum : packets;
}

/**
 * Resolve the detector's duration contract for replay analysis.
 *
 * Deployed runtimes deliberately keep the production lags at their nominal
 * packet offsets. Scaling only the numerator of the L1 lag ratio changes the
 * feature definition and regresses low-rate recall; scaling both offsets would
 * require a Classic refit and an ML retrain. The host replay path keeps this
 * helper so future timing candidates can be measured against shipped behavior.
 *
 * The lags describe how far the channel has moved over an interval, so they
 * track that interval. This is decisive at high rates: on the 1000 pps
 * diagnostic capture, restoring the turbulence-autocorrelation lag to its 10 ms
 * scale takes false positives from 17.8-32.7% to 0.0%, because at under a
 * millisecond of separation consecutive packets are almost perfectly correlated
 * and the feature leaves the range its coefficients were fitted over.
 *
 * The window is different. Its features are estimator averages, so their
 * sampling behaviour depends on how many samples they average, not on the time
 * those samples span. Holding a one-second span at 25 pps leaves 25 samples,
 * the estimates get noisier, startup calibration answers the wider quiet
 * distribution by lifting the threshold, and recall collapses while false
 * positives stay low.
 */
/**
 * @param window_override Window in packets to use instead of the rate-derived
 *        one, or 0 to derive it. The window is a configured sample count, not
 *        a duration: measurement puts the floor at 100 samples, below which the
 *        worst paired sessions fall under the production recall target (91.9%
 *        at 80 samples, 94.2% at 90, 96.8% at 100). Replay analyses can pass a
 *        configured window here when isolating lag behavior, or leave it 0 to
 *        evaluate the fully derived candidate timing.
 */
inline DetectorTiming derive_detector_timing(uint32_t interval_us,
                                             uint16_t window_override = 0U) {
  uint32_t interval = interval_us > 0U ? interval_us : 1U;

  // Near the nominal cadence, adapting buys nothing and costs homogeneity:
  // rounding a duration into packets flips between neighbouring counts across
  // streams that all run at essentially the nominal rate, so one coefficient
  // set has to cover slightly different feature definitions.
  constexpr uint32_t kNominal =
      nominal_packet_interval_us(DETECTOR_DEFAULT_WINDOW_SIZE);
  const uint32_t deviation =
      interval > kNominal ? interval - kNominal : kNominal - interval;
  if (static_cast<float>(deviation) <= RATE_ADAPTATION_DEAD_BAND * kNominal) {
    interval = kNominal;
  }

  uint32_t window = window_override > 0U
                        ? static_cast<uint32_t>(window_override)
                        : packets_for_duration(SEG_WINDOW_US, interval,
                                               DETECTOR_MIN_WINDOW_SIZE);
  if (window < DETECTOR_MIN_WINDOW_SIZE) {
    window = DETECTOR_MIN_WINDOW_SIZE;
  }
  if (window > DETECTOR_MAX_WINDOW_SIZE) {
    window = DETECTOR_MAX_WINDOW_SIZE;
  }

  // Both lags must leave a usable series inside the window.
  uint32_t lag_ceiling = window / 2U > 0U ? window / 2U : 1U;
  if (lag_ceiling > L1_DELTA_LAG_MAX) {
    lag_ceiling = L1_DELTA_LAG_MAX;
  }
  uint32_t lag = packets_for_duration(L1_DELTA_LAG_US, interval, 1U);
  uint32_t autocorr_lag = packets_for_duration(TURB_AUTOCORR_LAG_US, interval, 1U);

  DetectorTiming timing{};
  timing.interval_us = interval;
  timing.window_packets = static_cast<uint16_t>(window);
  timing.lag = static_cast<uint16_t>(lag < lag_ceiling ? lag : lag_ceiling);
  timing.autocorr_lag =
      static_cast<uint16_t>(autocorr_lag < lag_ceiling ? autocorr_lag : lag_ceiling);
  return timing;
}

/**
 * Track the effective packet cadence from observed inter-packet deltas.
 *
 * The estimate is a rolling median rather than a mean: a stream with holes
 * contains a few very large deltas, and a mean would let those dominate the
 * cadence the rest of the pipeline derives its windows from.
 */
class PacketRateEstimator {
 public:
  explicit PacketRateEstimator(
      uint32_t nominal_interval_us =
          nominal_packet_interval_us(DETECTOR_DEFAULT_WINDOW_SIZE))
      : nominal_interval_us_(nominal_interval_us > 0U ? nominal_interval_us : 1U) {
    reset();
  }

  /** Forget the observed cadence and fall back to the nominal interval. */
  void reset() {
    delta_count_ = 0U;
    delta_index_ = 0U;
    seq_count_ = 0U;
    seq_index_ = 0U;
    since_refresh_ = 0U;
    interval_cache_ = 0U;
    seq_cache_ = 0U;
    typical_cache_ = 0U;
  }

  /** Record one inter-packet interval. */
  void observe_interval(uint32_t delta_us) {
    if (delta_us == 0U) {
      return;
    }
    deltas_[delta_index_] = delta_us;
    delta_index_ = static_cast<uint8_t>((delta_index_ + 1U) % RATE_ESTIMATOR_SAMPLES);
    if (delta_count_ < RATE_ESTIMATOR_SAMPLES) {
      delta_count_++;
    }
    if (++since_refresh_ >= RATE_ESTIMATOR_REFRESH_STRIDE) {
      since_refresh_ = 0U;
      interval_cache_ = 0U;
      seq_cache_ = 0U;
      typical_cache_ = 0U;
    }
  }

  /** Record one observed advance of the packet sequence counter. */
  void observe_sequence_step(uint32_t seq_step) {
    if (seq_step == 0U) {
      return;
    }
    seq_steps_[seq_index_] = seq_step;
    seq_index_ = static_cast<uint8_t>((seq_index_ + 1U) % RATE_ESTIMATOR_SAMPLES);
    if (seq_count_ < RATE_ESTIMATOR_SAMPLES) {
      seq_count_++;
    }
  }

  /** True once enough intervals have been seen to trust the estimate. */
  bool ready() const {
    if (delta_count_ < RATE_ESTIMATOR_WARMUP) {
      return false;
    }
    return interval_us_raw_() >= MIN_PLAUSIBLE_PACKET_INTERVAL_US;
  }

  /**
   * Median inter-packet spacing, used to judge whether a gap is a hole.
   *
   * Mirrors the Python PacketRateEstimator.interval_us, including its warmup
   * gate: until the estimator has seen enough of the stream it cannot tell a
   * slower cadence from packet loss, so a rate-derived rule stays on the
   * nominal interval rather than acting on a guess.
   */
  uint32_t typical_interval_us() const {
    if (delta_count_ < RATE_ESTIMATOR_WARMUP) {
      return nominal_interval_us_;
    }
    if (typical_cache_ == 0U) {
      typical_cache_ = median(deltas_, delta_count_);
    }
    return typical_cache_ > 0U ? typical_cache_ : nominal_interval_us_;
  }

  /** True once the stream's own sequence step has been observed enough. */
  bool sequence_established() const { return seq_count_ >= RATE_ESTIMATOR_WARMUP; }

  /** Return the effective packet interval, or the nominal one until ready. */
  uint32_t interval_us() const {
    if (!ready()) {
      return nominal_interval_us_;
    }
    return interval_us_raw_();
  }

  /**
   * Return the cadence-normal sequence advance, or 1 until established.
   *
   * A stream that natively runs slower than the nominal rate advances its
   * sequence counter by more than one per delivered packet. That is the
   * stream's own step, not loss, so loss has to be measured against it.
   */
  uint32_t sequence_step() const {
    if (!sequence_established()) {
      return 1U;
    }
    if (seq_cache_ == 0U) {
      seq_cache_ = median(seq_steps_, seq_count_);
      if (seq_cache_ == 0U) {
        seq_cache_ = 1U;
      }
    }
    return seq_cache_;
  }

 private:
  /** Mean of the observed intervals, which is throughput rather than spacing. */
  uint32_t interval_us_raw_() const {
    if (interval_cache_ == 0U) {
      uint64_t total = 0U;
      for (uint8_t i = 0U; i < delta_count_; i++) {
        total += deltas_[i];
      }
      interval_cache_ = static_cast<uint32_t>(total / delta_count_);
      if (interval_cache_ == 0U) {
        interval_cache_ = 1U;
      }
    }
    return interval_cache_;
  }

  static uint32_t median(const uint32_t* values, uint8_t count) {
    if (count == 0U) {
      return 0U;
    }
    uint32_t scratch[RATE_ESTIMATOR_SAMPLES];
    for (uint8_t i = 0U; i < count; i++) {
      scratch[i] = values[i];
    }
    // Insertion sort: the buffer is small and nearly sorted in practice, and it
    // avoids pulling a general sort into the firmware image.
    for (uint8_t i = 1U; i < count; i++) {
      const uint32_t key = scratch[i];
      int16_t j = static_cast<int16_t>(i) - 1;
      while (j >= 0 && scratch[j] > key) {
        scratch[j + 1] = scratch[j];
        j--;
      }
      scratch[j + 1] = key;
    }
    const uint8_t middle = static_cast<uint8_t>(count / 2U);
    if (count % 2U) {
      return scratch[middle];
    }
    return (scratch[middle - 1U] + scratch[middle] + 1U) / 2U;
  }

  uint32_t nominal_interval_us_;
  uint32_t deltas_[RATE_ESTIMATOR_SAMPLES]{};
  uint32_t seq_steps_[RATE_ESTIMATOR_SAMPLES]{};
  uint8_t delta_count_{0U};
  uint8_t delta_index_{0U};
  uint8_t seq_count_{0U};
  uint8_t seq_index_{0U};
  uint8_t since_refresh_{0U};
  mutable uint32_t interval_cache_{0U};
  mutable uint32_t seq_cache_{0U};
  mutable uint32_t typical_cache_{0U};
};

}  // namespace espectre
