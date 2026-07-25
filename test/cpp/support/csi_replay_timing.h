#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "detector_limits.h"
#include "detector_timing.h"

namespace espectre {
namespace csi_replay_timing {

constexpr float GAP_RESET_RATIO = 4.0f;
constexpr uint32_t GAP_RESET_MIN_US = 250000U;
constexpr uint32_t GAP_RESET_SEQ_THRESHOLD = 3U;

inline uint32_t nominal_packet_interval_us(uint16_t window_packets) {
  const uint32_t packets = std::max<uint32_t>(1U, static_cast<uint32_t>(window_packets));
  return std::max<uint32_t>(1U, static_cast<uint32_t>(std::llround(1000000.0 / static_cast<double>(packets))));
}

inline uint16_t equivalent_packet_weight(uint64_t elapsed_us,
                                         uint32_t nominal_interval_us,
                                         uint16_t fallback_packets) {
  const uint32_t nominal = std::max<uint32_t>(1U, nominal_interval_us);
  if (elapsed_us == 0U) {
    return std::max<uint16_t>(1U, fallback_packets);
  }
  const uint64_t rounded = (elapsed_us + (nominal / 2U)) / nominal;
  return static_cast<uint16_t>(std::max<uint64_t>(1U, std::min<uint64_t>(rounded, UINT16_MAX)));
}

struct TimingObservation {
  uint32_t delta_us{0U};
  uint32_t coverage_us{0U};
  uint32_t missing_seq{0U};
  bool contaminated{false};
  /// True when delta_us was inferred from the cadence rather than measured.
  bool from_nominal{false};
};

class PacketTimingTracker {
 public:
  explicit PacketTimingTracker(uint32_t nominal_interval_us)
      : nominal_interval_us_(std::max<uint32_t>(1U, nominal_interval_us)),
        rate_(nominal_interval_us_) {}

  void reset() {
    // A gap says nothing about the stream's sustained cadence. Keeping the
    // learned rate avoids re-reading a slower but healthy stream as one long
    // sequence of losses after the first reset.
    has_last_seq_ = false;
    has_last_device_ticks_ = false;
    has_last_wifi_rx_ts_ = false;
    last_seq_num_ = 0U;
    last_device_ticks_us_ = 0U;
    last_wifi_rx_ts_us_ = 0U;
  }

  TimingObservation observe(uint32_t seq_num,
                            bool has_seq,
                            uint64_t device_ticks_us,
                            bool has_device_ticks,
                            uint32_t wifi_rx_ts_us,
                            bool has_wifi_rx_ts) {
    TimingObservation observation{};

    uint32_t seq_step = 0U;
    if (has_seq && has_last_seq_) {
      seq_step = seq_num - last_seq_num_;
      if (seq_step > 0U && seq_step < 0x80000000U) {
        rate_.observe_sequence_step(seq_step);
        const uint32_t baseline_step = rate_.sequence_step();
        observation.missing_seq =
            seq_step > baseline_step ? (seq_step - baseline_step) : 0U;
      } else {
        seq_step = 0U;
      }
    }

    if (has_device_ticks && has_last_device_ticks_ && device_ticks_us >= last_device_ticks_us_) {
      observation.delta_us = static_cast<uint32_t>(device_ticks_us - last_device_ticks_us_);
    } else if (has_wifi_rx_ts && has_last_wifi_rx_ts_) {
      const uint32_t candidate = elapsed_since_timestamp_us(
          wifi_rx_ts_us, last_wifi_rx_ts_us_);
      if (candidate > 0U) {
        observation.delta_us = candidate;
      }
    }

    if (observation.delta_us == 0U) {
      // No usable predecessor, which happens on the first packet and on the
      // one re-observed right after a reset. The question here is how far
      // this packet sits from the one before it, so the estimate is the
      // typical spacing (median), not the mean throughput. Python's
      // PacketRateEstimator.interval_us is that same median; using the mean
      // here made the two runtimes disagree by roughly a millisecond per
      // contamination, which then moved every later evaluation boundary.
      observation.delta_us =
          rate_.typical_interval_us() * std::max<uint32_t>(1U, observation.missing_seq + 1U);
      observation.from_nominal = true;
    } else {
      rate_.observe_interval(observation.delta_us);
    }

    const uint32_t contamination_delta_us =
        std::max<uint32_t>(GAP_RESET_MIN_US,
                           static_cast<uint32_t>(std::llround(
                               static_cast<double>(rate_.typical_interval_us()) * GAP_RESET_RATIO)));
    const bool has_prior_timing = has_last_seq_ || has_last_device_ticks_ || has_last_wifi_rx_ts_;
    observation.contaminated =
        (has_seq && has_last_seq_ && rate_.sequence_established() &&
         observation.missing_seq >= GAP_RESET_SEQ_THRESHOLD) ||
        (has_prior_timing && observation.delta_us >= contamination_delta_us);
    observation.coverage_us = observation.contaminated ? 0U : observation.delta_us;

    if (has_seq) {
      last_seq_num_ = seq_num;
      has_last_seq_ = true;
    }
    if (has_device_ticks) {
      last_device_ticks_us_ = device_ticks_us;
      has_last_device_ticks_ = true;
    }
    if (has_wifi_rx_ts) {
      last_wifi_rx_ts_us_ = wifi_rx_ts_us;
      has_last_wifi_rx_ts_ = true;
    }
    return observation;
  }

 private:
  uint32_t nominal_interval_us_;
  PacketRateEstimator rate_;
  bool has_last_seq_{false};
  bool has_last_device_ticks_{false};
  bool has_last_wifi_rx_ts_{false};
  uint32_t last_seq_num_{0U};
  uint64_t last_device_ticks_us_{0U};
  uint32_t last_wifi_rx_ts_us_{0U};
};

/// Sample budget shared with measure_packet_interval_us in dataset_metadata.py.
constexpr int MEASURE_INTERVAL_SAMPLES = 4096;

/**
 * Effective packet interval of a whole capture, in microseconds.
 *
 * Host-side counterpart of the runtime estimator, and the C++ mirror of
 * measure_packet_interval_us in dataset_metadata.py: every packet is observed
 * so the tracker stays in step, but only every stride-th delta is averaged,
 * and deltas inferred from the cadence rather than measured are excluded. The
 * mean is deliberate, because sizing a window is a throughput question.
 */
template <typename SeqAt, typename TicksAt, typename RxTsAt>
uint32_t measure_packet_interval_us(int packet_count,
                                    SeqAt seq_at,
                                    TicksAt ticks_at,
                                    RxTsAt rx_ts_at,
                                    uint32_t nominal_interval_us) {
  if (packet_count < 2) {
    return nominal_interval_us;
  }
  PacketTimingTracker tracker(nominal_interval_us);
  const int stride = std::max(1, packet_count / MEASURE_INTERVAL_SAMPLES);
  uint64_t total_us = 0U;
  int counted = 0;
  for (int i = 0; i < packet_count; i++) {
    uint32_t seq = 0U;
    uint64_t ticks = 0U;
    uint32_t rx_ts = 0U;
    const bool has_seq = seq_at(i, seq);
    const bool has_ticks = ticks_at(i, ticks);
    const bool has_rx_ts = rx_ts_at(i, rx_ts);
    const TimingObservation timing =
        tracker.observe(seq, has_seq, ticks, has_ticks, rx_ts, has_rx_ts);
    if (i == 0 || (i % stride) != 0) {
      continue;
    }
    if (timing.from_nominal || timing.contaminated) {
      continue;
    }
    total_us += timing.delta_us;
    counted++;
  }
  if (counted <= 0) {
    return nominal_interval_us;
  }
  return std::max<uint32_t>(1U, static_cast<uint32_t>(std::llround(
      static_cast<double>(total_us) / static_cast<double>(counted))));
}

class TimeAwareCadence {
 public:
  TimeAwareCadence(uint16_t window_packets, uint32_t evaluation_interval_packets)
      : nominal_interval_us_(nominal_packet_interval_us(window_packets)),
        evaluation_interval_packets_(std::max<uint32_t>(1U, evaluation_interval_packets)),
        evaluation_interval_us_(EVALUATION_INTERVAL_US) {}

  void reset() {
    packets_since_evaluation_ = 0U;
    elapsed_us_since_evaluation_ = 0U;
  }

  void note_packet(uint32_t elapsed_us) {
    packets_since_evaluation_++;
    elapsed_us_since_evaluation_ += elapsed_us;
  }

  /**
   * Mirrors RuntimeMotionPolicy.should_evaluate in runtime_policy.py.
   *
   * Contaminated packets contribute no coverage, so a stream that has just
   * been reset accumulates no elapsed time. Without the packet-count fallback
   * the replay would stall until clean time rebuilt, evaluating fewer windows
   * than the Python replay and than the production pipeline, which keeps the
   * same fallback for its own warmup.
   */
  bool should_evaluate(bool should_publish = false) const {
    if (should_publish) {
      return true;
    }
    if (elapsed_us_since_evaluation_ > 0U) {
      return elapsed_us_since_evaluation_ >= evaluation_interval_us_;
    }
    return packets_since_evaluation_ >= evaluation_interval_packets_;
  }

  uint16_t packet_weight() const {
    return equivalent_packet_weight(
        elapsed_us_since_evaluation_,
        nominal_interval_us_,
        static_cast<uint16_t>(std::min<uint32_t>(packets_since_evaluation_, UINT16_MAX)));
  }

  void after_evaluation() { reset(); }

 private:
  uint32_t nominal_interval_us_{0U};
  uint32_t evaluation_interval_packets_{0U};
  uint32_t evaluation_interval_us_{0U};
  uint32_t packets_since_evaluation_{0U};
  uint64_t elapsed_us_since_evaluation_{0U};
};

}  // namespace csi_replay_timing
}  // namespace espectre
