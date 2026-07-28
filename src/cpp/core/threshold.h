/*
 * ESPectre - Adaptive Threshold Calculator
 *
 * Calculates adaptive threshold from calibration baseline values.
 * Called after calibration to compute the detection threshold.
 *
 * Startup threshold calibration is automatic. Detectors may apply their own
 * session adaptation to the shared calibration metric.
 *
 * The default Classic path is motion-first with an internal quiet-first
 * fallback. Successful motion-first calibration can finish before the nominal
 * budget; otherwise the calibrator falls back to the quiet-first gate on the
 * same observed metrics and still completes within the configured packet
 * budget. Keep the semantics aligned with src/python/micro_espectre/threshold.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace espectre {

// =============================================================================
// Segmentation Threshold Constants
// =============================================================================

// Note: Window size constants are defined in base_detector.h:

constexpr float SEGMENTATION_DEFAULT_THRESHOLD = 1.0f;
// Min threshold lowered to support CV normalization (std/mean produces smaller values)
constexpr float SEGMENTATION_MIN_THRESHOLD = 1e-9f;
constexpr float SEGMENTATION_MAX_THRESHOLD = 10.0f;

/**
 * Validate threshold value against finite/range constraints.
 */
inline bool is_valid_threshold(float threshold, float min_threshold, float max_threshold) {
    return std::isfinite(threshold) &&
           threshold >= min_threshold &&
           threshold <= max_threshold;
}

/**
 * Clamp threshold to [min, max] and recover non-finite values.
 */
inline float clamp_threshold(float threshold, float min_threshold, float max_threshold) {
    if (!std::isfinite(threshold)) {
        return min_threshold;
    }
    if (threshold < min_threshold) {
        return min_threshold;
    }
    if (threshold > max_threshold) {
        return max_threshold;
    }
    return threshold;
}

// Default startup multiplier for detectors that use the shared metric.
constexpr float DEFAULT_ADAPTIVE_FACTOR = 1.3f;

// Startup calibration consistency gate (benchmark-tuned on the paired
// datasets; keep aligned with src/python/micro_espectre/threshold.py).
constexpr uint8_t STARTUP_GATE_CHUNKS = 6;
constexpr float STARTUP_GATE_SPREAD_RATIO = 1.10f;
constexpr float STARTUP_GATE_ANCHOR_RATIO = 1.5f;
constexpr uint8_t STARTUP_MOTION_CHUNK_SIZE = 25;
constexpr uint8_t STARTUP_MOTION_MIN_QUIET_CHUNKS = 2;
constexpr uint8_t STARTUP_MOTION_CONFIRM_CHUNKS = 2;
constexpr uint8_t STARTUP_POST_MOTION_QUIET_CHUNKS = 2;
constexpr float STARTUP_QUIET_STABILITY_RATIO = 1.20f;
constexpr float STARTUP_MOTION_TRIGGER_RATIO = 1.80f;
constexpr float STARTUP_QUIET_RETURN_RATIO = 1.25f;
constexpr float STARTUP_MOTION_GAP_RATIO = 1.35f;
constexpr float STARTUP_NO_MOTION_FALLBACK_MARGIN = 1.03f;
constexpr uint8_t STARTUP_MOTION_MAX_LEVELS = 40;

/**
 * Startup threshold calibrator with a motion-first primary path and an
 * internal quiet-first fallback. The fallback keeps the existing gated ring of
 * chunk maxima, but completion never exceeds the configured budget.
 */
class StartupThresholdCalibrator {
 public:
  void begin(uint16_t target_packets, bool gate_enabled) {
    target_packets_ = target_packets > 0 ? target_packets : 1;
    gate_enabled_ = gate_enabled;
    packet_count_ = 0;
    ready_packet_count_ = 0;
    has_value_ = false;
    max_motion_metric_ = 0.0f;
    gate_accepted_ = false;
    chunk_size_ = 0;
    chunk_count_ = 0;
    chunk_max_ = 0.0f;
    ring_count_ = 0;
    ring_next_ = 0;
    min_chunk_max_ = 0.0f;
    discarded_chunk_max_ = 0.0f;
    has_discarded_chunk_ = false;
    motion_chunk_sum_ = 0.0f;
    motion_chunk_max_ = 0.0f;
    motion_chunk_count_ = 0;
    bootstrap_count_ = 0;
    quiet_level_count_ = 0;
    motion_level_count_ = 0;
    post_quiet_level_count_ = 0;
    quiet_anchor_ready_ = false;
    motion_confirmed_ = false;
    motion_accepted_ = false;
    phase_ = Phase::SEEK_MOTION;
    consecutive_motion_chunks_ = 0;
    consecutive_post_quiet_chunks_ = 0;
  }

  /**
   * Consume one evaluated detector step representing one or more packets.
   *
   * @param detector_ready Whether the detector metric window is full
   * @param motion_metric Current detector motion metric
   * @param packet_weight Number of processed packets represented by the metric
   */
  void observe(bool detector_ready, float motion_metric, uint16_t packet_weight = 1U) {
    // The weight is folded in one step rather than replayed packet by packet.
    // Replaying it re-evaluated the chunk guards on every repetition, so a
    // chunk boundary landing inside one evaluation split differently here than
    // in the Python calibrator and the two produced different thresholds. See
    // threshold.py, which this mirrors.
    const uint32_t remaining = target_packets_ > packet_count_ ? target_packets_ - packet_count_ : 0U;
    const uint32_t weight = std::min<uint32_t>(std::max<uint16_t>(packet_weight, 1U), remaining);
    if (weight == 0U) {
      return;
    }
    const uint32_t initial_remaining = remaining;
    packet_count_ += weight;
    if (!detector_ready) {
      return;
    }
    ready_packet_count_ += weight;
    if (!has_value_ || motion_metric > max_motion_metric_) {
      max_motion_metric_ = motion_metric;
    }
    has_value_ = true;
    if (gate_enabled_ && !gate_accepted_) {
      observe_gate_metric_(motion_metric, weight, initial_remaining);
    }
    if (gate_enabled_ && !motion_accepted_ && packet_count_ <= target_packets_) {
      observe_motion_chunk_(motion_metric, weight);
    }
    if (gate_enabled_ && !gate_accepted_ && packet_count_ >= target_packets_ &&
        chunk_count_ > 0 && ring_count_ < STARTUP_GATE_CHUNKS) {
      close_gate_chunk_();
    }
  }

  /// True once motion-first succeeds early or the startup budget is spent.
  bool is_complete() const {
    return motion_accepted_ || packet_count_ >= target_packets_;
  }

  bool is_successful() const { return motion_accepted_ || has_value_; }
  bool gate_accepted() const { return gate_accepted_; }
  uint32_t packet_count() const { return packet_count_; }
  uint16_t target_packets() const { return target_packets_; }
  uint32_t ready_packet_count() const { return ready_packet_count_; }

  /// Metric the threshold formula (x factor) is applied to.
  float threshold_metric() const {
    if (motion_accepted_) {
      return motion_threshold_metric_();
    }
    if (!gate_enabled_ || ring_count_ == 0) {
      return has_value_ ? max_motion_metric_ : 0.0f;
    }
    if (gate_accepted_) {
      float metric = ring_max_();
      if (has_discarded_chunk_ &&
          discarded_chunk_max_ <= STARTUP_GATE_ANCHOR_RATIO * ring_median_()) {
        metric = std::max(metric, discarded_chunk_max_);
      }
      return metric;
    }
    if (quiet_anchor_ready_ && motion_level_count_ > 0) {
      const float quiet_ceiling = quiet_ceiling_();
      const float anchored_cap = STARTUP_GATE_ANCHOR_RATIO * quiet_ceiling;
      return std::max(quiet_ceiling, std::min(ring_median_(), anchored_cap));
    }
    if (!motion_confirmed_) {
      return STARTUP_NO_MOTION_FALLBACK_MARGIN * ring_max_();
    }
    return ring_median_();
  }

  /// Statistic name for threshold logging.
  const char* statistic_name() const {
    if (motion_accepted_) {
      return "motion gap midpoint";
    }
    if (!gate_enabled_ || ring_count_ == 0) {
      return "max";
    }
    if (gate_accepted_ || !motion_confirmed_) {
      if (!gate_accepted_ && quiet_anchor_ready_ && motion_level_count_ > 0) {
        return "quiet anchor";
      }
      return "gated max";
    }
    if (quiet_anchor_ready_ && motion_level_count_ > 0) {
      return "quiet anchor";
    }
    return "gated median";
  }

  const char* phase_label() const {
    if (motion_accepted_) {
      return "COMPLETE";
    }
    if (packet_count_ >= target_packets_) {
      return "FALLBACK";
    }
    switch (phase_) {
      case Phase::SEEK_MOTION:
        return "SEEK_MOTION";
      case Phase::SEEK_POST_MOTION_QUIET:
        return "SEEK_POST_QUIET";
      default:
        return "CALIBRATING";
    }
  }

 private:
  enum class Phase {
    SEEK_MOTION,
    SEEK_POST_MOTION_QUIET,
    COMPLETE,
  };

  /**
   * @param initial_remaining Budget left before this observation was counted,
   *        which is what sizes the chunks. Measuring it after the weight was
   *        added would shrink every chunk by the weight.
   */
  void observe_gate_metric_(float metric, uint32_t weight, uint32_t initial_remaining) {
    if (chunk_size_ == 0) {
      chunk_size_ = std::max<uint32_t>(1U, initial_remaining / STARTUP_GATE_CHUNKS);
    }

    uint32_t remaining_weight = weight;
    while (remaining_weight > 0U && !gate_accepted_) {
      if (chunk_count_ == 0 || metric > chunk_max_) {
        chunk_max_ = metric;
      }
      const uint32_t available = chunk_size_ > chunk_count_ ? chunk_size_ - chunk_count_ : 0U;
      const uint32_t take = std::min(remaining_weight, available);
      chunk_count_ += take;
      remaining_weight -= take;
      if (chunk_count_ >= chunk_size_) {
        close_gate_chunk_();
      }
    }
  }

  void close_gate_chunk_() {
    if (ring_count_ == STARTUP_GATE_CHUNKS) {
      const float discarded = ring_[ring_next_];
      if (!has_discarded_chunk_ || discarded > discarded_chunk_max_) {
        discarded_chunk_max_ = discarded;
        has_discarded_chunk_ = true;
      }
    }
    ring_[ring_next_] = chunk_max_;
    ring_next_ = (ring_next_ + 1) % STARTUP_GATE_CHUNKS;
    if (ring_count_ < STARTUP_GATE_CHUNKS) {
      ring_count_++;
    }
    if (ring_count_ == 1 || chunk_max_ < min_chunk_max_) {
      min_chunk_max_ = chunk_max_;
    }
    chunk_count_ = 0;
    chunk_max_ = 0.0f;

    if (ring_count_ >= STARTUP_GATE_CHUNKS && gate_ok_()) {
      gate_accepted_ = true;
    }
  }

  void observe_motion_chunk_(float metric, uint32_t weight) {
    uint32_t remaining_weight = weight;
    while (remaining_weight > 0U && !motion_accepted_) {
      if (motion_chunk_count_ == 0 || metric > motion_chunk_max_) {
        motion_chunk_max_ = metric;
      }
      const uint32_t available = STARTUP_MOTION_CHUNK_SIZE > motion_chunk_count_
                                     ? STARTUP_MOTION_CHUNK_SIZE - motion_chunk_count_
                                     : 0U;
      const uint32_t take = std::min(remaining_weight, available);
      motion_chunk_sum_ += metric * static_cast<float>(take);
      motion_chunk_count_ = static_cast<uint8_t>(motion_chunk_count_ + take);
      remaining_weight -= take;
      if (motion_chunk_count_ < STARTUP_MOTION_CHUNK_SIZE) {
        continue;
      }

      const float level = motion_chunk_sum_ / static_cast<float>(motion_chunk_count_);
      const float peak = motion_chunk_max_;
      consume_closed_motion_chunk_(level, peak);
      motion_chunk_sum_ = 0.0f;
      motion_chunk_max_ = 0.0f;
      motion_chunk_count_ = 0;
    }
  }

  void consume_closed_motion_chunk_(float level, float peak) {
    if (!quiet_anchor_ready_) {
      if (bootstrap_count_ == STARTUP_MOTION_MIN_QUIET_CHUNKS) {
        bootstrap_levels_[0] = bootstrap_levels_[1];
        bootstrap_count_--;
      }
      bootstrap_levels_[bootstrap_count_] = level;
      bootstrap_count_++;

      if (bootstrap_count_ >= STARTUP_MOTION_MIN_QUIET_CHUNKS &&
          levels_are_stable_(bootstrap_levels_, bootstrap_count_)) {
        quiet_anchor_ready_ = true;
        quiet_level_count_ = bootstrap_count_;
        for (uint8_t i = 0; i < bootstrap_count_; i++) {
          quiet_levels_[i] = bootstrap_levels_[i];
        }
      }
      return;
    }

    const float quiet_ref = std::max(quiet_reference_(), 1e-9f);
    const float motion_ratio = level / quiet_ref;
    const float peak_ratio = peak / quiet_ref;

    if (!motion_confirmed_) {
      if (motion_ratio >= STARTUP_MOTION_TRIGGER_RATIO &&
          peak_ratio >= STARTUP_MOTION_TRIGGER_RATIO) {
        append_motion_level_(level);
        consecutive_motion_chunks_++;
        if (consecutive_motion_chunks_ >= STARTUP_MOTION_CONFIRM_CHUNKS) {
          motion_confirmed_ = true;
          phase_ = Phase::SEEK_POST_MOTION_QUIET;
          consecutive_post_quiet_chunks_ = 0;
          post_quiet_level_count_ = 0;
        }
        return;
      }

      if (motion_ratio <= STARTUP_QUIET_RETURN_RATIO) {
        append_quiet_level_(level);
      }
      consecutive_motion_chunks_ = 0;
      return;
    }

    if (motion_ratio <= STARTUP_QUIET_RETURN_RATIO) {
      append_post_quiet_level_(level);
      consecutive_post_quiet_chunks_++;
      if (consecutive_post_quiet_chunks_ >= STARTUP_POST_MOTION_QUIET_CHUNKS &&
          motion_gap_ok_()) {
        motion_accepted_ = true;
        phase_ = Phase::COMPLETE;
      }
      return;
    }

    if (motion_ratio >= STARTUP_MOTION_TRIGGER_RATIO &&
        peak_ratio >= STARTUP_MOTION_TRIGGER_RATIO) {
      append_motion_level_(level);
      consecutive_post_quiet_chunks_ = 0;
      phase_ = Phase::SEEK_POST_MOTION_QUIET;
      return;
    }

    consecutive_post_quiet_chunks_ = 0;
  }

  bool levels_are_stable_(const float* values, uint8_t count) const {
    if (count == 0) {
      return false;
    }
    float low = values[0];
    float high = values[0];
    for (uint8_t i = 1; i < count; i++) {
      low = std::min(low, values[i]);
      high = std::max(high, values[i]);
    }
    if (low <= 0.0f) {
      return high <= 1e-9f;
    }
    return high <= STARTUP_QUIET_STABILITY_RATIO * low;
  }

  float quiet_reference_() const {
    if (quiet_level_count_ == 0) {
      return 0.0f;
    }
    float ordered[STARTUP_GATE_CHUNKS];
    std::copy(quiet_levels_, quiet_levels_ + quiet_level_count_, ordered);
    std::sort(ordered, ordered + quiet_level_count_);
    if (quiet_level_count_ % 2 != 0) {
      return ordered[quiet_level_count_ / 2];
    }
    return 0.5f * (ordered[quiet_level_count_ / 2 - 1] + ordered[quiet_level_count_ / 2]);
  }

  float motion_floor_() const {
    if (motion_level_count_ == 0) {
      return 0.0f;
    }
    float ordered[STARTUP_MOTION_MAX_LEVELS];
    std::copy(motion_levels_, motion_levels_ + motion_level_count_, ordered);
    std::sort(ordered, ordered + motion_level_count_);
    const uint8_t index = std::min<uint8_t>(
        motion_level_count_ - 1,
        static_cast<uint8_t>(0.10f * static_cast<float>(motion_level_count_)));
    return ordered[index];
  }

  float quiet_ceiling_() const {
    float quiet_ceiling = 0.0f;
    bool has_quiet = false;
    for (uint8_t i = 0; i < quiet_level_count_; i++) {
      quiet_ceiling = has_quiet ? std::max(quiet_ceiling, quiet_levels_[i]) : quiet_levels_[i];
      has_quiet = true;
    }
    for (uint8_t i = 0; i < post_quiet_level_count_; i++) {
      quiet_ceiling = has_quiet ? std::max(quiet_ceiling, post_quiet_levels_[i]) : post_quiet_levels_[i];
      has_quiet = true;
    }
    return has_quiet ? quiet_ceiling : 0.0f;
  }

  float motion_threshold_metric_() const {
    const float motion_floor = motion_floor_();
    const float quiet_ceiling = quiet_ceiling_();
    if (motion_floor <= quiet_ceiling) {
      return motion_floor;
    }
    return 0.5f * (motion_floor + quiet_ceiling);
  }

  bool motion_gap_ok_() const {
    if (motion_level_count_ == 0) {
      return false;
    }
    const float quiet_ceiling = quiet_ceiling_();
    if (quiet_ceiling <= 0.0f) {
      return false;
    }
    return motion_floor_() > STARTUP_MOTION_GAP_RATIO * quiet_ceiling;
  }

  void append_quiet_level_(float value) {
    if (quiet_level_count_ < STARTUP_GATE_CHUNKS) {
      quiet_levels_[quiet_level_count_++] = value;
      return;
    }
    for (uint8_t i = 1; i < STARTUP_GATE_CHUNKS; i++) {
      quiet_levels_[i - 1] = quiet_levels_[i];
    }
    quiet_levels_[STARTUP_GATE_CHUNKS - 1] = value;
  }

  void append_motion_level_(float value) {
    if (motion_level_count_ < STARTUP_MOTION_MAX_LEVELS) {
      motion_levels_[motion_level_count_++] = value;
      return;
    }
    for (uint8_t i = 1; i < STARTUP_MOTION_MAX_LEVELS; i++) {
      motion_levels_[i - 1] = motion_levels_[i];
    }
    motion_levels_[STARTUP_MOTION_MAX_LEVELS - 1] = value;
  }

  void append_post_quiet_level_(float value) {
    if (post_quiet_level_count_ < STARTUP_GATE_CHUNKS) {
      post_quiet_levels_[post_quiet_level_count_++] = value;
      return;
    }
    for (uint8_t i = 1; i < STARTUP_GATE_CHUNKS; i++) {
      post_quiet_levels_[i - 1] = post_quiet_levels_[i];
    }
    post_quiet_levels_[STARTUP_GATE_CHUNKS - 1] = value;
  }

  bool gate_ok_() const {
    const float ring_max = ring_max_();
    const float ring_median = ring_median_();
    if (ring_max > STARTUP_GATE_SPREAD_RATIO * ring_median) {
      return false;
    }
    if (ring_median > STARTUP_GATE_ANCHOR_RATIO * min_chunk_max_) {
      return false;
    }
    return true;
  }

  float ring_max_() const {
    float value = ring_[0];
    for (uint8_t i = 1; i < ring_count_; i++) {
      value = std::max(value, ring_[i]);
    }
    return value;
  }

  float ring_median_() const {
    float ordered[STARTUP_GATE_CHUNKS];
    std::copy(ring_, ring_ + ring_count_, ordered);
    std::sort(ordered, ordered + ring_count_);
    if (ring_count_ % 2 != 0) {
      return ordered[ring_count_ / 2];
    }
    return 0.5f * (ordered[ring_count_ / 2 - 1] + ordered[ring_count_ / 2]);
  }

  uint16_t target_packets_{1};
  bool gate_enabled_{false};
  uint32_t packet_count_{0};
  uint32_t ready_packet_count_{0};
  bool has_value_{false};
  float max_motion_metric_{0.0f};
  bool gate_accepted_{false};
  uint32_t chunk_size_{0};
  uint32_t chunk_count_{0};
  float chunk_max_{0.0f};
  float ring_[STARTUP_GATE_CHUNKS] = {};
  uint8_t ring_count_{0};
  uint8_t ring_next_{0};
  float min_chunk_max_{0.0f};
  float discarded_chunk_max_{0.0f};
  bool has_discarded_chunk_{false};
  float motion_chunk_sum_{0.0f};
  float motion_chunk_max_{0.0f};
  uint8_t motion_chunk_count_{0};
  float bootstrap_levels_[STARTUP_MOTION_MIN_QUIET_CHUNKS] = {};
  uint8_t bootstrap_count_{0};
  float quiet_levels_[STARTUP_GATE_CHUNKS] = {};
  uint8_t quiet_level_count_{0};
  float motion_levels_[STARTUP_MOTION_MAX_LEVELS] = {};
  uint8_t motion_level_count_{0};
  float post_quiet_levels_[STARTUP_GATE_CHUNKS] = {};
  uint8_t post_quiet_level_count_{0};
  bool quiet_anchor_ready_{false};
  bool motion_confirmed_{false};
  bool motion_accepted_{false};
  Phase phase_{Phase::SEEK_MOTION};
  uint8_t consecutive_motion_chunks_{0};
  uint8_t consecutive_post_quiet_chunks_{0};
};

}  // namespace espectre
