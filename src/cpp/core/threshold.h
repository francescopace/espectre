/*
 * ESPectre - Adaptive Threshold Calculator
 *
 * Calculates adaptive threshold from calibration baseline values.
 * Called after calibration to compute the detection threshold.
 *
 * Startup-threshold formula: threshold = max(cal_values) x factor
 *
 * Modes:
 * - "auto": max x 1.3 (default, lower false positives on no-gain-lock captures)
 * - "min": max x 1.0 (maximum sensitivity, may have FP)
 *
 * Detectors with a tight quiet floor (l1_delta) additionally enable a startup
 * consistency gate (StartupThresholdCalibrator below): calibration is accepted
 * only when the ring of per-chunk metric maxima is self-consistent, and it
 * extends chunk by chunk otherwise. Keep the semantics aligned with
 * src/python/micro_espectre/threshold.py; see docs/EXPERIMENTS.md,
 * "L1-Delta Contaminated-Calibration Gate And Extension Sweep" (2026-07-06).
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace esphome {
namespace espectre {

// Multiplier for "auto" mode threshold (reduces false positives)
constexpr float DEFAULT_ADAPTIVE_FACTOR = 1.3f;

// Startup calibration consistency gate (benchmark-tuned on the paired
// datasets; keep aligned with src/python/micro_espectre/threshold.py).
constexpr uint8_t STARTUP_GATE_CHUNKS = 6;
constexpr float STARTUP_GATE_SPREAD_RATIO = 1.10f;
constexpr float STARTUP_GATE_ANCHOR_RATIO = 1.5f;
constexpr uint16_t STARTUP_GATE_EXTENSION_PACKETS = 2000;

/**
 * Threshold mode enumeration
 */
enum class ThresholdMode {
  AUTO,    // max x 1.3 (default)
  MIN,     // max x 1.0 (maximum sensitivity)
  MANUAL   // User-specified fixed value (no adaptive calculation)
};


/**
 * Calculate the maximum value from a vector.
 * 
 * @param values Vector of numeric values
 * @return Maximum value (1.0f if vector is empty)
 */
inline float calculate_max_value(const std::vector<float>& values) {
  if (values.empty()) {
    return 1.0f;
  }
  return *std::max_element(values.begin(), values.end());
}

/**
 * Get threshold multiplier from mode
 *
 * @param mode Threshold mode (AUTO or MIN)
 * @param auto_factor Detector-specific AUTO multiplier (default: 1.3)
 * @return multiplier value (auto_factor for AUTO, 1.0 for MIN)
 */
inline float get_threshold_factor(ThresholdMode mode, float auto_factor = DEFAULT_ADAPTIVE_FACTOR) {
  if (mode == ThresholdMode::AUTO) {
    return auto_factor;
  }
  return 1.0f;  // MIN: no multiplier
}

/**
 * Calculate adaptive threshold from calibration baseline values
 * 
 * Shared startup path: threshold = max(cal_values) x factor for the current
 * production modes
 * 
 * AUTO mode applies a 1.3x multiplier to reduce false positives.
 * MIN mode uses the raw max moving variance for maximum sensitivity.
 * 
 * @param cal_values Vector of moving variance values from calibration
 * @param mode Threshold mode (AUTO or MIN)
 * @param out_threshold Output: calculated adaptive threshold
 * @param out_factor Output: multiplier used
 */
inline void calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    ThresholdMode mode,
    float& out_threshold,
    float& out_factor) {
  out_factor = get_threshold_factor(mode);
  out_threshold = calculate_max_value(cal_values) * out_factor;
}

/**
 * Calculate adaptive threshold with an explicit factor.
 *
 * @param cal_values Vector of moving variance values from baseline
 * @param factor Multiplier to apply to the max moving variance
 * @return Calculated adaptive threshold
 */
inline float calculate_adaptive_threshold(
    const std::vector<float>& cal_values,
    float factor) {
  return calculate_max_value(cal_values) * factor;
}

/**
 * Startup threshold calibrator with an optional consistency gate.
 *
 * Tracks the max ready-state motion metric during startup calibration.
 * With the gate enabled, ready-state metrics are grouped into
 * STARTUP_GATE_CHUNKS chunks and only the per-chunk maxima are kept
 * (fixed ring, no metric buffer). Calibration is accepted when:
 * - spread gate: max(ring) <= STARTUP_GATE_SPREAD_RATIO x median(ring)
 * - floor anchor: median(ring) <= STARTUP_GATE_ANCHOR_RATIO x min chunk
 *   max ever observed in the session
 *
 * On rejection the calibration window extends one chunk at a time, up to
 * STARTUP_GATE_EXTENSION_PACKETS beyond the nominal target; on budget
 * exhaustion the threshold metric falls back to the ring median.
 *
 * Chunks discarded during extension that stayed within the floor anchor
 * band of the accepted ring are treated as session quiet tail rather than
 * motion: their peak is folded back into the threshold metric ("tail
 * rescue"), so the extension can never lower the threshold below a
 * recurring quiet-tail bump.
 *
 * Mirrors StartupThresholdCalibrator in
 * src/python/micro_espectre/threshold.py.
 */
class StartupThresholdCalibrator {
 public:
  void begin(uint16_t target_packets, bool gate_enabled) {
    target_packets_ = target_packets > 0 ? target_packets : 1;
    gate_enabled_ = gate_enabled;
    packet_count_ = 0;
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
  }

  /**
   * Consume one processed detector step.
   *
   * @param detector_ready Whether the detector metric window is full
   * @param motion_metric Current detector motion metric
   */
  void observe(bool detector_ready, float motion_metric) {
    packet_count_++;
    if (!detector_ready) {
      return;
    }
    if (!has_value_ || motion_metric > max_motion_metric_) {
      max_motion_metric_ = motion_metric;
    }
    has_value_ = true;
    if (gate_enabled_ && !gate_accepted_) {
      observe_gate_metric_(motion_metric);
    }
  }

  /// True once calibration is accepted (or the extension budget is spent).
  bool is_complete() const {
    if (packet_count_ < target_packets_) {
      return false;
    }
    if (!gate_enabled_ || gate_accepted_) {
      return true;
    }
    return packet_count_ >=
           static_cast<uint32_t>(target_packets_) + STARTUP_GATE_EXTENSION_PACKETS;
  }

  /// True while the gate is holding calibration open past the nominal target.
  bool is_extending() const {
    return gate_enabled_ && !gate_accepted_ && packet_count_ >= target_packets_;
  }

  bool is_successful() const { return has_value_; }
  bool gate_accepted() const { return gate_accepted_; }
  uint32_t packet_count() const { return packet_count_; }
  uint16_t target_packets() const { return target_packets_; }

  /// Metric the threshold formula (x factor) is applied to.
  float threshold_metric() const {
    if (!gate_enabled_ || ring_count_ == 0) {
      return has_value_ ? max_motion_metric_ : 0.0f;
    }
    if (gate_accepted_) {
      float metric = ring_max_();
      // Tail rescue: discarded chunks within the anchor band of the
      // accepted floor are quiet tail, not motion; keep their peak so
      // the extension cannot end below a recurring tail bump.
      if (has_discarded_chunk_ &&
          discarded_chunk_max_ <= STARTUP_GATE_ANCHOR_RATIO * ring_median_()) {
        metric = std::max(metric, discarded_chunk_max_);
      }
      return metric;
    }
    // Extension budget exhausted: robust fallback on the last ring.
    return ring_median_();
  }

  /// Statistic name for threshold logging.
  const char* statistic_name() const {
    if (!gate_enabled_ || ring_count_ == 0) {
      return "max";
    }
    return gate_accepted_ ? "gated max" : "gated median";
  }

 private:
  void observe_gate_metric_(float metric) {
    if (chunk_size_ == 0) {
      // Size the chunks so the initial ring spans the remainder of the
      // nominal calibration window from the first ready sample.
      const uint32_t remaining =
          packet_count_ <= target_packets_ ? target_packets_ - packet_count_ + 1 : 1;
      chunk_size_ = std::max<uint32_t>(1, remaining / STARTUP_GATE_CHUNKS);
    }

    if (chunk_count_ == 0 || metric > chunk_max_) {
      chunk_max_ = metric;
    }
    chunk_count_++;
    if (chunk_count_ < chunk_size_) {
      return;
    }

    // Close the chunk: slide the ring and track the session floor.
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
};

}  // namespace espectre
}  // namespace esphome
