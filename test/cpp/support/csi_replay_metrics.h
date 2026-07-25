/*
 * ESPectre - CSI replay helpers for native C++ tests
 *
 * Shared startup-calibration and replay helpers for integration suites that run
 * detectors against saved packet streams and timing metadata.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <algorithm>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "classic_detector.h"
#include "runtime_sensing_schema.h"
#include "threshold.h"
#include "csi_replay_timing.h"

namespace espectre::test::replay {

struct ReplayPacketMetadata {
  const uint32_t* stream_seq_num{nullptr};
  const uint64_t* device_ticks_us{nullptr};
  const uint32_t* wifi_rx_ts_us{nullptr};
};

struct ReplayMetrics {
  int static_presence_eval_count{0};
  int motion_eval_count{0};
  int tp{0};
  int fn{0};
  int fp{0};
  int tn{0};
  int effective_alarms{0};
  int false_motion_evaluations{0};
  float recall{0.0f};
  float precision{0.0f};
  float fp_rate{0.0f};
  float f1{0.0f};
};

inline csi_replay_timing::TimingObservation observe_packet(
    csi_replay_timing::PacketTimingTracker& tracker,
    const ReplayPacketMetadata& metadata,
    int packet_index) {
  return tracker.observe(
      metadata.stream_seq_num != nullptr ? metadata.stream_seq_num[packet_index] : 0U,
      metadata.stream_seq_num != nullptr,
      metadata.device_ticks_us != nullptr ? metadata.device_ticks_us[packet_index] : 0U,
      metadata.device_ticks_us != nullptr,
      metadata.wifi_rx_ts_us != nullptr ? metadata.wifi_rx_ts_us[packet_index] : 0U,
      metadata.wifi_rx_ts_us != nullptr);
}

/**
 * Effective interval of one replayed stream.
 *
 * The Python replay seeds its timing helpers with the measured interval rather
 * than the nominal one, and the seed decides what the fallback returns before
 * the estimator warms up. Seeding differently moved evaluation boundaries and
 * broke report parity, so both sides measure the same way.
 */
inline uint32_t measure_stream_interval_us(const ReplayPacketMetadata& metadata,
                                           int packet_count,
                                           uint32_t nominal_interval_us) {
  return csi_replay_timing::measure_packet_interval_us(
      packet_count,
      [&](int i, uint32_t& out) {
        if (metadata.stream_seq_num == nullptr) return false;
        out = metadata.stream_seq_num[i];
        return true;
      },
      [&](int i, uint64_t& out) {
        if (metadata.device_ticks_us == nullptr) return false;
        out = metadata.device_ticks_us[i];
        return true;
      },
      [&](int i, uint32_t& out) {
        if (metadata.wifi_rx_ts_us == nullptr) return false;
        out = metadata.wifi_rx_ts_us[i];
        return true;
      },
      nominal_interval_us);
}

inline void reset_detector(BaseDetector& detector,
                           csi_replay_timing::TimeAwareCadence& cadence,
                           csi_replay_timing::PacketTimingTracker& tracker) {
  detector.reset();
  detector.clear_buffer();
  cadence.reset();
  tracker.reset();
}

inline void apply_runtime_policy_metrics(const std::vector<bool>& raw_motion_states,
                                         ReplayMetrics& metrics) {
  MotionState effective_state = MotionState::IDLE;
  MotionState pending_state = MotionState::IDLE;
  uint8_t pending_hits = 0;

  for (bool raw_motion : raw_motion_states) {
    const MotionState detector_state = raw_motion ? MotionState::MOTION : MotionState::IDLE;
    const MotionState previous_state = effective_state;

    if (detector_state == effective_state) {
      pending_state = effective_state;
      pending_hits = 0;
    } else {
      if (detector_state != pending_state) {
        pending_state = detector_state;
        pending_hits = 1;
      } else if (pending_hits < UINT8_MAX) {
        pending_hits++;
      }

      const uint8_t required_hits =
          pending_state == MotionState::MOTION ? RUNTIME_MOTION_ON_HITS_DEFAULT
                                               : RUNTIME_MOTION_OFF_HITS_DEFAULT;
      if (pending_hits >= required_hits) {
        effective_state = pending_state;
        pending_hits = 0;
      }
    }

    const bool changed = effective_state != previous_state;
    if (changed && effective_state == MotionState::MOTION) {
      metrics.effective_alarms++;
    }
    if (effective_state == MotionState::MOTION) {
      metrics.false_motion_evaluations++;
    }
  }
}

inline void finalize_metrics(ReplayMetrics& metrics) {
  metrics.recall = (metrics.tp + metrics.fn) > 0
                       ? static_cast<float>(metrics.tp) /
                             static_cast<float>(metrics.tp + metrics.fn) * 100.0f
                       : 0.0f;
  metrics.precision = (metrics.tp + metrics.fp) > 0
                          ? static_cast<float>(metrics.tp) /
                                static_cast<float>(metrics.tp + metrics.fp) * 100.0f
                          : 0.0f;
  metrics.fp_rate = metrics.static_presence_eval_count > 0
                        ? static_cast<float>(metrics.fp) /
                              static_cast<float>(metrics.static_presence_eval_count) * 100.0f
                        : 0.0f;
  metrics.f1 = (metrics.precision + metrics.recall) > 0.0f
                   ? 2.0f * (metrics.precision / 100.0f) * (metrics.recall / 100.0f) /
                         ((metrics.precision + metrics.recall) / 100.0f) * 100.0f
                   : 0.0f;
}

inline bool calibrate_classic_detector(
    ClassicDetector& detector,
    int calibration_packets,
    const int8_t* const* baseline_packets,
    int num_baseline_packets,
    const int8_t* baseline_rssi,
    const ReplayPacketMetadata& baseline_metadata,
    int pkt_size,
    const uint8_t* selected_band,
    uint8_t selected_band_size,
    float& out_threshold) {
  StartupThresholdCalibrator calibrator;
  // Startup runs on the resolved timing contract, so the seed is the interval
  // after the dead band has snapped it, matching _calibration_runtime in
  // dataset_metadata.py. The replay below seeds from the raw measurement,
  // matching _timing_cadence_for_window in performance_report.py.
  const uint32_t nominal_interval_us =
      derive_detector_timing(measure_stream_interval_us(
                                 baseline_metadata, num_baseline_packets,
                                 csi_replay_timing::nominal_packet_interval_us(
                                     detector.get_window_size())))
          .interval_us;
  csi_replay_timing::PacketTimingTracker timing_tracker(nominal_interval_us);
  csi_replay_timing::TimeAwareCadence cadence(
      detector.get_window_size(), RUNTIME_EVALUATION_INTERVAL_DEFAULT);
  detector.on_startup_calibration_begin();
  calibrator.begin(static_cast<uint16_t>(calibration_packets), detector.startup_gate_enabled());

  for (int i = 0; i < num_baseline_packets; i++) {
    csi_replay_timing::TimingObservation timing =
        observe_packet(timing_tracker, baseline_metadata, i);
    if (timing.contaminated) {
      detector.reset();
      detector.clear_buffer();
      detector.on_startup_calibration_begin();
      calibrator.begin(static_cast<uint16_t>(calibration_packets), detector.startup_gate_enabled());
      cadence.reset();
      timing_tracker.reset();
      timing = observe_packet(timing_tracker, baseline_metadata, i);
    }
    detector.process_packet(
        baseline_packets[i],
        static_cast<size_t>(pkt_size),
        selected_band,
        selected_band_size,
        baseline_rssi != nullptr ? baseline_rssi[i] : INT8_MIN);
    cadence.note_packet(timing.coverage_us);
    if (!cadence.should_evaluate()) {
      continue;
    }
    detector.update_state();
    calibrator.observe(
        detector.is_ready(),
        detector.get_motion_metric(),
        detector.get_startup_floor_metric(),
        cadence.packet_weight());
    cadence.after_evaluation();
    if (calibrator.is_complete()) {
      break;
    }
  }

  if (!calibrator.is_successful()) {
    out_threshold = CLASSIC_DEFAULT_THRESHOLD;
    return false;
  }

  detector.on_startup_calibration_complete();
  detector.set_adaptive_threshold(calibrator.threshold_metric());
  out_threshold = detector.get_threshold();
  detector.reset();
  detector.clear_buffer();
  return true;
}

template <typename Detector>
ReplayMetrics evaluate_detector(
    Detector& detector,
    const int8_t* const* baseline_packets,
    int num_baseline_packets,
    const int8_t* baseline_rssi,
    const ReplayPacketMetadata& baseline_metadata,
    const int8_t* const* motion_packets,
    int num_motion_packets,
    const int8_t* motion_rssi,
    const ReplayPacketMetadata& motion_metadata,
    int pkt_size,
    const uint8_t* selected_band,
    uint8_t selected_band_size) {
  ReplayMetrics metrics{};
  const int warmup = detector.get_window_size();
  const uint32_t nominal_interval_us = measure_stream_interval_us(
      baseline_metadata, num_baseline_packets,
      csi_replay_timing::nominal_packet_interval_us(detector.get_window_size()));
  csi_replay_timing::PacketTimingTracker timing_tracker(nominal_interval_us);
  csi_replay_timing::TimeAwareCadence cadence(
      detector.get_window_size(), RUNTIME_EVALUATION_INTERVAL_DEFAULT);
  int packets_since_reset = 0;
  int debug_contam_base = 0;
  int debug_contam_motion = 0;
  std::vector<bool> baseline_motion_states;

  for (int i = 0; i < num_baseline_packets; i++) {
    csi_replay_timing::TimingObservation timing =
        observe_packet(timing_tracker, baseline_metadata, i);
    if (timing.contaminated) {
      debug_contam_base++;
      reset_detector(detector, cadence, timing_tracker);
      packets_since_reset = 0;
      timing = observe_packet(timing_tracker, baseline_metadata, i);
    }
    detector.process_packet(
        baseline_packets[i],
        static_cast<size_t>(pkt_size),
        selected_band,
        selected_band_size,
        baseline_rssi != nullptr ? baseline_rssi[i] : INT8_MIN);
    packets_since_reset++;
    cadence.note_packet(timing.coverage_us);
    if (!cadence.should_evaluate()) {
      continue;
    }
    detector.update_state();
    cadence.after_evaluation();
    if (packets_since_reset < warmup) {
      continue;
    }
    metrics.static_presence_eval_count++;
    const bool raw_motion = detector.get_state() == MotionState::MOTION;
    baseline_motion_states.push_back(raw_motion);
    if (raw_motion) {
      metrics.fp++;
    }
  }

  // The motion stream is a separate recording, so it gets a fresh rate
  // estimator rather than inheriting the baseline's learned cadence.
  // PacketTimingTracker::reset() deliberately keeps that cadence, which is
  // right inside one stream and wrong across two; carrying it over made the
  // first motion packet fall back to a warm median where the Python replay,
  // which builds new helpers per stream, used the nominal interval.
  timing_tracker = csi_replay_timing::PacketTimingTracker(nominal_interval_us);
  cadence.reset();
  packets_since_reset = 0;
  for (int i = 0; i < num_motion_packets; i++) {
    csi_replay_timing::TimingObservation timing =
        observe_packet(timing_tracker, motion_metadata, i);
    if (timing.contaminated) {
      debug_contam_motion++;
      reset_detector(detector, cadence, timing_tracker);
      packets_since_reset = 0;
      timing = observe_packet(timing_tracker, motion_metadata, i);
    }
    detector.process_packet(
        motion_packets[i],
        static_cast<size_t>(pkt_size),
        selected_band,
        selected_band_size,
        motion_rssi != nullptr ? motion_rssi[i] : INT8_MIN);
    packets_since_reset++;
    cadence.note_packet(timing.coverage_us);
    if (!cadence.should_evaluate()) {
      continue;
    }
    detector.update_state();
    cadence.after_evaluation();
    if (packets_since_reset < warmup) {
      continue;
    }
    metrics.motion_eval_count++;
    if (detector.get_state() == MotionState::MOTION) {
      metrics.tp++;
    } else {
      metrics.fn++;
    }
  }

  if (std::getenv("ESPECTRE_REPLAY_DEBUG") != nullptr) {
    std::fprintf(stderr, "[replay] base_evals=%d motion_evals=%d fp=%d tp=%d fn=%d contam_base=%d contam_motion=%d\n",
                 metrics.static_presence_eval_count, metrics.motion_eval_count,
                 metrics.fp, metrics.tp, metrics.fn, debug_contam_base, debug_contam_motion);
  }
  metrics.tn = std::max(metrics.static_presence_eval_count - metrics.fp, 0);
  finalize_metrics(metrics);
  apply_runtime_policy_metrics(baseline_motion_states, metrics);
  return metrics;
}

}  // namespace espectre::test::replay
