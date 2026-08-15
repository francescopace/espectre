/*
 * ESPectre - CSI replay helpers for native C++ tests
 *
 * Shared startup-calibration and replay helpers for integration suites that run
 * detectors against saved packet streams and timing metadata.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include "lightweight_detector.h"
#include "runtime_sensing_schema.h"
#include "threshold.h"
#include "temporal_csi_sampler.h"
#include "csi_replay_timing.h"

namespace espectre::test::replay {

struct ReplayPacketMetadata {
  const uint32_t* stream_seq_num{nullptr};
  const uint64_t* device_ticks_us{nullptr};
  const uint32_t* wifi_rx_ts_us{nullptr};
  uint32_t csi_target_pps{0U};
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

inline uint64_t packet_timestamp_us(const ReplayPacketMetadata& metadata,
                                    int packet_index,
                                    uint32_t nominal_interval_us) {
  if (metadata.wifi_rx_ts_us != nullptr) {
    return metadata.wifi_rx_ts_us[packet_index];
  }
  if (metadata.device_ticks_us != nullptr) {
    return metadata.device_ticks_us[packet_index];
  }
  return static_cast<uint64_t>(packet_index) * nominal_interval_us;
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

inline uint32_t target_pps(const ReplayPacketMetadata& metadata,
                           int packet_count) {
  if (metadata.csi_target_pps > 0U) {
    return metadata.csi_target_pps;
  }
  const uint32_t measured_interval_us = measure_stream_interval_us(
      metadata, packet_count,
      csi_replay_timing::nominal_packet_interval_us(DETECTOR_DEFAULT_WINDOW_SIZE));
  return std::max<uint32_t>(
      1U, static_cast<uint32_t>(std::llround(1000000.0 / measured_interval_us)));
}

/** Resolve the production time window for one replay stream. */
inline uint16_t detector_window_packets(const ReplayPacketMetadata& metadata,
                                        int packet_count) {
  return temporal_window_slots(
      target_pps(metadata, packet_count),
      RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT);
}

/** Resolve the production startup duration for one replay stream. */
inline uint16_t calibration_packet_count(const ReplayPacketMetadata& metadata,
                                         int packet_count) {
  const uint64_t duration_us =
      static_cast<uint64_t>(RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT) *
      1000U * CALIBRATION_NUM_WINDOWS;
  const uint64_t rounded_packets =
      (duration_us * target_pps(metadata, packet_count) + 500000U) / 1000000U;
  return static_cast<uint16_t>(std::max<uint64_t>(
      1U, std::min<uint64_t>(rounded_packets, UINT16_MAX)));
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

inline bool calibrate_lightweight_detector(
    LightweightDetector& detector,
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
  const uint32_t replay_target_pps = target_pps(
      baseline_metadata, num_baseline_packets);
  const uint32_t nominal_interval_us =
      std::max<uint32_t>(1U, static_cast<uint32_t>(std::llround(
          1000000.0 / replay_target_pps)));
  TemporalCsiSampler sampler(
      replay_target_pps, RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT);
  detector.set_minimum_valid_samples(
      static_cast<uint16_t>(sampler.minimum_valid_slots()));
  csi_replay_timing::TimeAwareCadence cadence(
      detector.get_window_size(), RUNTIME_EVALUATION_INTERVAL_MS_DEFAULT,
      nominal_interval_us);
  detector.on_startup_calibration_begin();
  calibrator.begin(static_cast<uint16_t>(calibration_packets), detector.startup_gate_enabled());

  for (int i = 0; i < num_baseline_packets; i++) {
    const uint32_t timestamp_us = static_cast<uint32_t>(packet_timestamp_us(
        baseline_metadata, i, nominal_interval_us));
    if (!sampler.admit(timestamp_us)) {
      continue;
    }
    if (sampler.reset_required()) {
      detector.reset();
      detector.clear_buffer();
      detector.on_startup_calibration_begin();
      calibrator.begin(static_cast<uint16_t>(calibration_packets), detector.startup_gate_enabled());
      cadence.reset();
    }
    if (sampler.missing_slots_before() > 0U) {
      detector.advance_missing_slots(static_cast<uint32_t>(std::min<uint64_t>(
          sampler.missing_slots_before(), detector.get_window_size())));
    }
    detector.set_packet_timestamp_us(timestamp_us);
    detector.process_packet(
        baseline_packets[i],
        static_cast<size_t>(pkt_size),
        selected_band,
        selected_band_size,
        baseline_rssi != nullptr ? baseline_rssi[i] : INT8_MIN);
    cadence.note_packet(static_cast<uint32_t>(
        sampler.slots_advanced() * nominal_interval_us));
    if (!cadence.should_evaluate()) {
      continue;
    }
    detector.update_state();
    // Occupancy holes leave the detector not ready. Firmware and Python
    // calibration skip those ticks so they do not consume the startup budget.
    // Observing them here previously completed calibration earlier than the
    // production path and moved Lightweight report metrics off Python.
    if (detector.is_ready()) {
      calibrator.observe(
          true,
          detector.get_motion_metric(),
          cadence.packet_weight());
    }
    cadence.after_evaluation();
    if (calibrator.is_complete()) {
      break;
    }
  }

  if (!calibrator.is_successful()) {
    out_threshold = LIGHTWEIGHT_DEFAULT_THRESHOLD;
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
  const uint32_t replay_target_pps = target_pps(
      baseline_metadata, num_baseline_packets);
  const uint32_t nominal_interval_us =
      std::max<uint32_t>(1U, static_cast<uint32_t>(std::llround(
          1000000.0 / replay_target_pps)));
  TemporalCsiSampler sampler(
      replay_target_pps, RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT);
  detector.set_minimum_valid_samples(
      static_cast<uint16_t>(sampler.minimum_valid_slots()));
  csi_replay_timing::TimeAwareCadence cadence(
      detector.get_window_size(), RUNTIME_EVALUATION_INTERVAL_MS_DEFAULT,
      nominal_interval_us);
  int packets_since_reset = 0;
  int debug_contam_base = 0;
  int debug_contam_motion = 0;
  std::vector<bool> baseline_motion_states;

  for (int i = 0; i < num_baseline_packets; i++) {
    const uint32_t timestamp_us = static_cast<uint32_t>(packet_timestamp_us(
        baseline_metadata, i, nominal_interval_us));
    if (!sampler.admit(timestamp_us)) {
      continue;
    }
    if (sampler.reset_required()) {
      debug_contam_base++;
      detector.reset();
      detector.clear_buffer();
      cadence.reset();
      packets_since_reset = 0;
    }
    if (sampler.missing_slots_before() > 0U) {
      detector.advance_missing_slots(static_cast<uint32_t>(std::min<uint64_t>(
          sampler.missing_slots_before(), detector.get_window_size())));
    }
    detector.set_packet_timestamp_us(timestamp_us);
    detector.process_packet(
        baseline_packets[i],
        static_cast<size_t>(pkt_size),
        selected_band,
        selected_band_size,
        baseline_rssi != nullptr ? baseline_rssi[i] : INT8_MIN);
    packets_since_reset = static_cast<int>(std::min<uint64_t>(
        sampler.current_slot() + 1U, INT_MAX));
    cadence.note_packet(static_cast<uint32_t>(
        sampler.slots_advanced() * nominal_interval_us));
    if (!cadence.should_evaluate()) {
      continue;
    }
    detector.update_state();
    cadence.after_evaluation();
    if (packets_since_reset < warmup || !detector.is_ready()) {
      continue;
    }
    metrics.static_presence_eval_count++;
    const bool raw_motion = detector.get_state() == MotionState::MOTION;
    baseline_motion_states.push_back(raw_motion);
    if (raw_motion) {
      metrics.fp++;
    }
  }

  // The motion stream is a separate recording, so it gets a fresh temporal
  // admission grid while retaining the detector state used by paired replay.
  sampler = TemporalCsiSampler(
      replay_target_pps, RUNTIME_SEGMENTATION_WINDOW_SIZE_MS_DEFAULT);
  cadence.reset();
  packets_since_reset = 0;
  for (int i = 0; i < num_motion_packets; i++) {
    const uint32_t timestamp_us = static_cast<uint32_t>(packet_timestamp_us(
        motion_metadata, i, nominal_interval_us));
    if (!sampler.admit(timestamp_us)) {
      continue;
    }
    if (sampler.reset_required()) {
      debug_contam_motion++;
      detector.reset();
      detector.clear_buffer();
      cadence.reset();
      packets_since_reset = 0;
    }
    if (sampler.missing_slots_before() > 0U) {
      detector.advance_missing_slots(static_cast<uint32_t>(std::min<uint64_t>(
          sampler.missing_slots_before(), detector.get_window_size())));
    }
    detector.set_packet_timestamp_us(timestamp_us);
    detector.process_packet(
        motion_packets[i],
        static_cast<size_t>(pkt_size),
        selected_band,
        selected_band_size,
        motion_rssi != nullptr ? motion_rssi[i] : INT8_MIN);
    packets_since_reset = static_cast<int>(std::min<uint64_t>(
        sampler.current_slot() + 1U, INT_MAX));
    cadence.note_packet(static_cast<uint32_t>(
        sampler.slots_advanced() * nominal_interval_us));
    if (!cadence.should_evaluate()) {
      continue;
    }
    detector.update_state();
    cadence.after_evaluation();
    if (packets_since_reset < warmup || !detector.is_ready()) {
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
