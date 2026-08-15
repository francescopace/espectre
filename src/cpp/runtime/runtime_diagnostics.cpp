/*
 * ESPectre - Runtime Diagnostics
 *
 * Runtime diagnostics snapshot helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_diagnostics.h"

#include <cstdio>

#include "runtime_config_utils.h"

namespace espectre {

namespace {

uint64_t counter_delta(uint64_t current, uint64_t previous) {
  return current >= previous ? current - previous : current;
}

float packets_per_second(uint64_t delta, uint32_t elapsed_ms) {
  return elapsed_ms > 0U
             ? static_cast<float>(delta) * 1000.0f / static_cast<float>(elapsed_ms)
             : 0.0f;
}

}  // namespace

void RuntimeDiagnosticsSampler::reset(const RuntimeDiagnosticsSnapshot &snapshot, uint32_t now_ms) {
  previous_ = snapshot;
  previous_ms_ = now_ms;
  baseline_ready_ = true;
}

RuntimeDiagnosticsSample RuntimeDiagnosticsSampler::sample(const RuntimeDiagnosticsSnapshot &snapshot,
                                                            uint32_t now_ms) {
  RuntimeDiagnosticsSample result;
  result.wifi_rssi_dbm = snapshot.wifi_rssi_dbm;
  result.wifi_channel = snapshot.wifi_channel;
  if (!baseline_ready_) {
    reset(snapshot, now_ms);
    return result;
  }

  const uint32_t elapsed_ms = now_ms - previous_ms_;
  if (elapsed_ms == 0U) {
    return result;
  }
  result.traffic_tx_pps = packets_per_second(
      counter_delta(snapshot.traffic_packets_total, previous_.traffic_packets_total), elapsed_ms);
  result.csi_callback_pps = packets_per_second(
      counter_delta(snapshot.csi_callbacks_total, previous_.csi_callbacks_total), elapsed_ms);
  result.csi_accepted_pps = packets_per_second(
      counter_delta(snapshot.csi_accepted_total, previous_.csi_accepted_total), elapsed_ms);
  result.csi_admitted_pps = packets_per_second(
      counter_delta(snapshot.csi_admitted_total, previous_.csi_admitted_total), elapsed_ms);
  result.csi_filtered_pps = packets_per_second(
      counter_delta(snapshot.csi_filtered_total, previous_.csi_filtered_total), elapsed_ms);
  result.csi_missing_slots_pps = packets_per_second(
      counter_delta(snapshot.csi_missing_slots_total, previous_.csi_missing_slots_total), elapsed_ms);
  result.csi_excess_pps = packets_per_second(
      counter_delta(snapshot.csi_excess_total, previous_.csi_excess_total), elapsed_ms);
  result.csi_stale_pps = packets_per_second(
      counter_delta(snapshot.csi_stale_total, previous_.csi_stale_total), elapsed_ms);
  result.csi_out_of_order_pps = packets_per_second(
      counter_delta(snapshot.csi_out_of_order_total, previous_.csi_out_of_order_total), elapsed_ms);
  result.csi_occupancy_ratio = snapshot.csi_window_slots > 0U
      ? static_cast<float>(snapshot.csi_occupancy_slots) /
            static_cast<float>(snapshot.csi_window_slots)
      : 0.0f;
  reset(snapshot, now_ms);
  return result;
}

void visit_runtime_diagnostics(const RuntimeConfig &config,
                               const RuntimeSnapshot &snapshot,
                               runtime_diagnostic_visitor_t visitor) {
  if (!visitor) {
    return;
  }

  char value[64];

  std::snprintf(value, sizeof(value), "%.6f", snapshot.threshold);
  visitor("threshold", value);
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.segmentation_window_size_ms));
  visitor("window_ms", value);
  visitor("detector", snapshot.detector_name);
  visitor("lowpass", config.lowpass_enabled ? "on" : "off");
  std::snprintf(value, sizeof(value), "%.1f", config.lowpass_cutoff);
  visitor("lowpass_cutoff", value);
  visitor("hampel", config.hampel_enabled ? "on" : "off");
  if (config.hampel_enabled) {
    std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.hampel_window));
    visitor("hampel_window", value);
    std::snprintf(value, sizeof(value), "%.1f", config.hampel_threshold);
    visitor("hampel_threshold", value);
  }
  visitor("traffic_mode", traffic_mode_name(config.traffic_generator_mode));
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.csi_target_pps));
  visitor("csi_target_pps", value);
  visitor("traffic_adaptive", config.traffic_generator_adaptive ? "on" : "off");
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.publish_interval_ms));
  visitor("publish_interval_ms", value);
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.evaluation_interval_ms));
  visitor("evaluation_interval_ms", value);
  std::snprintf(value,
                sizeof(value),
                "%u/%u",
                static_cast<unsigned>(config.motion_on_hits),
                static_cast<unsigned>(config.motion_off_hits));
  visitor("motion_hits", value);
}

}  // namespace espectre
