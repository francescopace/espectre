/*
 * ESPectre - Runtime Diagnostics
 *
 * Runtime diagnostics rate sampling.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_diagnostics.h"

#include "counter_helpers.h"

namespace espectre {

namespace {

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
  result.generator_pps = packets_per_second(
      counter_delta(snapshot.generator_packets_total, previous_.generator_packets_total), elapsed_ms);
  result.traffic_tx_pps = packets_per_second(
      static_cast<uint32_t>(snapshot.traffic_tx_packets_total - previous_.traffic_tx_packets_total), elapsed_ms);
  result.traffic_rx_pps = packets_per_second(
      static_cast<uint32_t>(snapshot.traffic_rx_packets_total - previous_.traffic_rx_packets_total), elapsed_ms);
  result.csi_callback_pps = packets_per_second(
      counter_delta(snapshot.csi_callbacks_total, previous_.csi_callbacks_total), elapsed_ms);
  result.csi_accepted_pps = packets_per_second(
      counter_delta(snapshot.csi_accepted_total, previous_.csi_accepted_total), elapsed_ms);
  result.csi_admitted_pps = packets_per_second(
      counter_delta(snapshot.csi_admitted_total, previous_.csi_admitted_total), elapsed_ms);
  result.csi_filtered_pps = packets_per_second(
      counter_delta(snapshot.csi_filtered_total, previous_.csi_filtered_total), elapsed_ms);
  result.csi_hw_error_pps = packets_per_second(
      counter_delta(snapshot.csi_rx_error_total, previous_.csi_rx_error_total) +
          counter_delta(snapshot.csi_rx_end_error_total, previous_.csi_rx_end_error_total) +
          counter_delta(snapshot.csi_invalid_estimate_total, previous_.csi_invalid_estimate_total) +
          counter_delta(snapshot.csi_invalid_first_word_total, previous_.csi_invalid_first_word_total),
      elapsed_ms);
  result.csi_pending_frame_drop_pps = packets_per_second(
      counter_delta(snapshot.csi_pending_frame_drops_total,
                    previous_.csi_pending_frame_drops_total),
      elapsed_ms);
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

}  // namespace espectre
