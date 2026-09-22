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
  result.wifi_rssi_dbm = snapshot.link.rssi_dbm;
  result.wifi_channel = snapshot.link.channel;
  if (!baseline_ready_) {
    reset(snapshot, now_ms);
    return result;
  }

  const uint32_t elapsed_ms = now_ms - previous_ms_;
  if (elapsed_ms == 0U) {
    return result;
  }
  result.generator_pps = packets_per_second(
      counter_delta(snapshot.traffic.generator_packets_total, previous_.traffic.generator_packets_total), elapsed_ms);
  result.traffic_tx_pps = packets_per_second(
      static_cast<uint32_t>(snapshot.traffic.tx_packets_total - previous_.traffic.tx_packets_total), elapsed_ms);
  result.traffic_rx_pps = packets_per_second(
      static_cast<uint32_t>(snapshot.traffic.rx_packets_total - previous_.traffic.rx_packets_total), elapsed_ms);
  result.csi_callback_pps = packets_per_second(
      counter_delta(snapshot.csi.callbacks_total, previous_.csi.callbacks_total), elapsed_ms);
  result.csi_accepted_pps = packets_per_second(
      counter_delta(snapshot.csi.accepted_total, previous_.csi.accepted_total), elapsed_ms);
  result.csi_admitted_pps = packets_per_second(
      counter_delta(snapshot.csi.admitted_total, previous_.csi.admitted_total), elapsed_ms);
  result.csi_filtered_pps = packets_per_second(
      counter_delta(snapshot.csi.filtered_total, previous_.csi.filtered_total), elapsed_ms);
  result.csi_hw_error_pps = packets_per_second(
      counter_delta(snapshot.csi.rx_error_total, previous_.csi.rx_error_total) +
          counter_delta(snapshot.csi.rx_end_error_total, previous_.csi.rx_end_error_total) +
          counter_delta(snapshot.csi.invalid_estimate_total, previous_.csi.invalid_estimate_total) +
          counter_delta(snapshot.csi.invalid_first_word_total, previous_.csi.invalid_first_word_total),
      elapsed_ms);
  result.csi_pending_frame_drop_pps = packets_per_second(
      counter_delta(snapshot.csi.pending_frame_drops_total,
                    previous_.csi.pending_frame_drops_total),
      elapsed_ms);
  result.csi_missing_slots_pps = packets_per_second(
      counter_delta(snapshot.csi.missing_slots_total, previous_.csi.missing_slots_total), elapsed_ms);
  result.csi_excess_pps = packets_per_second(
      counter_delta(snapshot.csi.excess_total, previous_.csi.excess_total), elapsed_ms);
  result.csi_stale_pps = packets_per_second(
      counter_delta(snapshot.csi.stale_total, previous_.csi.stale_total), elapsed_ms);
  result.csi_out_of_order_pps = packets_per_second(
      counter_delta(snapshot.csi.out_of_order_total, previous_.csi.out_of_order_total), elapsed_ms);
  result.csi_occupancy_ratio = snapshot.csi.window_slots > 0U
      ? static_cast<float>(snapshot.csi.occupancy_slots) /
            static_cast<float>(snapshot.csi.window_slots)
      : 0.0f;
  reset(snapshot, now_ms);
  return result;
}

}  // namespace espectre
