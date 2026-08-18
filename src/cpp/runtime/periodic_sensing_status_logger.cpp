/*
 * ESPectre - Periodic Sensing Status Logger
 *
 * Periodically logs sensing status snapshots.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "periodic_sensing_status_logger.h"

#include "espectre_log.h"
#include "runtime_time.h"

namespace espectre {

void PeriodicSensingStatusLogger::log_status(const char *tag,
                                             const RuntimeSnapshot &snapshot,
                                             uint32_t packets_per_publish,
                                             const RuntimeDiagnosticsSample *diagnostics) {
  if (!tag) {
    return;
  }

  const float motion_metric = snapshot.movement_metric;
  const float threshold = snapshot.threshold;
  const bool is_motion = (snapshot.motion_state == MotionState::MOTION);

  const uint32_t now_ms = monotonic_now_ms();

  uint32_t rate_pps = 0;
  uint32_t raw_rate_pps = 0;
  uint32_t traffic_rate_pps = 0;
  uint32_t missing_rate_pps = 0;
  uint32_t excess_rate_pps = 0;
  uint32_t stale_rate_pps = 0;
  uint32_t out_of_order_rate_pps = 0;
  uint32_t occupancy_percent = 0;
  if (diagnostics != nullptr) {
    rate_pps = static_cast<uint32_t>(diagnostics->csi_admitted_pps);
    raw_rate_pps = static_cast<uint32_t>(diagnostics->csi_accepted_pps);
    traffic_rate_pps = static_cast<uint32_t>(diagnostics->traffic_tx_pps);
    missing_rate_pps = static_cast<uint32_t>(diagnostics->csi_missing_slots_pps);
    excess_rate_pps = static_cast<uint32_t>(diagnostics->csi_excess_pps);
    stale_rate_pps = static_cast<uint32_t>(diagnostics->csi_stale_pps);
    out_of_order_rate_pps = static_cast<uint32_t>(diagnostics->csi_out_of_order_pps);
    occupancy_percent = static_cast<uint32_t>(diagnostics->csi_occupancy_ratio * 100.0f + 0.5f);
  } else if (last_log_time_ms_ > 0 && now_ms > last_log_time_ms_) {
    const uint32_t elapsed_ms = now_ms - last_log_time_ms_;
    if (elapsed_ms > 0) {
      rate_pps = static_cast<uint32_t>((static_cast<uint64_t>(packets_per_publish) * 1000U) / elapsed_ms);
    }
  }
  last_log_time_ms_ = now_ms;

  // Link quality comes from the packets that produced the metric, not from a
  // fresh AP query taken at print time.
  const int8_t rssi = snapshot.link_rssi_dbm;
  const uint8_t channel = snapshot.link_channel;
  constexpr int kBarWidth = 20;

  if (snapshot.calibrating) {
    float calibration_progress = 0.0f;
    if (snapshot.calibration_target_packets > 0U) {
      calibration_progress =
          static_cast<float>(snapshot.calibration_packets) /
          static_cast<float>(snapshot.calibration_target_packets);
    }
    if (calibration_progress < 0.0f) {
      calibration_progress = 0.0f;
    } else if (calibration_progress > 1.0f) {
      calibration_progress = 1.0f;
    }
    log_progress_bar(tag, calibration_progress, kBarWidth, -1,
                     "| mvmt:%.6f thr:%.6f | CALIBRATING | csi:%u/%u tx:%u occ:%u%% "
                     "miss:%u excess:%u stale:%u ooo:%u | ch:%u rssi:%d",
                     motion_metric, threshold,
                     static_cast<unsigned>(rate_pps),
                     static_cast<unsigned>(raw_rate_pps),
                     static_cast<unsigned>(traffic_rate_pps),
                     static_cast<unsigned>(occupancy_percent),
                     static_cast<unsigned>(missing_rate_pps),
                     static_cast<unsigned>(excess_rate_pps),
                     static_cast<unsigned>(stale_rate_pps),
                     static_cast<unsigned>(out_of_order_rate_pps),
                     static_cast<unsigned>(channel),
                     static_cast<int>(rssi));
    return;
  }

  float bar_progress = motion_metric;
  if (bar_progress < 0.0f) {
    bar_progress = 0.0f;
  } else if (bar_progress > 1.0f) {
    bar_progress = 1.0f;
  }

  int threshold_pos = -1;
  if (threshold > 0.0f) {
    threshold_pos = static_cast<int>(threshold * static_cast<float>(kBarWidth) + 0.5f);
    if (threshold_pos >= kBarWidth) {
      threshold_pos = kBarWidth - 1;
    } else if (threshold_pos < 0) {
      threshold_pos = 0;
    }
  }

  log_progress_bar(tag, bar_progress, kBarWidth, threshold_pos,
                   "| mvmt:%.6f thr:%.6f | %s | csi:%u/%u tx:%u occ:%u%% "
                   "miss:%u excess:%u stale:%u ooo:%u | ch:%u rssi:%d",
                   motion_metric, threshold,
                   is_motion ? "MOTION" : "IDLE",
                   static_cast<unsigned>(rate_pps),
                   static_cast<unsigned>(raw_rate_pps),
                   static_cast<unsigned>(traffic_rate_pps),
                   static_cast<unsigned>(occupancy_percent),
                   static_cast<unsigned>(missing_rate_pps),
                   static_cast<unsigned>(excess_rate_pps),
                   static_cast<unsigned>(stale_rate_pps),
                   static_cast<unsigned>(out_of_order_rate_pps),
                   static_cast<unsigned>(channel),
                   static_cast<int>(rssi));
}

}  // namespace espectre
