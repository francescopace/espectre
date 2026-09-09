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

#include <cstdarg>
#include <cstdio>

namespace espectre {

namespace {

void log_progress_bar(const char *tag, float progress, int width, int threshold_pos,
                      const char *format, ...) {
  if (!log_enabled(LogLevel::INFO, tag)) {
    return;
  }
  if (width < 1) {
    width = 1;
  } else if (width > 20) {
    width = 20;
  }
  if (threshold_pos >= width) {
    threshold_pos = width - 1;
  }

  int filled = static_cast<int>(progress * static_cast<float>(width));
  filled = (filled < 0) ? 0 : (filled > width ? width : filled);

  char bar[24];
  int index = 0;
  bar[index++] = '[';
  for (int position = 0; position < width; position++) {
    if (threshold_pos >= 0 && position == threshold_pos) {
      bar[index++] = '|';
    } else if (position < filled) {
      bar[index++] = '#';
    } else {
      bar[index++] = '-';
    }
  }
  bar[index++] = ']';
  bar[index] = '\0';

  if (format == nullptr) {
    ESPECTRE_LOGI(tag, "%s", bar);
    return;
  }

  char text[256];
  va_list args;
  va_start(args, format);
  std::vsnprintf(text, sizeof(text), format, args);
  va_end(args);
  ESPECTRE_LOGI(tag, "%s %s", bar, text);
}

}  // namespace

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

  // The legacy packet count represents admitted samples, not capture acceptance.
  // Keep the public argument without using it to invent missing diagnostics.
  (void) packets_per_publish;
  char rates[128];
  char link[40];
  if (diagnostics != nullptr) {
    std::snprintf(rates, sizeof(rates),
                  "tx:%.1f cb:%.1f accepted:%.1f hwerr:%.1f occ:%u%%",
                  static_cast<double>(diagnostics->traffic_tx_pps),
                  static_cast<double>(diagnostics->csi_callback_pps),
                  static_cast<double>(diagnostics->csi_accepted_pps),
                  static_cast<double>(diagnostics->csi_hw_error_pps),
                  static_cast<unsigned>(diagnostics->csi_occupancy_ratio * 100.0f + 0.5f));
    char channel[8] = "--";
    char rssi[8] = "--";
    if (diagnostics->wifi_channel != 0U) {
      std::snprintf(channel, sizeof(channel), "%u", static_cast<unsigned>(diagnostics->wifi_channel));
    }
    if (diagnostics->wifi_rssi_dbm != INT8_MIN) {
      std::snprintf(rssi, sizeof(rssi), "%d", static_cast<int>(diagnostics->wifi_rssi_dbm));
    }
    std::snprintf(link, sizeof(link), "ch:%s rssi:%s", channel, rssi);
  } else {
    std::snprintf(rates, sizeof(rates), "tx:-- cb:-- accepted:-- hwerr:-- occ:--%%");
    std::snprintf(link, sizeof(link), "ch:-- rssi:--");
  }
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
                     "| mvmt:%.6f thr:%.6f | CALIBRATING | %s | %s",
                     motion_metric, threshold, rates, link);
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
                   "| mvmt:%.6f thr:%.6f | %s | %s | %s",
                   motion_metric, threshold,
                   is_motion ? "MOTION" : "IDLE", rates, link);
}

}  // namespace espectre
