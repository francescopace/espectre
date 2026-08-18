/*
 * ESPectre - Runtime Listener Utils
 *
 * Shared helpers for keeping frontend calibration and snapshot state in
 * sync.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "espectre_log.h"
#include "runtime_config_utils.h"
#include "runtime_frontend_controller.h"

namespace espectre {

/**
 * Record a threshold-change snapshot and mirror it into the frontend config.
 *
 * Every frontend has to do both: the snapshot drives what it publishes, and the
 * config copy is what its own accessors and diagnostics read back.
 */
inline void apply_threshold_snapshot(RuntimeFrontendController &runtime,
                                     const RuntimeSnapshot &snapshot) {
  runtime.record_snapshot(snapshot);
  runtime.config().segmentation_threshold = snapshot.threshold;
}

/** Same contract as apply_threshold_snapshot, for a detector change. */
inline void apply_detector_snapshot(RuntimeFrontendController &runtime,
                                    const RuntimeSnapshot &snapshot) {
  runtime.record_snapshot(snapshot);
  runtime.config().detection_algorithm = parse_detection_algorithm(snapshot.detector_name);
}

inline void finalize_frontend_calibration(RuntimeFrontendController &runtime,
                                          const RuntimeSnapshot &snapshot,
                                          bool success,
                                          const char *tag) {
  runtime.record_snapshot(snapshot);
  if (!success) {
    ESP_LOGW(tag, "Calibration finished without a valid update");
  }
}

}  // namespace espectre
