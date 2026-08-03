/*
 * ESPectre - Runtime Snapshot
 *
 * Runtime snapshot types shared by sensing status and diagnostics.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

#include "base_detector.h"
#include "csi_format.h"
#include "threshold.h"

namespace espectre {

/** How the runtime chose the subcarriers it measures on. */
enum class RuntimeSubcarrierSource {
  /** The fixed band validated for the shipped detectors. Currently the only mode. */
  FIXED_DEFAULT,
};

/**
 * Low-frequency counters and radio state used by optional diagnostic surfaces.
 *
 * This deliberately stays separate from `RuntimeSnapshot`: sensing snapshots
 * travel through the hot callback path, while diagnostics are queried on a
 * slow timer only when the frontend enables them.
 */
struct RuntimeDiagnosticsSnapshot {
  /** RSSI of the current Wi-Fi association. `INT8_MIN` when unavailable. */
  int8_t wifi_rssi_dbm{INT8_MIN};
  /** Primary channel of the current Wi-Fi association. Zero when unavailable. */
  uint8_t wifi_channel{0U};
  /** Traffic packets sent or observed by the active traffic source. */
  uint64_t traffic_packets_total{0U};
  /** Raw invocations of the ESP-IDF CSI callback. */
  uint64_t csi_callbacks_total{0U};
  /** CSI packets accepted by the sensing pipeline. */
  uint64_t csi_accepted_total{0U};
  /** CSI packets rejected by capture-level validation. */
  uint64_t csi_filtered_total{0U};
  /** Wi-Fi channel changes observed while capture was active. */
  uint32_t channel_changes_total{0U};
};

/**
 * A consistent view of the sensing state at one instant.
 *
 * Passed to every `IRuntimeListener` callback and returned by
 * `RuntimeFrontendController::snapshot()`. It is a plain value type: copy it
 * freely, and copy it if you need it past the callback that delivered it.
 *
 * Read `ready_to_publish` before anything else. The runtime keeps emitting
 * snapshots while it calibrates, and `motion_state` is not meaningful until
 * that flag is true.
 */
struct RuntimeSnapshot {
  /** Debounced motion state, after the `motion_on_hits` / `motion_off_hits` filter. */
  MotionState motion_state{MotionState::IDLE};
  /**
   * Current motion metric, on a 0..1 probability scale for both detectors.
   *
   * Comparable to `threshold`, but not comparable across detectors: Classic
   * and ML produce the number differently even though the scale matches.
   */
  float movement_metric{0.0f};
  /** Threshold `movement_metric` is compared against, on the same scale. */
  float threshold{SEGMENTATION_DEFAULT_THRESHOLD};
  // Link quality of the packets that produced `movement_metric`, carried here
  // so the shared status logger stays a formatter instead of querying the radio
  // itself at print time.
  /** RSSI of the packets behind this metric. `INT8_MIN` when unknown. */
  int8_t link_rssi_dbm{INT8_MIN};
  /** Wi-Fi channel those packets arrived on. Zero when unknown. */
  uint8_t link_channel{0};
  /** Startup calibration is running; detection results are not valid yet. */
  bool calibrating{false};
  /**
   * The runtime is calibrated, linked, and its output is safe to act on.
   *
   * Gate every user-visible publication on this. It goes false again when the
   * Wi-Fi link drops.
   */
  bool ready_to_publish{false};
  /** Threshold startup calibration settled on. Zero before it completes. */
  float startup_threshold{0.0f};
  /**
   * Active detector label: `"classic"`, `"ml"`, or `"stream"` under
   * `RuntimeProfile::STREAM`.
   *
   * Always a static string literal, so it stays valid for the process, but the
   * pointer changes when the detector changes. `parse_detection_algorithm()`
   * turns it back into a `DetectionAlgorithm`. Note these are the protocol
   * names, not `BaseDetector::get_name()`, which is capitalized for logs.
   */
  const char *detector_name{"unknown"};
  /** How `fixed_subcarriers` was chosen. */
  RuntimeSubcarrierSource subcarrier_source{RuntimeSubcarrierSource::FIXED_DEFAULT};
  /** Subcarrier indices the detector is measuring on. */
  SelectedSubcarriers fixed_subcarriers{make_default_subcarriers()};
};

}  // namespace espectre
