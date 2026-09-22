/*
 * ESPectre - Runtime Snapshot
 *
 * Runtime snapshot types shared by sensing status and diagnostics.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "csi_capture_profile.h"
#include "core/csi_types.h"
#include "core/detector_types.h"
#include "runtime_sensing_schema.h"

namespace espectre {

/**
 * Low-frequency counters and radio state used by optional diagnostic surfaces.
 *
 * This deliberately stays separate from `RuntimeSnapshot`: sensing snapshots
 * travel through the hot callback path, while frontends query diagnostics when
 * they already handle a periodic sensing update.
 *
 * Counters are cumulative and monotonic within a session; pass them through
 * `RuntimeDiagnosticsSampler` to turn them into rates.
 */
struct RuntimeDiagnosticsSnapshot {
  /** Current Wi-Fi association. */
  struct Link {
    /** RSSI of the current Wi-Fi association. `INT8_MIN` when unavailable. */
    int8_t rssi_dbm{INT8_MIN};
    /** Primary channel of the current Wi-Fi association. Zero when unavailable. */
    uint8_t channel{0U};
  };
  /** Internal generator and station traffic counters. */
  struct Traffic {
    /** Successful internal generator sends; zero with external traffic ownership. */
    uint32_t generator_packets_total{0U};
    /** Station packets accepted by the network driver; wraps modulo 2^32. */
    uint32_t tx_packets_total{0U};
    /** Station packets delivered by the network driver; wraps modulo 2^32. */
    uint32_t rx_packets_total{0U};
  };
  /** CSI capture, validation, and temporal admission counters. */
  struct Csi {
    /** Raw invocations of the ESP-IDF CSI callback. */
    uint64_t callbacks_total{0U};
    /** CSI callbacks rejected because their packet provenance did not match. */
    uint64_t provenance_rejected_total{0U};
    /** CSI packets accepted by capture validation, before temporal admission. */
    uint64_t accepted_total{0U};
    /** CSI packets admitted to the detector's temporal grid. */
    uint64_t admitted_total{0U};
    /** CSI packets rejected by capture-level validation. */
    uint64_t filtered_total{0U};
    /** Packets rejected because the receiver reported an error. */
    uint64_t rx_error_total{0U};
    /** Packets rejected because reception ended with an error (HE-capable chips). */
    uint64_t rx_end_error_total{0U};
    /** Packets rejected because the hardware CSI estimate was invalid (HE-capable chips). */
    uint64_t invalid_estimate_total{0U};
    /** Packets rejected because hardware-invalid source pairs affect live or unknown tones. */
    uint64_t invalid_first_word_total{0U};
    /** Frames whose hardware-invalid guard pairs were zeroed without changing live tones. */
    uint64_t sanitized_first_word_total{0U};
    /** Valid CSI callbacks dropped because the callback-to-runtime queue was full. */
    uint64_t pending_frame_drops_total{0U};
    /** Empty temporal detector slots observed before admitted packets. */
    uint64_t missing_slots_total{0U};
    /** Valid packets dropped because their temporal slot was already occupied. */
    uint64_t excess_total{0U};
    /** Packets rejected because processing began after the active window. */
    uint64_t stale_total{0U};
    /** Packets rejected because their timestamp moved backwards. */
    uint64_t out_of_order_total{0U};
    /** Valid slots in the current detector window. */
    uint32_t occupancy_slots{0U};
    /** Total slots in the configured detector window. */
    uint32_t window_slots{0U};
    /** Frames currently waiting in the callback-to-runtime queue. */
    uint32_t pending_frames{0U};
    /** Fixed capacity of the callback-to-runtime queue. */
    uint32_t pending_frame_capacity{0U};
  };
  /** Heap and CPU state. */
  struct Platform {
    /** Current free heap in bytes. Zero when unavailable. */
    uint32_t free_memory_bytes{0U};
    /** Minimum free heap observed since boot, in bytes. Zero when unavailable. */
    uint32_t minimum_free_memory_bytes{0U};
    /** Largest currently allocatable heap block, in bytes. Zero when unavailable. */
    uint32_t largest_free_memory_block_bytes{0U};
    /** Resolved CPU frequency in MHz. Zero when unavailable. */
    uint32_t cpu_frequency_mhz{0U};
  };
  /** Runtime loop and detector timing over the latest complete window. */
  struct Performance {
    /** True after the first complete performance aggregation window. */
    bool window_ready{false};
    /** Duration of the latest complete performance window, in microseconds. */
    uint32_t window_duration_us{0U};
    /** Share of the window spent inside the ESPectre runtime loop. */
    float runtime_load_percent{0.0f};
    /** Runtime loop iterations measured in the latest complete window. */
    uint32_t loop_samples{0U};
    /** Mean runtime loop duration in the latest complete window. */
    uint32_t loop_average_us{0U};
    /** Maximum runtime loop duration in the latest complete window. */
    uint32_t loop_maximum_us{0U};
    /** Whether this runtime executes a detector and reports its timing. */
    bool detection_timing_supported{false};
    /** Detector evaluations measured in the latest complete window. */
    uint32_t detection_samples{0U};
    /** Total detector evaluation time in the latest complete window. */
    uint64_t detection_sum_us{0U};
    /** Mean detector evaluation time in the latest complete window. */
    uint32_t detection_average_us{0U};
    /** Minimum detector evaluation time in the latest complete window. */
    uint32_t detection_minimum_us{0U};
    /** Maximum detector evaluation time in the latest complete window. */
    uint32_t detection_maximum_us{0U};
  };
  /** Current Wi-Fi association. */
  Link link{};
  /** Internal generator and station traffic counters. */
  Traffic traffic{};
  /** CSI capture, validation, and temporal admission counters. */
  Csi csi{};
  /** Heap and CPU state. */
  Platform platform{};
  /** Runtime loop and detector timing over the latest complete window. */
  Performance performance{};
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
   * Comparable to `threshold`, but not comparable across detectors: Lightweight
   * and ML produce the number differently even though the scale matches.
   */
  float movement_metric{0.0f};
  /** Threshold `movement_metric` is compared against, on the same scale. */
  float threshold{RUNTIME_SEGMENTATION_THRESHOLD_DEFAULT};
  // Link quality of the packets that produced `movement_metric`, carried here
  // so the shared status logger stays a formatter instead of querying the radio
  // itself at print time.
  /** RSSI of the packets behind this metric. `INT8_MIN` when unknown. */
  int8_t link_rssi_dbm{INT8_MIN};
  /** Wi-Fi channel those packets arrived on. Zero when unknown. */
  uint8_t link_channel{0};
  /** Automatically selected CSI training-field and 20 MHz PHY profile. */
  CsiCaptureProfile csi_capture_profile{CsiCaptureProfile::HT20};
  /** Startup calibration is running; detection results are not valid yet. */
  bool calibrating{false};
  /**
   * Packets observed by the current Lightweight startup calibrator.
   *
   * Zero when calibration is not running. Lightweight can finish early once
   * motion evidence is accepted, so this may stay below `calibration_target_packets`.
   */
  uint32_t calibration_packets{0};
  /**
   * Packet budget for the current Lightweight startup calibrator.
   *
   * Zero when calibration is not running. `csi:`/`miss:` on the status heartbeat
   * are last-second pipeline rates, not remaining calibration work.
   */
  uint16_t calibration_target_packets{0};
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
   * Active detector label: `"lightweight"` or `"high_accuracy"`.
   *
   * Always a static string literal, so it stays valid for the process, but the
   * pointer changes when the detector changes. `parse_detection_algorithm()`
   * turns it back into a `DetectionAlgorithm`. Note these are the protocol
   * names, not `BaseDetector::get_name()`, which is capitalized for logs.
   */
  const char *detector_name{"unknown"};
};

}  // namespace espectre
