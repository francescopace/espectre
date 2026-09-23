/*
 * ESPectre - Runtime Backend
 *
 * Internal backend contract behind RuntimeFrontendController.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "runtime/raw_csi.h"
#include "runtime/runtime_capabilities.h"
#include "runtime/runtime_config.h"
#include "runtime/runtime_events.h"
#include "runtime/runtime_snapshot.h"

namespace espectre {

struct RuntimeDiagnosticsSample;

/**
 * The sensing backend owned by `RuntimeFrontendController`.
 *
 * Internal: the controller creates the ESP-IDF backend itself and forwards
 * control calls, gated on `get_capabilities()`.
 *
 * Implementations are not thread-safe. Run `setup()`, `loop()`, and
 * `shutdown()` on the task that owns the runtime, and deliver listener
 * callbacks on that task rather than from an interrupt or a driver callback.
 */
class IEspectreRuntime {
 public:
  virtual ~IEspectreRuntime() = default;

  /**
   * Bring the runtime up: radio hooks, CSI capture, detector, traffic.
   *
   * @return false if the runtime cannot sense. The caller must not call
   *         `loop()` afterwards; the controller drops the instance instead.
   */
  virtual bool setup() = 0;
  /** Stop sensing and release everything `setup()` acquired. Safe to repeat. */
  virtual void shutdown() = 0;
  /** Advance runtime work and deliver deferred listener callbacks. */
  virtual void loop() = 0;
  /**
   * Gate the runtime-owned services without tearing the runtime down.
   *
   * Disarmed, the runtime stays configured but starts no CSI capture or
   * traffic, and keeps the current Wi-Fi association. During raw collection,
   * the requested state takes effect when collection stops.
   */
  virtual void set_services_armed(bool armed) = 0;
  /** Enable or suppress the high-rate `on_live_telemetry()` stream. */
  virtual void set_live_telemetry_enabled(bool enabled) = 0;

  /** Retune the motion threshold. False when out of range or not applied. */
  virtual bool set_threshold(float threshold) = 0;
  /** Retune the hit filter. False when a count is out of range or not applied. */
  virtual bool set_motion_hits(uint8_t motion_on_hits, uint8_t motion_off_hits) = 0;
  /** Change how CSI traffic is produced, including `EXTERNAL`. False when not applied. */
  virtual bool set_traffic_generator_mode(TrafficGeneratorMode mode) = 0;
  /** Switch detector, rebuilding detector state. False when not applied. */
  virtual bool set_detection_algorithm(DetectionAlgorithm algorithm) = 0;
  /** Restart startup calibration. False when calibration cannot start. */
  virtual bool trigger_recalibration() = 0;
  /** True while startup calibration is running and detection is not yet valid. */
  virtual bool is_calibrating() const = 0;

  /**
   * Enter transient raw collection while preserving the sensing config.
   *
   * The callback runs in the CSI capture context and must remain bounded and
   * allocation-free.
   */
  virtual bool start_raw_collection(raw_csi_packet_callback_t callback, void *context) = 0;
  /** Leave raw collection and restore the previous sensing lifecycle. */
  virtual bool stop_raw_collection(RawCsiStopReason reason) = 0;
  /** Current transient operation state. */
  virtual RuntimeOperationState operation_state() const = 0;

  /** Current sensing state. */
  virtual RuntimeSnapshot get_snapshot() const = 0;
  /** Cumulative capture, traffic, and link counters for this session. */
  virtual RuntimeDiagnosticsSnapshot get_diagnostics() const = 0;
  /** Latest rate sample, owned by the runtime, or `nullptr`. */
  virtual const RuntimeDiagnosticsSample *get_diagnostics_sample() const = 0;
  /** What this backend supports. Stable after `setup()`. */
  virtual RuntimeCapabilities get_capabilities() const = 0;

  /** Install the event sink, or `nullptr` to detach. Not owned. */
  virtual void set_listener(IRuntimeListener *listener) = 0;
};

}  // namespace espectre
