/*
 * ESPectre - Runtime Events
 *
 * Runtime listener and event contracts.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#include "runtime_snapshot.h"

namespace espectre {

/**
 * Everything the runtime tells your firmware.
 *
 * Subclass it, override only what your product reacts to, and install it with
 * `RuntimeFrontendController::setup(listener)`. Every callback has an empty
 * default, so an integration that only cares about motion overrides one method.
 *
 * @par Threading and reentrancy
 * Sensing events are dispatched from `loop()`; control-driven events, such as
 * threshold writes, detector selection, and manual recalibration, fire inline
 * on the task that called the control method. No callback runs in an
 * interrupt or Wi-Fi driver context. Keep callbacks bounded and non-blocking,
 * and queue network, storage, and other blocking work for another task.
 * Calling back into the controller is allowed, except from
 * `on_runtime_fault()`. @ref integration_threading has the full contract.
 *
 * `on_threshold_changed()` covers every threshold source: a setter, a
 * calibration result, or Lightweight settled-level recovery.
 *
 * @par Snapshot lifetime
 * The `snapshot` reference is only valid for the duration of the call. Copy it
 * if you need it later.
 *
 * @par Readiness
 * Snapshots are delivered during startup calibration as well. Gate anything
 * user-visible on `RuntimeSnapshot::ready_to_publish` so you do not report
 * motion from an uncalibrated detector.
 */
class IRuntimeListener {
 public:
  virtual ~IRuntimeListener() = default;

  /**
   * Public sensing readiness changed, including warm-up and input expiry.
   *
   * RuntimeFrontendController emits this from loop(), after caching the current
   * snapshot, and once from an explicit shutdown() when sensing was ready; the
   * controller destructor does not call the listener. Publish the sensing
   * resource on both availability transitions.
   *
   * @param snapshot Current sensing state, including public readiness.
   */
  virtual void on_sensing_readiness_changed(const RuntimeSnapshot &snapshot) {}

  /**
   * The debounced motion state changed.
   *
   * Edge-triggered and already filtered by `motion_on_hits` / `motion_off_hits`,
   * so this is the hook for occupancy, relays, and notifications.
   *
   * It also fires with `MotionState::IDLE` when the Wi-Fi link drops, and that
   * call carries `ready_to_publish == false`. A frontend that gates on that
   * flag leaves its last published value in place across a disconnect; if
   * your product would rather fail open, handle the
   * not-ready edge explicitly instead of returning early.
   *
   * @param snapshot Sensing state at the moment of the change.
   */
  virtual void on_motion_state_changed(const RuntimeSnapshot &snapshot) {}
  /**
   * Heartbeat, emitted every fixed `RUNTIME_HEARTBEAT_INTERVAL_MS` milliseconds.
   *
   * Use it for status logging and diagnostics sampling rather than sensing
   * telemetry. Movement and canonical MQTT telemetry follow detector evaluation
   * through `on_live_telemetry()`.
   *
   * @param snapshot Current sensing state, including the metric and threshold.
   * @param csi_accepted CSI packets the detector processed since the previous
   *        heartbeat, after temporal admission. This is the achieved sensing
   *        rate, not the capture-validation count in
   *        `RuntimeDiagnosticsSnapshot::Csi::accepted_total`.
   */
  virtual void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t csi_accepted) {}
  /**
   * The active threshold changed, from a control call, calibration, or
   * detector-driven adaptation such as Lightweight settled-level recovery.
   *
   * Refresh any threshold you mirror in a UI or a published entity. Live
   * telemetry still carries the per-sample comparison value; this hook is the
   * control-plane notification when that value itself has moved.
   */
  virtual void on_threshold_changed(const RuntimeSnapshot &snapshot) {}
  /**
   * The active detector changed.
   *
   * Thresholds are per-detector, so `on_threshold_changed()` follows this one.
   */
  virtual void on_detector_changed(const RuntimeSnapshot &snapshot) {}
  /**
   * Startup calibration began; detection results are not valid yet.
   *
   * Lightweight only. High Accuracy keeps a fixed threshold, so it skips this
   * callback and reports `on_calibration_finished()` with `success` true at
   * once.
   */
  virtual void on_calibration_started(const RuntimeSnapshot &snapshot) {}
  /**
   * Startup calibration finished.
   *
   * The runtime releases completed threshold calibration resources before
   * notifying the listener.
   *
   * @param snapshot Sensing state at completion, carrying the applied threshold.
   * @param success false when calibration was cancelled or could not settle on
   *        a threshold. The runtime keeps sensing with the threshold in force
   *        before this calibration, which after a failed startup calibration
   *        is the configured value. Treat this as a signal to surface, not a
   *        fatal error.
   */
  virtual void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {}
  /**
   * High-rate movement stream, one call per detector evaluation.
   *
   * Frontends publish canonical telemetry and Movement Score from this hook.
   * Considerably more frequent than `on_periodic_update()`; suppress it with
   * `set_live_telemetry_enabled(false)` when nothing is watching.
   *
   * Evaluations also run during calibration and warm-up, so gate publication
   * on `ready_to_publish` as in every other sensing callback.
   *
   * @param snapshot Sensing state with this evaluation's `movement_metric`
   *        and the `threshold` it is compared against.
   */
  virtual void on_live_telemetry(const RuntimeSnapshot &snapshot) {}
  /**
   * A runtime-owned failure your firmware should surface.
   *
   * @param message Human-readable cause, valid only for this call.
   *
   * Do not drive the runtime from here beyond `shutdown()`: the fault is
   * reported from inside runtime work, and re-entering control paths from it
   * is not supported.
   */
  virtual void on_runtime_fault(const char *message) {}
};

}  // namespace espectre
