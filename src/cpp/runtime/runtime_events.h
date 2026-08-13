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
 * Callbacks are always delivered on the caller's task, never from an interrupt
 * or the Wi-Fi driver:
 * - Sensing events (motion, periodic, live telemetry, calibration completion)
 *   originate in the CSI callback but are deferred through an internal mailbox
 *   and dispatched from `loop()`.
 * - Control-driven events (threshold, detector) fire inline on whichever task
 *   called the corresponding setter.
 *
 * Because they run on your own task, blocking is allowed: publishing over MQTT
 * or writing NVS from a callback costs loop latency, not CSI frames. Calling
 * back into the controller is allowed too, with one exception noted on
 * `on_runtime_fault()`.
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
   * The debounced motion state changed.
   *
   * Edge-triggered and already filtered by `motion_on_hits` / `motion_off_hits`,
   * so this is the hook for occupancy, relays, and notifications.
   *
   * It also fires with `MotionState::IDLE` when the Wi-Fi link drops, and that
   * call carries `ready_to_publish == false`. The shipped frontends gate on
   * that flag and therefore leave their last published value in place across a
   * disconnect; if your product would rather fail open, handle the
   * not-ready edge explicitly instead of returning early.
   *
   * @param snapshot Sensing state at the moment of the change.
   */
  virtual void on_motion_state_changed(const RuntimeSnapshot &snapshot) {}
  /**
   * Heartbeat, emitted every `RuntimeConfig::publish_interval_ms` milliseconds.
   *
   * Use it for periodic telemetry and status logging rather than polling.
   *
   * @param snapshot Current sensing state, including the metric and threshold.
   * @param packets_received CSI packets accepted since the previous heartbeat,
   *        which is the honest measure of the achieved capture rate.
   */
  virtual void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {}
  /**
   * The active threshold changed, from a control call or from calibration.
   *
   * Refresh any threshold you mirror in a UI or a published entity.
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
   * Lightweight only. ML ships a fixed threshold and completes immediately.
   */
  virtual void on_calibration_started(const RuntimeSnapshot &snapshot) {}
  /**
   * Startup calibration finished.
   *
   * @param snapshot Sensing state at completion, carrying the settled threshold.
   * @param success false when calibration was cancelled or could not settle on
   *        a threshold. The runtime keeps sensing with the configured value,
   *        so treat this as a signal to surface, not a fatal error.
   */
  virtual void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {}
  /**
   * High-rate movement stream, one call per detector evaluation.
   *
   * Intended for live views such as BLE notifications. Considerably more
   * frequent than `on_periodic_update()`; suppress it with
   * `set_live_telemetry_enabled(false)` when nothing is watching.
   *
   * @param movement Current motion metric.
   * @param threshold Threshold it is compared against, on the same scale.
   */
  virtual void on_live_telemetry(float movement, float threshold) {}
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
