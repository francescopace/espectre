/*
 * ESPectre - Runtime Frontend Controller
 *
 * Owns runtime lifecycle and exposes a frontend-friendly control surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <memory>

#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace espectre {

/**
 * The recommended entry point for firmware embedding ESPectre.
 *
 * It owns the runtime backend, picks the right one from
 * `RuntimeConfig::runtime_profile`, caches the latest snapshot and the
 * discovered capabilities, and validates control calls before they reach the
 * backend. The shipped Native and Matter frontends are thin wrappers over it.
 *
 * @code
 * class ProductFrontend : public espectre::IRuntimeListener {
 *  public:
 *   bool setup() {
 *     espectre::RuntimeConfig config;
 *     config.detection_algorithm = espectre::DetectionAlgorithm::CLASSIC;
 *     runtime_.set_config(config);
 *     return runtime_.setup(this);
 *   }
 *
 *   void loop() { runtime_.loop(); }
 *
 *   void on_motion_state_changed(const espectre::RuntimeSnapshot &snapshot) override {
 *     runtime_.record_snapshot(snapshot);
 *     if (snapshot.ready_to_publish) publish(snapshot.motion_state);
 *   }
 *
 *  private:
 *   espectre::RuntimeFrontendController runtime_;
 * };
 * @endcode
 *
 * @par Lifecycle
 * `set_config()` -> `setup(listener)` -> `loop()` repeatedly -> `shutdown()`.
 * The controller is reusable after `shutdown()`: configuration survives, and
 * `set_config()` becomes effective again.
 *
 * @par Threading
 * Carries no internal locking. Run `setup()`, `loop()`, and `shutdown()` on
 * one task. See `espectre_sdk.h` for the full contract, including where
 * listener callbacks land and how to handle controls driven from a transport
 * callback.
 *
 * @par Control calls before setup
 * The setters work before `setup()` and simply update the pending
 * configuration, so a frontend can accept provisioning commands during boot
 * without special-casing the ordering.
 */
class RuntimeFrontendController {
 public:
  /**
   * Stage the configuration used by the next `setup()`.
   *
   * Ignored once setup has completed, so reconfiguring a running runtime means
   * `shutdown()` first, or the `set_*_runtime()` methods for the fields that
   * support live changes.
   */
  void set_config(const RuntimeConfig &config);
  /**
   * Mutable access to the staged configuration.
   *
   * Provided so a frontend can adjust individual fields before `setup()`
   * without rebuilding the whole struct. Writing to it after setup changes
   * only this cached copy, not the running runtime.
   */
  RuntimeConfig &config() { return config_; }
  /** Read-only view of the staged configuration. */
  const RuntimeConfig &config() const { return config_; }
  /**
   * Latest known snapshot, without querying the backend.
   *
   * Refreshed at `setup()`, by control calls, and by whatever you pass to
   * `record_snapshot()`. Use it for on-demand reads such as answering a status
   * query; use the listener callbacks to react to change.
   */
  const RuntimeSnapshot &snapshot() const { return snapshot_; }
  /**
   * Read backend counters without touching the cached sensing snapshot.
   *
   * Unlike `snapshot()`, this queries the backend on every call. Invoke it from
   * an existing periodic sensing callback, not from the hot loop. Returns a
   * zeroed snapshot before `setup()`.
   */
  RuntimeDiagnosticsSnapshot diagnostics() const;
  /**
   * What the active backend supports. Meaningful only after `setup()`.
   *
   * Gate your product surface on it rather than hardcoding: the controller
   * already refuses capability-gated calls, and this is how you avoid exposing
   * a control the runtime will reject.
   */
  const RuntimeCapabilities &capabilities() const { return capabilities_; }
  /** True between a successful `setup()` and the next `shutdown()`. */
  bool is_setup_complete() const { return setup_complete_; }

  /**
   * Create the backend, apply the configuration, and start sensing.
   *
   * Calling it twice is a no-op that returns true.
   *
   * @param listener Event sink, or `nullptr` for none. Not owned; it must
   *        outlive the controller.
   * @return false when the backend cannot start, for example a
   *         `RuntimeProfile::STREAM` config in a build without the stream
   *         runtime. On failure the backend is dropped and the controller
   *         stays un-setup, so it is safe to fix the config and retry.
   */
  bool setup(IRuntimeListener *listener);
  /**
   * Advance runtime work and deliver pending listener callbacks.
   *
   * Call it continuously from your loop task. Safe, and a no-op, before setup.
   */
  void loop();
  /** Stop sensing and release the backend. Safe before setup and to repeat. */
  void shutdown();

  /**
   * Gate runtime-owned services without tearing the runtime down.
   *
   * Sticky: the value is remembered and reapplied to the backend created by a
   * later `setup()`. Matter uses it to stay silent until commissioning.
   */
  void set_services_armed(bool armed);
  /** Enable or suppress `IRuntimeListener::on_live_telemetry()`. Also sticky. */
  void set_live_telemetry_enabled(bool enabled);
  /** Current armed state, including before setup. */
  bool services_armed() const { return services_armed_; }
  /**
   * Quiet the runtime ahead of an OTA update.
   *
   * Drops live telemetry and disarms services so the download is not competing
   * with CSI capture and traffic generation. Reverse it with
   * `set_services_armed(true)` if the update is abandoned.
   */
  void quiesce_for_ota();

  /**
   * Set the motion threshold, validating it against the active detector.
   *
   * @param threshold Value on the 0..1 metric scale.
   * @return false when out of range, or when the backend refuses it. Before
   *         setup the value is staged and returns true.
   */
  bool set_threshold_runtime(float threshold);
  /**
   * Set the hit filter.
   *
   * @param motion_on_hits Consecutive above-threshold evaluations to report
   *        motion (1..20). Higher trades latency for fewer false positives.
   * @param motion_off_hits Consecutive below-threshold evaluations to clear it
   *        (1..20).
   * @return false when either value is out of range, or when the runtime is up
   *         and does not advertise
   *         `RuntimeCapabilities::supports_runtime_motion_hits_updates`.
   */
  bool set_motion_hits_runtime(uint8_t motion_on_hits, uint8_t motion_off_hits);
  /**
   * Switch detector while running.
   *
   * The threshold follows the detector: the controller adopts the new
   * detector's threshold rather than carrying the old value across scales.
   *
   * @return false for an unknown algorithm, or when the runtime is up and does
   *         not advertise
   *         `RuntimeCapabilities::supports_runtime_detector_selection`.
   */
  bool set_detection_algorithm_runtime(DetectionAlgorithm algorithm);
  /**
   * Restart startup calibration.
   *
   * @return false before setup, or when the backend does not advertise
   *         `RuntimeCapabilities::supports_manual_recalibration`. Success only
   *         means calibration started; the outcome arrives through
   *         `IRuntimeListener::on_calibration_finished()`.
   */
  bool trigger_recalibration();
  /** True while the backend is calibrating. False before setup. */
  bool is_calibrating() const;

  /**
   * Cache a snapshot delivered to your listener.
   *
   * The controller does not intercept callbacks, so call this from them to
   * keep `snapshot()` current for code that reads state on demand.
   */
  void record_snapshot(const RuntimeSnapshot &snapshot);

 private:
  RuntimeConfig config_{};
  RuntimeSnapshot snapshot_{};
  RuntimeCapabilities capabilities_{};
  std::unique_ptr<IEspectreRuntime> runtime_;
  bool setup_complete_{false};
  bool services_armed_{true};
  bool live_telemetry_enabled_{true};
};

}  // namespace espectre
