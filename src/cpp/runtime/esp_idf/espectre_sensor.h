// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.
#pragma once

#include <functional>

#include "runtime/esp_idf/runtime_frontend_controller.h"
#include "runtime/runtime_events.h"

namespace espectre {

/**
 * Arduino-style motion sensor built on `RuntimeFrontendController`.
 *
 * The Arduino library names it `ESPectre` in the global namespace. The sketch
 * owns Wi-Fi. Call `begin()` after `WiFi.begin()`, then `loop()`
 * from the Arduino `loop()`. Callbacks run on the sketch's task inside
 * `begin()`, `loop()`, and `end()`, so keep them short. The configuration
 * starts from the SDK defaults and does not restore saved controls; change
 * `config()` before `begin()`.
 *
 * When the build can see the Arduino core's logging and no `LogSink` is
 * registered, `begin()` sends ESPectre logs to the core's log output at the
 * Core Debug Level. Register a sink before `begin()` to send them elsewhere.
 */
class ESPectre : private IRuntimeListener {
 public:
  /** Receives each change of `motion()`, including false when sensing stops being ready. */
  using MotionCallback = std::function<void(bool motion)>;
  /** Receives whether sensing results can be trusted. */
  using ReadyCallback = std::function<void(bool ready)>;
  /** Receives a runtime fault description. */
  using FaultCallback = std::function<void(const char *message)>;

  ESPectre();

  /** Configuration for the next `begin()`. */
  RuntimeConfig &config() { return runtime_.config(); }
  /**
   * Start sensing. Needs the Wi-Fi station, so call it after `WiFi.begin()`.
   * Returns false when setup fails, after reporting the reason through
   * `onFault()` when it is a configuration or allocation fault; the sensor can
   * then retry.
   */
  bool begin();
  /** Advance sensing and deliver callbacks. Call it from every Arduino loop. */
  void loop() { runtime_.loop(); }
  /** Stop sensing. `begin()` can start it again. */
  void end() { runtime_.shutdown(); }

  /** True when calibration has finished and results can be published. */
  bool ready() const { return runtime_.snapshot().ready_to_publish; }
  /** True while motion is detected. Always false before `ready()`. */
  bool motion() const;
  /** Latest movement score, compared against `threshold()`. */
  float movement() const { return runtime_.snapshot().movement_metric; }
  /** Current detection threshold. */
  float threshold() const { return runtime_.snapshot().threshold; }

  /** Call `callback` when `motion()` changes. */
  void onMotion(MotionCallback callback) { motion_callback_ = std::move(callback); }
  /** Call `callback` when sensing becomes ready or stops being ready. */
  void onReady(ReadyCallback callback) { ready_callback_ = std::move(callback); }
  /** Call `callback` when the runtime reports a fault. */
  void onFault(FaultCallback callback) { fault_callback_ = std::move(callback); }

  /** Full SDK controller for thresholds, recalibration, and diagnostics. */
  RuntimeFrontendController &runtime() { return runtime_; }

 private:
  void on_sensing_readiness_changed(const RuntimeSnapshot &snapshot) override;
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_runtime_fault(const char *message) override;
  void report_motion_(bool motion);

  RuntimeFrontendController runtime_;
  MotionCallback motion_callback_;
  ReadyCallback ready_callback_;
  FaultCallback fault_callback_;
  bool reported_motion_{false};
};

}  // namespace espectre

