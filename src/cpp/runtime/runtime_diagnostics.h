/*
 * ESPectre - Runtime Diagnostics
 *
 * Runtime diagnostics snapshot helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <functional>

#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace espectre {

/**
 * Rate and link diagnostics derived from cumulative runtime counters.
 *
 * Produced by `RuntimeDiagnosticsSampler`, never by the runtime directly: the
 * runtime only exposes monotonic totals, and the rates here are what those
 * totals moved by between two periodic sensing updates.
 *
 * A zero rate means the counter did not move over the interval, and the first
 * sample after `RuntimeDiagnosticsSampler::reset()` reports zero rates because
 * it establishes the baseline. The link fields are carried through either way.
 */
struct RuntimeDiagnosticsSample {
  /** Traffic packets per second sent or observed by the active traffic source. */
  float traffic_tx_pps{0.0f};
  /** Raw CSI callbacks per second, before any capture-level validation. */
  float csi_callback_pps{0.0f};
  /** CSI packets per second accepted by the sensing pipeline. This is the rate the detector actually sees. */
  float csi_accepted_pps{0.0f};
  /** CSI packets per second rejected by capture-level validation. */
  float csi_filtered_pps{0.0f};
  /** RSSI of the current association. `INT8_MIN` when unavailable. */
  int8_t wifi_rssi_dbm{INT8_MIN};
  /** Primary channel of the current association. Zero when unavailable. */
  uint8_t wifi_channel{0U};
};

/**
 * Converts cumulative diagnostics into rates over the interval between reads.
 *
 * Call `reset()` when the owning frontend starts. Counter resets are treated
 * as a new epoch, so rearming a traffic source cannot underflow a rate.
 *
 * @code
 * // once, at frontend startup:
 * sampler.reset(controller.diagnostics(), now_ms);
 * // whenever the owning frontend already produces a sensing update:
 * latest = sampler.sample(controller.diagnostics(), now_ms);
 * @endcode
 *
 * @par Threading
 * Not synchronized, and it holds the previous read. Sample it from the task
 * that owns the runtime.
 */
class RuntimeDiagnosticsSampler {
 public:
  /**
   * Establish the baseline the next `sample()` measures against.
   *
   * @param snapshot Current cumulative counters.
   * @param now_ms Monotonic frontend clock, in milliseconds.
   */
  void reset(const RuntimeDiagnosticsSnapshot &snapshot, uint32_t now_ms);
  /**
   * Derive rates since the previous read and adopt this one as the baseline.
   *
   * The caller owns the window. Shipped frontends invoke this from their
   * existing periodic sensing update, so diagnostics do not add a timer.
   *
   * @param snapshot Current cumulative counters.
   * @param now_ms Monotonic frontend clock, in milliseconds.
   * @return Rates over the elapsed interval. The link fields are always
   *         carried through; the rates are zero when there is no baseline yet
   *         or no time has elapsed.
   */
  RuntimeDiagnosticsSample sample(const RuntimeDiagnosticsSnapshot &snapshot, uint32_t now_ms);

 private:
  RuntimeDiagnosticsSnapshot previous_{};
  uint32_t previous_ms_{0U};
  bool baseline_ready_{false};
};

using runtime_diagnostic_visitor_t = std::function<void(const char *key, const char *value)>;

void visit_runtime_diagnostics(const RuntimeConfig &config,
                               const RuntimeSnapshot &snapshot,
                               runtime_diagnostic_visitor_t visitor);

}  // namespace espectre
