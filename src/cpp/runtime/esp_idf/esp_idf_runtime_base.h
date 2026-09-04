/*
 * ESPectre - ESP-IDF Runtime Base
 *
 * State and fault reporting shared by ESP-IDF runtime specializations.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <string>

#include "espectre_log.h"
#include "runtime_interface.h"
#include "runtime_performance_diagnostics.h"

namespace espectre {

class EspIdfRuntimeBase : public IEspectreRuntime {
 public:
  /**
   * @param fault_tag Log tag for runtime faults
   * @param unknown_fault_message Reported when a caller passes a null message
   */
  EspIdfRuntimeBase(const RuntimeConfig &config, const char *fault_tag,
                    const char *unknown_fault_message)
      : config_(config), fault_tag_(fault_tag), unknown_fault_message_(unknown_fault_message) {}

  RuntimeSnapshot get_snapshot() const override { return snapshot_; }
  RuntimeDiagnosticsSnapshot get_diagnostics() const override;
  RuntimeCapabilities get_capabilities() const override { return capabilities_; }
  void set_listener(IRuntimeListener *listener) override { listener_ = listener; }

 protected:
  void notify_fault_(const char *message) {
    last_fault_ = message != nullptr ? message : unknown_fault_message_;
    ESPECTRE_LOGE(fault_tag_, "Runtime fault: %s", last_fault_.c_str());
    if (listener_ != nullptr) {
      listener_->on_runtime_fault(last_fault_.c_str());
    }
  }

  RuntimeConfig config_{};
  RuntimeSnapshot snapshot_{};
  RuntimeCapabilities capabilities_{};
  IRuntimeListener *listener_{nullptr};
  RuntimePerformanceDiagnostics performance_diagnostics_;
  bool detection_timing_supported_{false};

  bool setup_complete_{false};
  bool services_armed_{true};
  bool live_telemetry_enabled_{true};
  std::string last_fault_;

 private:
  const char *fault_tag_;
  const char *unknown_fault_message_;
};

}  // namespace espectre
