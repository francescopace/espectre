/*
 * ESPectre - Frontend Runtime Shim
 *
 * Host-side shim that exposes a configurable runtime to frontend tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace espectre {

class EspIdfRuntime;

namespace frontend_runtime_shim {

/**
 * What the real sensing runtime declares in its constructor.
 *
 * RuntimeCapabilities defaults every flag to false so a runtime has to declare
 * what it offers; the shim has to mirror the declaration rather than inherit
 * those defaults, or the frontend tests would exercise a runtime that supports
 * nothing. Detector selection stays false here because the real runtime derives
 * it from the config, and the shim does the same in its constructor.
 */
inline RuntimeCapabilities sensing_runtime_capabilities() {
  RuntimeCapabilities capabilities{};
  capabilities.supports_runtime_threshold_updates = true;
  capabilities.supports_runtime_motion_hits_updates = true;
  capabilities.supports_manual_recalibration = true;
  capabilities.supports_ble_telemetry = true;
  capabilities.supports_extended_diagnostics = true;
  capabilities.supports_traffic_control = true;
  return capabilities;
}

struct State {
  bool setup_result{true};
  RuntimeSnapshot snapshot{};
  RuntimeDiagnosticsSnapshot diagnostics{};
  RuntimeCapabilities capabilities{sensing_runtime_capabilities()};
  bool shutdown_called{false};
  int loop_calls{0};
  int set_threshold_calls{0};
  float last_threshold{0.0f};
  int set_motion_hits_calls{0};
  uint8_t last_motion_on_hits{RUNTIME_MOTION_ON_HITS_DEFAULT};
  uint8_t last_motion_off_hits{RUNTIME_MOTION_OFF_HITS_DEFAULT};
  int set_detector_calls{0};
  DetectionAlgorithm last_detector{DetectionAlgorithm::CLASSIC};
  int trigger_recalibration_calls{0};
  bool calibrating{false};
  bool services_armed{true};
  bool live_telemetry_enabled{true};
  int set_live_telemetry_enabled_calls{0};
  IRuntimeListener *last_listener{nullptr};
  EspIdfRuntime *last_instance{nullptr};
};

extern State state;

void reset();

}  // namespace frontend_runtime_shim

}  // namespace espectre
