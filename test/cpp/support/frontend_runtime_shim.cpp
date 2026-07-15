#include "frontend_runtime_shim.h"

#include "esp_idf_runtime.h"
#include "runtime_config_utils.h"

namespace espectre {
namespace frontend_runtime_shim {

State state{};

void reset() { state = State{}; }

}  // namespace frontend_runtime_shim

EspIdfRuntime::EspIdfRuntime(const RuntimeConfig &config)
    : config_(config),
      snapshot_(frontend_runtime_shim::state.snapshot),
      capabilities_(frontend_runtime_shim::state.capabilities),
      listener_(nullptr),
      detector_(nullptr) {
  frontend_runtime_shim::state.last_instance = this;
  capabilities_.supports_runtime_detector_selection = config.runtime_detector_selection_enabled;
  if (frontend_runtime_shim::state.snapshot.threshold == SEGMENTATION_DEFAULT_THRESHOLD) {
    snapshot_.threshold = config.segmentation_threshold;
  }
}

bool EspIdfRuntime::setup() { return frontend_runtime_shim::state.setup_result; }

void EspIdfRuntime::shutdown() { frontend_runtime_shim::state.shutdown_called = true; }

void EspIdfRuntime::loop() { frontend_runtime_shim::state.loop_calls++; }

void EspIdfRuntime::set_services_armed(bool armed) { frontend_runtime_shim::state.services_armed = armed; }

void EspIdfRuntime::set_live_telemetry_enabled(bool enabled) {
  frontend_runtime_shim::state.live_telemetry_enabled = enabled;
  frontend_runtime_shim::state.set_live_telemetry_enabled_calls++;
}

bool EspIdfRuntime::set_threshold_runtime(float threshold) {
  frontend_runtime_shim::state.set_threshold_calls++;
  frontend_runtime_shim::state.last_threshold = threshold;
  snapshot_.threshold = threshold;
  frontend_runtime_shim::state.snapshot.threshold = threshold;
  return true;
}

bool EspIdfRuntime::set_detection_algorithm_runtime(DetectionAlgorithm algorithm) {
  frontend_runtime_shim::state.set_detector_calls++;
  frontend_runtime_shim::state.last_detector = algorithm;
  snapshot_.detector_name = detection_algorithm_name(algorithm);
  snapshot_.threshold = runtime_default_threshold(algorithm);
  frontend_runtime_shim::state.snapshot = snapshot_;
  return true;
}

bool EspIdfRuntime::trigger_recalibration() {
  frontend_runtime_shim::state.trigger_recalibration_calls++;
  frontend_runtime_shim::state.calibrating = true;
  snapshot_.calibrating = true;
  frontend_runtime_shim::state.snapshot.calibrating = true;
  return true;
}

bool EspIdfRuntime::is_calibrating() const {
  return frontend_runtime_shim::state.calibrating;
}

RuntimeSnapshot EspIdfRuntime::get_snapshot() const { return snapshot_; }

RuntimeCapabilities EspIdfRuntime::get_capabilities() const {
  return capabilities_;
}

void EspIdfRuntime::set_listener(IRuntimeListener *listener) {
  listener_ = listener;
  frontend_runtime_shim::state.last_listener = listener;
}

}  // namespace espectre
