#include "runtime_frontend_controller.h"

#include "esp_idf_runtime.h"
#include "runtime_config_utils.h"
#include "stream_esp_idf_runtime.h"

namespace espectre {

void RuntimeFrontendController::set_config(const RuntimeConfig &config) {
  if (setup_complete_) {
    return;
  }
  config_ = config;
  snapshot_.threshold = config_.segmentation_threshold;
}

bool RuntimeFrontendController::setup(IRuntimeListener *listener) {
  if (setup_complete_) {
    return true;
  }

  switch (config_.runtime_profile) {
    case RuntimeProfile::STREAM:
      runtime_.reset(new StreamEspIdfRuntime(config_));
      break;
    case RuntimeProfile::SENSING:
    default:
      runtime_.reset(new EspIdfRuntime(config_));
      break;
  }
  runtime_->set_listener(listener);
  runtime_->set_services_armed(services_armed_);
  runtime_->set_live_telemetry_enabled(live_telemetry_enabled_);
  if (!runtime_->setup()) {
    runtime_.reset();
    return false;
  }

  snapshot_ = runtime_->get_snapshot();
  capabilities_ = runtime_->get_capabilities();
  setup_complete_ = true;
  return true;
}

void RuntimeFrontendController::loop() {
  if (runtime_) {
    runtime_->loop();
  }
}

void RuntimeFrontendController::shutdown() {
  if (runtime_) {
    runtime_->shutdown();
    runtime_.reset();
  }
  setup_complete_ = false;
}

void RuntimeFrontendController::set_services_armed(bool armed) {
  services_armed_ = armed;
  if (runtime_) {
    runtime_->set_services_armed(armed);
  }
}

void RuntimeFrontendController::set_live_telemetry_enabled(bool enabled) {
  live_telemetry_enabled_ = enabled;
  if (runtime_) {
    runtime_->set_live_telemetry_enabled(enabled);
  }
}

void RuntimeFrontendController::quiesce_for_ota() {
  set_live_telemetry_enabled(false);
  set_services_armed(false);
}

bool RuntimeFrontendController::set_threshold_runtime(float threshold) {
  if (!validate_runtime_threshold(threshold)) {
    return false;
  }
  if (runtime_) {
    if (!runtime_->set_threshold_runtime(threshold)) {
      return false;
    }
  } else {
    snapshot_.threshold = threshold;
  }
  set_manual_threshold(config_, threshold);
  snapshot_.threshold = threshold;
  return true;
}

bool RuntimeFrontendController::trigger_recalibration() {
  if (!capabilities_.supports_manual_recalibration || !runtime_) {
    return false;
  }
  return runtime_->trigger_recalibration();
}

bool RuntimeFrontendController::is_calibrating() const {
  return runtime_ != nullptr && runtime_->is_calibrating();
}

void RuntimeFrontendController::record_snapshot(const RuntimeSnapshot &snapshot) {
  snapshot_ = snapshot;
}

}  // namespace espectre
