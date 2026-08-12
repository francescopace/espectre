/*
 * ESPectre - Runtime Frontend Controller
 *
 * Owns runtime lifecycle and exposes a frontend-friendly control surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "runtime_frontend_controller.h"

#include "espectre_log.h"
#include "esp_idf_runtime.h"
#include "runtime_config_utils.h"
#include "stream_runtime_factory.h"

namespace espectre {

namespace {

static const char *const TAG = "espectre.runtime";

}  // namespace

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
      runtime_ = make_stream_runtime(config_);
      if (!runtime_) {
        ESP_LOGE(TAG, "Stream runtime requested but not enabled in this build");
        return false;
      }
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
  if (config_.runtime_profile == RuntimeProfile::SENSING && config_.runtime_detector_selection_enabled) {
    config_.detection_algorithm = parse_detection_algorithm(snapshot_.detector_name);
    config_.segmentation_threshold = snapshot_.threshold;
  }
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
  if (!validate_runtime_threshold_for_algorithm(threshold, config_.detection_algorithm)) {
    return false;
  }
  if (runtime_) {
    if (!runtime_->set_threshold_runtime(threshold)) {
      return false;
    }
  } else {
    snapshot_.threshold = threshold;
  }
  config_.segmentation_threshold = threshold;
  snapshot_.threshold = threshold;
  return true;
}

bool RuntimeFrontendController::set_motion_hits_runtime(uint8_t motion_on_hits, uint8_t motion_off_hits) {
  if (motion_on_hits < RUNTIME_MOTION_HITS_MIN || motion_on_hits > RUNTIME_MOTION_HITS_MAX ||
      motion_off_hits < RUNTIME_MOTION_HITS_MIN || motion_off_hits > RUNTIME_MOTION_HITS_MAX) {
    return false;
  }
  if (runtime_) {
    if (!capabilities_.supports_runtime_motion_hits_updates ||
        !runtime_->set_motion_hits_runtime(motion_on_hits, motion_off_hits)) {
      return false;
    }
  }
  config_.motion_on_hits = motion_on_hits;
  config_.motion_off_hits = motion_off_hits;
  return true;
}

bool RuntimeFrontendController::set_detection_algorithm_runtime(DetectionAlgorithm algorithm) {
  if (!runtime_detection_algorithm_valid(algorithm)) {
    return false;
  }
  if (runtime_) {
    if (!capabilities_.supports_runtime_detector_selection ||
        !runtime_->set_detection_algorithm_runtime(algorithm)) {
      return false;
    }
    snapshot_ = runtime_->get_snapshot();
  } else {
    config_.detection_algorithm = algorithm;
    config_.segmentation_threshold = runtime_default_threshold(algorithm);
    snapshot_.threshold = config_.segmentation_threshold;
    snapshot_.detector_name = detection_algorithm_name(algorithm);
  }
  config_.detection_algorithm = algorithm;
  config_.segmentation_threshold = snapshot_.threshold;
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

RuntimeDiagnosticsSnapshot RuntimeFrontendController::diagnostics() const {
  return runtime_ != nullptr ? runtime_->get_diagnostics() : RuntimeDiagnosticsSnapshot{};
}

}  // namespace espectre
