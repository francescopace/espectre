/*
 * ESPectre - Matter Frontend Adapter
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "matter_frontend.h"

#include "esp_idf_runtime.h"
#include "espectre_log.h"

namespace esphome {
namespace espectre {

static const char *const TAG = "espectre.matter";

MatterFrontend::MatterFrontend(IMatterBindings *bindings, uint16_t endpoint_id)
    : bindings_(bindings), endpoint_id_(endpoint_id) {}

void MatterFrontend::set_runtime_config(const RuntimeConfig &config) { runtime_config_ = config; }

bool MatterFrontend::setup() {
  if (bindings_ == nullptr) {
    ESP_LOGE(TAG, "Matter bindings are not configured");
    return false;
  }

  runtime_.reset(new EspIdfRuntime(runtime_config_));
  runtime_->set_listener(this);
  if (!runtime_->setup()) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    runtime_.reset();
    return false;
  }

  runtime_snapshot_ = runtime_->get_snapshot();
  runtime_capabilities_ = runtime_->get_capabilities();
  setup_complete_ = true;
  ESP_LOGI(TAG, "Matter frontend initialized on endpoint %u", endpoint_id_);
  return true;
}

void MatterFrontend::shutdown() {
  if (runtime_) {
    runtime_->shutdown();
    runtime_.reset();
  }
  setup_complete_ = false;
}

MatterFrontend::~MatterFrontend() { shutdown(); }

void MatterFrontend::loop() {
  if (runtime_) {
    runtime_->loop();
  }
}

bool MatterFrontend::handle_threshold_write(float threshold) {
  if (!validate_matter_threshold(threshold)) {
    ESP_LOGW(TAG, "Rejected invalid threshold write: %.3f", threshold);
    return false;
  }
  if (!runtime_capabilities_.supports_runtime_threshold_updates) {
    ESP_LOGW(TAG, "Runtime threshold updates are not supported");
    return false;
  }

  runtime_config_.segmentation_threshold = threshold;
  runtime_config_.threshold_mode = ThresholdMode::MANUAL;
  if (runtime_) {
    return runtime_->set_threshold_runtime(threshold);
  }

  runtime_snapshot_.threshold = threshold;
  bindings_->publish_threshold(endpoint_id_, threshold);
  return true;
}

bool MatterFrontend::handle_recalibrate_request() {
  if (!runtime_capabilities_.supports_manual_recalibration) {
    ESP_LOGW(TAG, "Manual recalibration is not supported");
    return false;
  }
  if (!runtime_) {
    return false;
  }
  return runtime_->trigger_recalibration();
}

void MatterFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    threshold_republished_ = false;
  }
  runtime_snapshot_ = snapshot;
  if (!snapshot.ready_to_publish) {
    return;
  }

  bindings_->publish_motion(endpoint_id_, snapshot_to_motion_detected(snapshot));
}

void MatterFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  if (!runtime_snapshot_.ready_to_publish && snapshot.ready_to_publish) {
    threshold_republished_ = false;
  }
  runtime_snapshot_ = snapshot;
  if (!snapshot.ready_to_publish) {
    return;
  }

  if (!threshold_republished_) {
    bindings_->publish_threshold(endpoint_id_, snapshot.threshold);
    threshold_republished_ = true;
  }

  bindings_->publish_periodic_state(endpoint_id_, snapshot_to_periodic_state(snapshot, packets_received));
}

void MatterFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  runtime_snapshot_ = snapshot;
  runtime_config_.segmentation_threshold = snapshot.threshold;
  bindings_->publish_threshold(endpoint_id_, snapshot.threshold);
}

void MatterFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) {
  runtime_snapshot_ = snapshot;
  bindings_->publish_calibrating(endpoint_id_, true);
}

void MatterFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  runtime_snapshot_ = snapshot;
  bindings_->publish_calibrating(endpoint_id_, false);
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
}

void MatterFrontend::on_live_telemetry(float movement, float threshold) {
  (void) movement;
  (void) threshold;
}

void MatterFrontend::on_runtime_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW(TAG, "Runtime fault: %s", message);
    bindings_->report_fault(message);
  }
}

}  // namespace espectre
}  // namespace esphome
