/*
 * ESPectre - Matter Frontend Adapter
 *
 * Bridges runtime events to the standard Matter occupancy surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "matter_frontend.h"

#include "espectre_log.h"
#include "matter_surface.h"
#include "runtime_listener_utils.h"

namespace espectre {

static const char *const TAG = "espectre.matter";

MatterFrontend::MatterFrontend(IMatterBindings *bindings, uint16_t endpoint_id)
    : bindings_(bindings), endpoint_id_(endpoint_id) {}

void MatterFrontend::set_runtime_config(const RuntimeConfig &config) { runtime_.set_config(config); }

void MatterFrontend::set_runtime_services_armed(bool armed) {
  runtime_.set_services_armed(armed);
}

bool MatterFrontend::setup() {
  if (runtime_.is_setup_complete()) {
    return true;
  }

  if (bindings_ == nullptr) {
    ESP_LOGE(TAG, "Matter bindings are not configured");
    return false;
  }

  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    return false;
  }

  ESP_LOGI(TAG, "Matter frontend initialized on endpoint %u", endpoint_id_);
  return true;
}

void MatterFrontend::shutdown() {
  runtime_.shutdown();
}

MatterFrontend::~MatterFrontend() { shutdown(); }

void MatterFrontend::loop() {
  runtime_.loop();
}

void MatterFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  if (!snapshot.ready_to_publish) {
    return;
  }

  bindings_->publish_motion(endpoint_id_, snapshot_to_motion_detected(snapshot));
}

void MatterFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  runtime_.record_snapshot(snapshot);
  if (!snapshot.ready_to_publish) {
    return;
  }

  status_logger_.log_status(TAG, snapshot, packets_received);
}

void MatterFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  runtime_.config().segmentation_threshold = snapshot.threshold;
}

void MatterFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) { runtime_.record_snapshot(snapshot); }

void MatterFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  finalize_frontend_calibration(runtime_, snapshot, [this]() { status_logger_.reset(); }, success, TAG);
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
