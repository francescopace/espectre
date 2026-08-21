/*
 * ESPectre - Matter Frontend Adapter
 *
 * Bridges runtime events to the standard Matter occupancy surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "matter_frontend.h"

#include "espectre_log.h"
#include "matter_surface.h"

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
  if (!snapshot.ready_to_publish) {
    return;
  }

  bindings_->publish_motion(endpoint_id_, snapshot_to_motion_detected(snapshot));
}

void MatterFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) snapshot;
  (void) packets_received;
}

void MatterFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  (void) snapshot;
}

void MatterFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) { (void) snapshot; }

void MatterFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  (void) snapshot;
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
}

void MatterFrontend::on_live_telemetry(float movement, float threshold) {
  (void) movement;
  (void) threshold;
}

void MatterFrontend::on_runtime_fault(const char *message) {
  // setup() refuses a null bindings_ and the runtime only calls back once
  // setup() succeeded, so the pointer is an invariant rather than something to
  // re-test per hook; on_motion_state_changed already relies on that.
  if (message != nullptr) {
    bindings_->report_fault(message);
  }
}

}  // namespace espectre
