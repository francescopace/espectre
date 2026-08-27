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

#include "device_identity.h"
#include "direct_http_protocol.h"
#include "espectre_log.h"
#include "firmware_version.h"
#include "matter_surface.h"
#include "protocol_json.h"
#include "runtime_config_utils.h"
#include "runtime_time.h"
#include "sdkconfig.h"

namespace espectre {

static const char *const TAG = "espectre.matter";

MatterFrontend::MatterFrontend(IMatterBindings *bindings,
                               uint16_t endpoint_id,
                               IDirectHttpService *direct_service)
    : bindings_(bindings), endpoint_id_(endpoint_id), direct_service_(direct_service) {}

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

  update_live_telemetry_enabled_();
  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    return false;
  }

  const uint64_t device_id = runtime_.config().device_id != 0U
                                 ? runtime_.config().device_id
                                 : derive_runtime_device_id();
  const uint32_t diagnostics_now_ms = monotonic_now_ms();
  const RuntimeDiagnosticsSnapshot diagnostics = runtime_.diagnostics();
  diagnostics_sampler_.reset(diagnostics, diagnostics_now_ms);
  latest_diagnostics_ = diagnostics_sampler_.sample(diagnostics, diagnostics_now_ms);
  if (direct_service_ != nullptr && !direct_bridge_.setup(
          direct_service_,
          &runtime_,
          RuntimeDirectHttpBridgeConfig{
              "matter",
              espectre_device_name(device_id, CONFIG_IDF_TARGET),
              "",
              espectre_firmware_version(),
              CONFIG_IDF_TARGET,
              device_id,
              ESPECTRE_DIRECT_HTTP_PORT,
              true,
              false,
              [this]() {
                std::string label;
                return bindings_->get_node_label(&label) ? label : fallback_device_label_;
              },
              [this](const std::string &label, std::string *message) {
                const bool accepted = bindings_->set_node_label(label);
                if (message != nullptr) {
                  *message = accepted ? "Matter NodeLabel updated" : "Matter NodeLabel update rejected";
                }
                return accepted;
              },
              {},
              &peer_discovery_,
              [this]() { return &this->latest_diagnostics_; },
          })) {
    ESP_LOGE(TAG, "Matter Direct HTTP setup failed");
    runtime_.shutdown();
    return false;
  }

  ESP_LOGI(TAG, "Matter frontend initialized on endpoint %u", endpoint_id_);
  return true;
}

void MatterFrontend::sync_device_label() {
  (void) direct_bridge_.publish_changes(
      FrontendCommandChange::INFO | FrontendCommandChange::CONFIG);
}

void MatterFrontend::shutdown() {
  direct_bridge_.shutdown();
  runtime_.shutdown();
}

MatterFrontend::~MatterFrontend() { shutdown(); }

void MatterFrontend::loop() {
  runtime_.loop();
  direct_bridge_.loop();
  update_live_telemetry_enabled_();
}

void MatterFrontend::update_live_telemetry_enabled_() {
  const bool enabled = direct_bridge_.event_client_count() > 0U;
  if (enabled == live_telemetry_enabled_) {
    return;
  }
  live_telemetry_enabled_ = enabled;
  runtime_.set_live_telemetry_enabled(enabled);
}

void MatterFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    return;
  }

  bindings_->publish_motion(endpoint_id_, snapshot_to_motion_detected(snapshot));
  (void) direct_bridge_.publish_telemetry(snapshot);
}

void MatterFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) snapshot;
  (void) packets_received;
  latest_diagnostics_ = diagnostics_sampler_.sample(runtime_.diagnostics(), monotonic_now_ms());
}

void MatterFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  (void) snapshot;
  (void) direct_bridge_.publish_changes(FrontendCommandChange::CONFIG);
}

void MatterFrontend::on_detector_changed(const RuntimeSnapshot &snapshot) {
  (void) snapshot;
  (void) direct_bridge_.publish_changes(FrontendCommandChange::CONFIG);
}

void MatterFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) {
  (void) snapshot;
  (void) direct_bridge_.publish_changes(FrontendCommandChange::STATUS);
}

void MatterFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  (void) snapshot;
  FrontendCommandChange changes = FrontendCommandChange::STATUS;
  if (success) changes = changes | FrontendCommandChange::CONFIG;
  (void) direct_bridge_.publish_changes(changes);
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
}

void MatterFrontend::on_live_telemetry(float movement, float threshold) {
  (void) movement;
  (void) threshold;
  (void) direct_bridge_.publish_telemetry(runtime_.snapshot());
}

void MatterFrontend::on_runtime_fault(const char *message) {
  // setup() refuses a null bindings_ and the runtime only calls back once
  // setup() succeeded, so the pointer is an invariant rather than something to
  // re-test per hook; on_motion_state_changed already relies on that.
  if (message != nullptr) {
    bindings_->report_fault(message);
  }
  std::string data{"{"};
  append_json_pair(&data, "message", message != nullptr ? message : "runtime fault", true);
  data += "}";
  (void) direct_bridge_.publish_event("fault", data);
}

}  // namespace espectre
