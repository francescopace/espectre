/*
 * ESPectre - BLE Frontend Adapter
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "ble_frontend.h"

#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "espectre_log.h"
#include "esp_timer.h"
#include "runtime_config_utils.h"
#include "runtime_diagnostics.h"
#include "sdkconfig.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "espectre.ble";
constexpr uint32_t kDefaultTelemetryIntervalMs = 40;

}  // namespace

BleFrontend::BleFrontend(IBleBindings *bindings) : bindings_(bindings) {}

void BleFrontend::set_runtime_config(const RuntimeConfig &config) { runtime_.set_config(config); }

bool BleFrontend::setup() {
  if (bindings_ == nullptr) {
    ESP_LOGE(TAG, "BLE bindings are not configured");
    return false;
  }

  bindings_->set_connection_state_callback([this](bool connected) { this->handle_connection_state_(connected); });
  bindings_->set_control_write_callback([this](const std::string &command) { this->handle_control_command_(command); });
  if (!bindings_->setup()) {
    ESP_LOGE(TAG, "BLE bindings setup failed");
    return false;
  }

  if (!runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    bindings_->shutdown();
    return false;
  }

  telemetry_interval_ms_ = kDefaultTelemetryIntervalMs;
  last_telemetry_ms_ = 0;
  ESP_LOGI(TAG, "BLE frontend initialized");
  return true;
}

void BleFrontend::loop() {
  runtime_.loop();
}

void BleFrontend::shutdown() {
  runtime_.shutdown();
  if (bindings_ != nullptr) {
    bindings_->shutdown();
  }
  client_connected_ = false;
}

BleFrontend::~BleFrontend() { shutdown(); }

void BleFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) { runtime_.record_snapshot(snapshot); }

void BleFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) packets_received;
  runtime_.record_snapshot(snapshot);
}

void BleFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  runtime_.config().segmentation_threshold = snapshot.threshold;
  send_system_info_();
}

void BleFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) {
  runtime_.record_snapshot(snapshot);
  send_system_info_();
}

void BleFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  runtime_.record_snapshot(snapshot);
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
  send_system_info_();
}

void BleFrontend::on_live_telemetry(float movement, float threshold) {
  if (!client_connected_ || bindings_ == nullptr) {
    return;
  }

  const uint32_t now = now_ms_();
  if (now - last_telemetry_ms_ < telemetry_interval_ms_) {
    return;
  }
  last_telemetry_ms_ = now;

  uint8_t payload[sizeof(float) * 2] = {0};
  std::memcpy(payload, &movement, sizeof(float));
  std::memcpy(payload + sizeof(float), &threshold, sizeof(float));
  bindings_->publish_telemetry(payload, sizeof(payload));
}

void BleFrontend::on_runtime_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW(TAG, "Runtime fault: %s", message);
  }
  if (bindings_ != nullptr) {
    bindings_->report_fault(message);
  }
}

bool BleFrontend::handle_control_command_(const std::string &command) {
  if (command == "REQ_SYSINFO") {
    send_system_info_();
    return true;
  }
  if (command.rfind("SET_THRESHOLD:", 0) == 0) {
    const char *value_str = command.c_str() + 14;
    char *end_ptr = nullptr;
    errno = 0;
    const float threshold = strtof(value_str, &end_ptr);
    const bool parse_ok = (end_ptr != value_str) && (end_ptr != nullptr) && (*end_ptr == '\0') &&
                          (errno != ERANGE) && std::isfinite(threshold);
    if (!parse_ok || !validate_runtime_threshold(threshold)) {
      ESP_LOGW(TAG, "Invalid BLE threshold command: %s", command.c_str());
      return false;
    }
    return handle_threshold_write_(threshold);
  }

  ESP_LOGW(TAG, "Unknown BLE control command: %s", command.c_str());
  return false;
}

bool BleFrontend::handle_threshold_write_(float threshold) {
  if (!runtime_.capabilities().supports_runtime_threshold_updates) {
    ESP_LOGW(TAG, "Runtime threshold updates are not supported");
    return false;
  }

  if (!runtime_.set_threshold_runtime(threshold)) {
    return false;
  }
  send_system_info_();
  return true;
}

void BleFrontend::handle_connection_state_(bool connected) {
  client_connected_ = connected;
  if (connected) {
    last_telemetry_ms_ = 0;
    send_system_info_();
  }
}

void BleFrontend::send_system_info_() {
  if (!client_connected_ || bindings_ == nullptr) {
    return;
  }

  char line[96];

  bindings_->publish_sysinfo_line("proto_version=1");
  std::snprintf(line, sizeof(line), "chip=%s", CONFIG_IDF_TARGET);
  bindings_->publish_sysinfo_line(line);
  visit_runtime_diagnostics(runtime_.config(), runtime_.snapshot(), [this, &line](const char *key, const char *value) {
    std::snprintf(line, sizeof(line), "%s=%s", key, value);
    bindings_->publish_sysinfo_line(line);
  });
  bindings_->publish_sysinfo_line("END");
}

uint32_t BleFrontend::now_ms_() const { return static_cast<uint32_t>(esp_timer_get_time() / 1000ULL); }

}  // namespace espectre
}  // namespace esphome
