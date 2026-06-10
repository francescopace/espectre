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

#include "esp_idf_runtime.h"
#include "espectre_log.h"
#include "esp_timer.h"
#include "sdkconfig.h"

namespace esphome {
namespace espectre {

namespace {

static const char *const TAG = "espectre.ble";
constexpr uint32_t kDefaultTelemetryIntervalMs = 40;

}  // namespace

BleFrontend::BleFrontend(IBleBindings *bindings) : bindings_(bindings) {}

void BleFrontend::set_runtime_config(const RuntimeConfig &config) { runtime_config_ = config; }

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

  runtime_.reset(new EspIdfRuntime(runtime_config_));
  runtime_->set_listener(this);
  if (!runtime_->setup()) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    runtime_.reset();
    bindings_->shutdown();
    return false;
  }

  runtime_snapshot_ = runtime_->get_snapshot();
  runtime_capabilities_ = runtime_->get_capabilities();
  telemetry_interval_ms_ = kDefaultTelemetryIntervalMs;
  last_telemetry_ms_ = 0;
  setup_complete_ = true;
  ESP_LOGI(TAG, "BLE frontend initialized");
  return true;
}

void BleFrontend::loop() {
  if (runtime_) {
    runtime_->loop();
  }
}

void BleFrontend::shutdown() {
  if (runtime_) {
    runtime_->shutdown();
    runtime_.reset();
  }
  if (bindings_ != nullptr) {
    bindings_->shutdown();
  }
  setup_complete_ = false;
  client_connected_ = false;
}

BleFrontend::~BleFrontend() { shutdown(); }

void BleFrontend::on_motion_state_changed(const RuntimeSnapshot &snapshot) { runtime_snapshot_ = snapshot; }

void BleFrontend::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  (void) packets_received;
  runtime_snapshot_ = snapshot;
}

void BleFrontend::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  runtime_snapshot_ = snapshot;
  runtime_config_.segmentation_threshold = snapshot.threshold;
  send_system_info_();
}

void BleFrontend::on_calibration_started(const RuntimeSnapshot &snapshot) {
  runtime_snapshot_ = snapshot;
  send_system_info_();
}

void BleFrontend::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  runtime_snapshot_ = snapshot;
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
    if (!parse_ok || threshold < 0.0f || threshold > 10.0f) {
      ESP_LOGW(TAG, "Invalid BLE threshold command: %s", command.c_str());
      return false;
    }
    return handle_threshold_write_(threshold);
  }

  ESP_LOGW(TAG, "Unknown BLE control command: %s", command.c_str());
  return false;
}

bool BleFrontend::handle_threshold_write_(float threshold) {
  if (!runtime_capabilities_.supports_runtime_threshold_updates || runtime_ == nullptr) {
    ESP_LOGW(TAG, "Runtime threshold updates are not supported");
    return false;
  }

  runtime_config_.segmentation_threshold = threshold;
  runtime_config_.threshold_mode = ThresholdMode::MANUAL;
  if (!runtime_->set_threshold_runtime(threshold)) {
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

  const char *thr_mode = (runtime_config_.threshold_mode == ThresholdMode::MANUAL)
                             ? "manual"
                             : (runtime_config_.threshold_mode == ThresholdMode::MIN) ? "min" : "auto";
  const char *subcarrier_source = "fixed";
  char line[96];

  bindings_->publish_sysinfo_line("proto_version=1");
  std::snprintf(line, sizeof(line), "chip=%s", CONFIG_IDF_TARGET);
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "threshold=%.2f (%s)", runtime_snapshot_.threshold, thr_mode);
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "window=%d", runtime_config_.segmentation_window_size);
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "detector=%s", runtime_snapshot_.detector_name);
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "subcarriers=%s", subcarrier_source);
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "lowpass=%s", runtime_config_.lowpass_enabled ? "on" : "off");
  bindings_->publish_sysinfo_line(line);
  if (runtime_config_.lowpass_enabled) {
    std::snprintf(line, sizeof(line), "lowpass_cutoff=%.1f", runtime_config_.lowpass_cutoff);
    bindings_->publish_sysinfo_line(line);
  }
  std::snprintf(line, sizeof(line), "hampel=%s", runtime_config_.hampel_enabled ? "on" : "off");
  bindings_->publish_sysinfo_line(line);
  if (runtime_config_.hampel_enabled) {
    std::snprintf(line, sizeof(line), "hampel_window=%d", runtime_config_.hampel_window);
    bindings_->publish_sysinfo_line(line);
    std::snprintf(line, sizeof(line), "hampel_threshold=%.1f", runtime_config_.hampel_threshold);
    bindings_->publish_sysinfo_line(line);
  }
  std::snprintf(line, sizeof(line), "traffic_rate=%u", static_cast<unsigned>(runtime_config_.traffic_generator_rate));
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "publish_interval=%u", static_cast<unsigned>(runtime_config_.publish_interval));
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "evaluation_interval=%u", static_cast<unsigned>(runtime_config_.evaluation_interval));
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "motion_hits=%u/%u", runtime_config_.motion_on_hits,
                runtime_config_.motion_off_hits);
  bindings_->publish_sysinfo_line(line);
  std::snprintf(line, sizeof(line), "best_pxx=%.4f", runtime_snapshot_.best_pxx);
  bindings_->publish_sysinfo_line(line);
  bindings_->publish_sysinfo_line("END");
}

uint32_t BleFrontend::now_ms_() const { return static_cast<uint32_t>(esp_timer_get_time() / 1000ULL); }

}  // namespace espectre
}  // namespace esphome
