/*
 * ESPectre - Main Component Implementation
 * 
 * Main ESPHome component that orchestrates all ESPectre subsystems.
 * Integrates CSI processing, calibration, and Home Assistant publishing.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "espectre.h"
#include "threshold_number.h"
#include "calibrate_switch.h"

#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include "esphome/core/defines.h"
#include "esphome/core/hal.h"

#include "espectre_banner.h"
#include "runtime_listener_utils.h"
#include "sdkconfig.h"

namespace esphome {
namespace espectre {

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");

  if (!this->runtime_.setup(this)) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    this->mark_failed();
    return;
  }

  ESP_LOGI(TAG, "ESPectre initialized successfully");
}

ESpectreComponent::~ESpectreComponent() {
  this->runtime_.shutdown();
}

void ESpectreComponent::loop() {
  this->runtime_.loop();
}

void ESpectreComponent::set_threshold_runtime(float threshold) {
  this->runtime_.set_threshold_runtime(threshold);
}

void ESpectreComponent::trigger_recalibration() {
  this->runtime_.trigger_recalibration();
}

void ESpectreComponent::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
  }
  this->runtime_.record_snapshot(snapshot);
  if (snapshot.ready_to_publish) {
    this->sensor_publisher_.publish_motion_binary(snapshot.motion_state);
  }
}

void ESpectreComponent::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  if (!this->runtime_.snapshot().ready_to_publish && snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
  }
  this->runtime_.record_snapshot(snapshot);
  if (!snapshot.ready_to_publish) {
    return;
  }

  if (!this->threshold_republished_ && this->threshold_number_ != nullptr) {
    auto *threshold_num = static_cast<ESpectreThresholdNumber *>(this->threshold_number_);
    threshold_num->republish_state();
    this->threshold_republished_ = true;
  }

  this->sensor_publisher_.log_status(TAG, snapshot, packets_received);
  this->sensor_publisher_.publish_movement_metric(snapshot.movement_metric);
}

void ESpectreComponent::on_threshold_changed(const RuntimeSnapshot &snapshot) {
  this->runtime_.record_snapshot(snapshot);
  this->runtime_.config().segmentation_threshold = snapshot.threshold;
  if (this->threshold_number_ != nullptr) {
    this->threshold_number_->publish_state(snapshot.threshold);
  }
}

void ESpectreComponent::on_calibration_started(const RuntimeSnapshot &snapshot) {
  this->runtime_.record_snapshot(snapshot);
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(true);
  }
}

void ESpectreComponent::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(false);
  }
  finalize_frontend_calibration(this->runtime_, snapshot,
                                [this]() { this->sensor_publisher_.reset_rate_counter(); },
                                success, TAG);
}

void ESpectreComponent::on_runtime_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW(TAG, "Runtime fault: %s", message);
  }
}

void ESpectreComponent::dump_config() {
  log_espectre_banner([](const char *line) { ESP_LOGCONFIG(TAG, "%s", line); });
  const RuntimeConfig &config = this->runtime_.config();
  const RuntimeSnapshot &snapshot = this->runtime_.snapshot();
  ESP_LOGCONFIG(TAG, " MOTION DETECTION");
  ESP_LOGCONFIG(TAG, " ├─ Detector ........... %s", snapshot.detector_name);
  ESP_LOGCONFIG(TAG, " ├─ Threshold .......... %.6f (%s)", snapshot.threshold, threshold_mode_display_name(config.threshold_mode));
  ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", config.segmentation_window_size);
  ESP_LOGCONFIG(TAG, " └─ Startup threshold .. %.6f", snapshot.startup_threshold);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SUBCARRIERS [%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d]",
                snapshot.fixed_subcarriers[0], snapshot.fixed_subcarriers[1],
                snapshot.fixed_subcarriers[2], snapshot.fixed_subcarriers[3],
                snapshot.fixed_subcarriers[4], snapshot.fixed_subcarriers[5],
                snapshot.fixed_subcarriers[6], snapshot.fixed_subcarriers[7],
                snapshot.fixed_subcarriers[8], snapshot.fixed_subcarriers[9],
                snapshot.fixed_subcarriers[10], snapshot.fixed_subcarriers[11]);
  ESP_LOGCONFIG(TAG, " └─ Source ............. %s", subcarrier_source_name(snapshot.subcarrier_source));
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " TRAFFIC GENERATOR");
  if (config.traffic_generator_rate > 0) {
    ESP_LOGCONFIG(TAG, " ├─ Mode ............... %s", traffic_mode_name(config.traffic_generator_mode));
    ESP_LOGCONFIG(TAG, " ├─ Rate ............... %u pps", config.traffic_generator_rate);
    ESP_LOGCONFIG(TAG, " └─ Status ............. %s", snapshot.ready_to_publish ? "[ACTIVE]" : "[IDLE]");
  } else {
    ESP_LOGCONFIG(TAG, " └─ Mode ............... External Traffic");
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " PUBLISH INTERVAL");
  ESP_LOGCONFIG(TAG, " └─ Packets ............ %u", config.publish_interval);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " EVALUATION");
  ESP_LOGCONFIG(TAG, " ├─ Interval ........... %u pkts", config.evaluation_interval);
  ESP_LOGCONFIG(TAG, " └─ Hits on/off ........ %u / %u", config.motion_on_hits, config.motion_off_hits);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " LOW-PASS FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", config.lowpass_enabled ? "[ENABLED]" : "[DISABLED]");
  if (config.lowpass_enabled) {
    ESP_LOGCONFIG(TAG, " └─ Cutoff ............. %.1f Hz", config.lowpass_cutoff);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " HAMPEL FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", config.hampel_enabled ? "[ENABLED]" : "[DISABLED]");
  if (config.hampel_enabled) {
    ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", config.hampel_window);
    ESP_LOGCONFIG(TAG, " └─ Threshold .......... %.1f MAD", config.hampel_threshold);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SENSORS");
  ESP_LOGCONFIG(TAG, " ├─ Movement ........... %s", 
                this->sensor_publisher_.has_movement_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, " └─ Motion Binary ...... %s", 
                this->sensor_publisher_.has_motion_binary_sensor() ? "[OK]" : "[--]");
  ESP_LOGCONFIG(TAG, "");
}

}  // namespace espectre
}  // namespace esphome
