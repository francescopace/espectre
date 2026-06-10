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
#include "../../../runtime/esp_idf/esp_idf_runtime.h"

#include "esphome/core/log.h"
#include "esphome/core/application.h"
#include "esphome/core/defines.h"
#include "esphome/core/hal.h"

#include "sdkconfig.h"

namespace esphome {
namespace espectre {

void ESpectreComponent::setup() {
  ESP_LOGI(TAG, "Initializing ESPectre component...");

  this->runtime_.reset(new EspIdfRuntime(this->runtime_config_));
  this->runtime_->set_listener(this);
  if (!this->runtime_->setup()) {
    ESP_LOGE(TAG, "ESPectre runtime setup failed");
    this->mark_failed();
    return;
  }

  this->runtime_snapshot_ = this->runtime_->get_snapshot();
  this->runtime_capabilities_ = this->runtime_->get_capabilities();
  ESP_LOGI(TAG, "ESPectre initialized successfully");
}

ESpectreComponent::~ESpectreComponent() {
  if (this->runtime_) {
    this->runtime_->shutdown();
  }
}

void ESpectreComponent::loop() {
  if (this->runtime_) {
    this->runtime_->loop();
  }
}

void ESpectreComponent::set_threshold_runtime(float threshold) {
  this->runtime_config_.segmentation_threshold = threshold;
  if (this->runtime_) {
    this->runtime_->set_threshold_runtime(threshold);
  } else {
    this->runtime_snapshot_.threshold = threshold;
  }
}

void ESpectreComponent::trigger_recalibration() {
  if (this->runtime_ && this->runtime_capabilities_.supports_manual_recalibration) {
    this->runtime_->trigger_recalibration();
  }
}

void ESpectreComponent::on_motion_state_changed(const RuntimeSnapshot &snapshot) {
  if (!snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
  }
  this->runtime_snapshot_ = snapshot;
  if (snapshot.ready_to_publish) {
    this->sensor_publisher_.publish_motion_binary(snapshot.motion_state);
  }
}

void ESpectreComponent::on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) {
  if (!this->runtime_snapshot_.ready_to_publish && snapshot.ready_to_publish) {
    this->threshold_republished_ = false;
  }
  this->runtime_snapshot_ = snapshot;
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
  this->runtime_snapshot_ = snapshot;
  this->runtime_config_.segmentation_threshold = snapshot.threshold;
  if (this->threshold_number_ != nullptr) {
    this->threshold_number_->publish_state(snapshot.threshold);
  }
}

void ESpectreComponent::on_calibration_started(const RuntimeSnapshot &snapshot) {
  this->runtime_snapshot_ = snapshot;
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(true);
  }
}

void ESpectreComponent::on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) {
  this->runtime_snapshot_ = snapshot;
  if (this->calibrate_switch_ != nullptr) {
    static_cast<ESpectreCalibrateSwitch *>(this->calibrate_switch_)->set_calibrating(false);
  }
  this->sensor_publisher_.reset_rate_counter();
  if (!success) {
    ESP_LOGW(TAG, "Calibration finished without a valid update");
  }
}

void ESpectreComponent::on_runtime_fault(const char *message) {
  if (message != nullptr) {
    ESP_LOGW(TAG, "Runtime fault: %s", message);
  }
}

void ESpectreComponent::dump_config() {
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, "  _____ ____  ____           __            ");
  ESP_LOGCONFIG(TAG, " | ____/ ___||  _ \\ ___  ___| |_ _ __ ___ ");
  ESP_LOGCONFIG(TAG, " |  _| \\___ \\| |_) / _ \\/ __| __| '__/ _ \\");
  ESP_LOGCONFIG(TAG, " | |___ ___) |  __/  __/ (__| |_| | |  __/");
  ESP_LOGCONFIG(TAG, " |_____|____/|_|   \\___|\\___|\\__|_|  \\___|");
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, "      Wi-Fi CSI Motion Detection System");
  ESP_LOGCONFIG(TAG, "");
  const char *thr_mode_str = (this->runtime_config_.threshold_mode == ThresholdMode::MANUAL)
                                 ? "Manual"
                                 : (this->runtime_config_.threshold_mode == ThresholdMode::MIN) ? "Min (P100)"
                                                                                                 : "Auto (P95x1.1)";
  const char *subcarrier_source = "FIXED";
  ESP_LOGCONFIG(TAG, " MOTION DETECTION");
  ESP_LOGCONFIG(TAG, " ├─ Detector ........... %s", this->runtime_snapshot_.detector_name);
  ESP_LOGCONFIG(TAG, " ├─ Threshold .......... %.2f (%s)", this->runtime_snapshot_.threshold, thr_mode_str);
  ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", this->runtime_config_.segmentation_window_size);
  ESP_LOGCONFIG(TAG, " └─ Baseline Pxx ....... %.4f", this->runtime_snapshot_.best_pxx);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " SUBCARRIERS [%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d,%02d]",
                this->runtime_snapshot_.fixed_subcarriers[0], this->runtime_snapshot_.fixed_subcarriers[1],
                this->runtime_snapshot_.fixed_subcarriers[2], this->runtime_snapshot_.fixed_subcarriers[3],
                this->runtime_snapshot_.fixed_subcarriers[4], this->runtime_snapshot_.fixed_subcarriers[5],
                this->runtime_snapshot_.fixed_subcarriers[6], this->runtime_snapshot_.fixed_subcarriers[7],
                this->runtime_snapshot_.fixed_subcarriers[8], this->runtime_snapshot_.fixed_subcarriers[9],
                this->runtime_snapshot_.fixed_subcarriers[10], this->runtime_snapshot_.fixed_subcarriers[11]);
  ESP_LOGCONFIG(TAG, " └─ Source ............. %s", subcarrier_source);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " TRAFFIC GENERATOR");
  if (this->runtime_config_.traffic_generator_rate > 0) {
    const char *mode_str = (this->runtime_config_.traffic_generator_mode == RuntimeTrafficMode::PING) ? "ping" : "dns";
    ESP_LOGCONFIG(TAG, " ├─ Mode ............... %s", mode_str);
    ESP_LOGCONFIG(TAG, " ├─ Rate ............... %u pps", this->runtime_config_.traffic_generator_rate);
    ESP_LOGCONFIG(TAG, " └─ Status ............. %s", this->runtime_snapshot_.ready_to_publish ? "[ACTIVE]" : "[IDLE]");
  } else {
    ESP_LOGCONFIG(TAG, " └─ Mode ............... External Traffic");
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " PUBLISH INTERVAL");
  ESP_LOGCONFIG(TAG, " └─ Packets ............ %u", this->runtime_config_.publish_interval);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " EVALUATION");
  ESP_LOGCONFIG(TAG, " ├─ Interval ........... %u pkts", this->runtime_config_.evaluation_interval);
  ESP_LOGCONFIG(TAG, " └─ Hits on/off ........ %u / %u", this->runtime_config_.motion_on_hits,
                this->runtime_config_.motion_off_hits);
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " LOW-PASS FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", this->runtime_config_.lowpass_enabled ? "[ENABLED]" : "[DISABLED]");
  if (this->runtime_config_.lowpass_enabled) {
    ESP_LOGCONFIG(TAG, " └─ Cutoff ............. %.1f Hz", this->runtime_config_.lowpass_cutoff);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " HAMPEL FILTER");
  ESP_LOGCONFIG(TAG, " ├─ Status ............. %s", this->runtime_config_.hampel_enabled ? "[ENABLED]" : "[DISABLED]");
  if (this->runtime_config_.hampel_enabled) {
    ESP_LOGCONFIG(TAG, " ├─ Window ............. %d pkts", this->runtime_config_.hampel_window);
    ESP_LOGCONFIG(TAG, " └─ Threshold .......... %.1f MAD", this->runtime_config_.hampel_threshold);
  }
  ESP_LOGCONFIG(TAG, "");
  ESP_LOGCONFIG(TAG, " GAIN LOCK");
  const char *gain_mode_str = "auto";
  if (this->runtime_config_.gain_lock_mode == RuntimeGainLockMode::ENABLED) {
    gain_mode_str = "enabled";
  } else if (this->runtime_config_.gain_lock_mode == RuntimeGainLockMode::DISABLED) {
    gain_mode_str = "disabled";
  }
  ESP_LOGCONFIG(TAG, " └─ Mode ............... %s", gain_mode_str);
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
