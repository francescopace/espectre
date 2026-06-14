/*
 * ESPectre - Main Component
 * 
 * Main ESPHome component that orchestrates all ESPectre subsystems.
 * Integrates CSI processing, calibration, and Home Assistant publishing.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esphome/core/component.h"
#include "esphome/core/log.h"
#include "esphome/core/preferences.h"
#include "esphome/components/sensor/sensor.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "esphome/components/number/number.h"
#include "esphome/components/switch/switch.h"

#include <algorithm>
#include <memory>
#include <string>
#include <vector>

#include "sensor_publisher.h"
#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

static const char *const TAG = "espectre";

class ESpectreComponent : public Component, public IRuntimeListener {
 public:
  void setup() override;
  void loop() override;
  ~ESpectreComponent();
  void dump_config() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  // Setters for YAML configuration
  void set_segmentation_threshold(float threshold) { 
    this->runtime_config_.segmentation_threshold = threshold;
    this->runtime_config_.threshold_mode = ThresholdMode::MANUAL;
  }
  void set_threshold_mode(const std::string &mode) {
    if (mode == "min") {
      this->runtime_config_.threshold_mode = ThresholdMode::MIN;
    } else {
      this->runtime_config_.threshold_mode = ThresholdMode::AUTO;
    }
  }
  void set_segmentation_window_size(uint16_t size) { this->runtime_config_.segmentation_window_size = size; }
  void set_traffic_generator_rate(uint32_t rate) { this->runtime_config_.traffic_generator_rate = rate; }
  void set_traffic_generator_mode(const std::string &mode) { 
    this->runtime_config_.traffic_generator_mode =
        (mode == "ping") ? RuntimeTrafficMode::PING : RuntimeTrafficMode::DNS;
  }
  void set_gain_lock_mode(const std::string &mode) {
    if (mode == "enabled") {
      this->runtime_config_.gain_lock_mode = RuntimeGainLockMode::ENABLED;
    } else if (mode == "disabled") {
      this->runtime_config_.gain_lock_mode = RuntimeGainLockMode::DISABLED;
    } else {
      this->runtime_config_.gain_lock_mode = RuntimeGainLockMode::AUTO;
    }
  }
  void set_detection_algorithm(const std::string &algo) {
    if (algo == "ml") {
      this->runtime_config_.detection_algorithm = DetectionAlgorithm::ML;
    } else {
      this->runtime_config_.detection_algorithm = DetectionAlgorithm::MVS;
    }
  }
  void set_publish_interval(uint32_t interval) { this->runtime_config_.publish_interval = interval; }
  void set_evaluation_interval(uint32_t interval) { this->runtime_config_.evaluation_interval = interval; }
  void set_motion_on_hits(uint8_t hits) { this->runtime_config_.motion_on_hits = hits; }
  void set_motion_off_hits(uint8_t hits) { this->runtime_config_.motion_off_hits = hits; }
  void set_lowpass_enabled(bool enabled) { this->runtime_config_.lowpass_enabled = enabled; }
  void set_lowpass_cutoff(float cutoff) { this->runtime_config_.lowpass_cutoff = cutoff; }
  void set_hampel_enabled(bool enabled) { this->runtime_config_.hampel_enabled = enabled; }
  void set_hampel_window(uint8_t window) { this->runtime_config_.hampel_window = window; }
  void set_hampel_threshold(float threshold) { this->runtime_config_.hampel_threshold = threshold; }
  
  // Setters for ESPHome sensors (delegated to SensorPublisher)
  void set_movement_sensor(sensor::Sensor *sensor) { this->sensor_publisher_.set_movement_sensor(sensor); }
  void set_motion_binary_sensor(binary_sensor::BinarySensor *sensor) { this->sensor_publisher_.set_motion_binary_sensor(sensor); }
  
  // Setter for threshold number control
  void set_threshold_number(number::Number *num) { this->threshold_number_ = num; }
  
  // Runtime threshold adjustment (called from HA via number component)
  void set_threshold_runtime(float threshold);
  float get_threshold() const { return this->runtime_ ? this->runtime_snapshot_.threshold
                                                      : this->runtime_config_.segmentation_threshold; }
  
  // Runtime calibration trigger (called from HA via switch component)
  void trigger_recalibration();
  
  // Check if calibration is in progress
  bool is_calibrating() const { return this->runtime_ && this->runtime_->is_calibrating(); }
  
  // Setter for calibrate switch control
  void set_calibrate_switch(switch_::Switch *sw) { this->calibrate_switch_ = sw; }
  
 protected:
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) override;
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override;
  void on_runtime_fault(const char *message) override;

  RuntimeConfig runtime_config_{};
  RuntimeSnapshot runtime_snapshot_{};
  RuntimeCapabilities runtime_capabilities_{};
  std::unique_ptr<IEspectreRuntime> runtime_;

  SensorPublisher sensor_publisher_;

  // Number controls
  number::Number *threshold_number_{nullptr};
  
  // Switch controls
  switch_::Switch *calibrate_switch_{nullptr};

  bool threshold_republished_{false};
};

}  // namespace espectre
}  // namespace esphome
