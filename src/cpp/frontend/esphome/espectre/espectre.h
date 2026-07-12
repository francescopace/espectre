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
#include <string>
#include <vector>

#include "sensor_publisher.h"
#include "runtime_config_utils.h"
#include "runtime_events.h"
#include "runtime_frontend_controller.h"

namespace esphome {
namespace espectre_component {

using namespace ::espectre;

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
    set_manual_threshold(this->runtime_.config(), threshold);
  }
  void set_threshold_mode(const std::string &mode) {
    this->runtime_.config().threshold_mode = parse_threshold_mode(mode.c_str());
  }
  void set_segmentation_window_size(uint16_t size) { this->runtime_.config().segmentation_window_size = size; }
  void set_traffic_generator_rate(uint32_t rate) { this->runtime_.config().traffic_generator_rate = rate; }
  void set_traffic_generator_mode(const std::string &mode) { 
    this->runtime_.config().traffic_generator_mode = parse_traffic_mode(mode.c_str());
  }
  void set_detection_algorithm(const std::string &algo) {
    this->runtime_.config().detection_algorithm = parse_detection_algorithm(algo.c_str());
  }
  void set_publish_interval(uint32_t interval) { this->runtime_.config().publish_interval = interval; }
  void set_evaluation_interval(uint32_t interval) { this->runtime_.config().evaluation_interval = interval; }
  void set_motion_on_hits(uint8_t hits) { this->runtime_.config().motion_on_hits = hits; }
  void set_motion_off_hits(uint8_t hits) { this->runtime_.config().motion_off_hits = hits; }
  void set_lowpass_enabled(bool enabled) { this->runtime_.config().lowpass_enabled = enabled; }
  void set_lowpass_cutoff(float cutoff) { this->runtime_.config().lowpass_cutoff = cutoff; }
  void set_hampel_enabled(bool enabled) { this->runtime_.config().hampel_enabled = enabled; }
  void set_hampel_window(uint8_t window) { this->runtime_.config().hampel_window = window; }
  void set_hampel_threshold(float threshold) { this->runtime_.config().hampel_threshold = threshold; }
  
  // Setters for ESPHome sensors (delegated to SensorPublisher)
  void set_movement_sensor(sensor::Sensor *sensor) { this->sensor_publisher_.set_movement_sensor(sensor); }
  void set_motion_binary_sensor(binary_sensor::BinarySensor *sensor) { this->sensor_publisher_.set_motion_binary_sensor(sensor); }
  
  // Setter for threshold number control
  void set_threshold_number(number::Number *num) { this->threshold_number_ = num; }
  
  // Runtime threshold adjustment (called from HA via number component)
  void set_threshold_runtime(float threshold);
  float get_threshold() const { return this->runtime_.snapshot().threshold; }
  
  // Runtime calibration trigger (called from HA via switch component)
  void trigger_recalibration();
  
  // Check if calibration is in progress
  bool is_calibrating() const { return this->runtime_.is_calibrating(); }
  
  // Setter for calibrate switch control
  void set_calibrate_switch(switch_::Switch *sw) { this->calibrate_switch_ = sw; }
  
 protected:
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) override;
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override;
  void on_runtime_fault(const char *message) override;

  RuntimeFrontendController runtime_;

  SensorPublisher sensor_publisher_;

  // Number controls
  number::Number *threshold_number_{nullptr};
  
  // Switch controls
  switch_::Switch *calibrate_switch_{nullptr};

  bool threshold_republished_{false};
};

}  // namespace espectre_component
}  // namespace esphome
