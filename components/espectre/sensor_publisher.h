/*
 * ESPectre - Sensor Publisher
 * 
 * Centralizes ESPHome sensor publishing logic.
 * Reduces code duplication and improves maintainability.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esphome/components/sensor/sensor.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "utils.h"
#include "detector_interface.h"

namespace esphome {
namespace espectre {

/**
 * Sensor Publisher
 * 
 * Manages publishing of all ESPectre sensors to ESPHome.
 * Handles both motion sensors and feature sensors.
 */
class SensorPublisher {
 public:
  // Motion sensors
  void set_movement_sensor(sensor::Sensor *sensor) { movement_sensor_ = sensor; }
  void set_motion_binary_sensor(binary_sensor::BinarySensor *sensor) { motion_binary_sensor_ = sensor; }
  
  /**
   * Publish all sensors with current values
   * 
   * @param detector Motion detector (IDetector*)
   * @param motion_state Current motion state
   */
  void publish_all(const IDetector *detector,
                   csi_motion_state_t motion_state);
  
  /**
   * Log status with progress bar
   * 
   * @param tag Log tag
   * @param detector Motion detector
   * @param motion_state Current motion state
   * @param packets_per_publish Number of packets processed per publish cycle
   */
  void log_status(const char *tag,
                  const IDetector *detector,
                  csi_motion_state_t motion_state,
                  uint32_t packets_per_publish);
  
  /**
   * Check if sensors are configured
   */
  bool has_movement_sensor() const { return movement_sensor_ != nullptr; }
  bool has_motion_binary_sensor() const { return motion_binary_sensor_ != nullptr; }
  
  /**
   * Reset rate counter
   */
  void reset_rate_counter() { last_log_time_ms_ = 0; }
  
 private:
  sensor::Sensor *movement_sensor_{nullptr};
  binary_sensor::BinarySensor *motion_binary_sensor_{nullptr};
  uint32_t last_log_time_ms_{0};
};

}  // namespace espectre
}  // namespace esphome
