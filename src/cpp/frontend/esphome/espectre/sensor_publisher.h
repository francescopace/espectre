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
#include "base_detector.h"
#include "periodic_sensing_status_logger.h"
#include "runtime_snapshot.h"

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
   * Publish the motion binary sensor only.
   * 
   * @param motion_state Current motion state
   */
  void publish_motion_binary(MotionState motion_state);
  
  /**
   * Publish the movement metric only.
   *
   * @param movement_metric Movement metric value
   */
  void publish_movement_metric(float movement_metric);
  
  /**
   * Log status with progress bar
   * 
   * @param tag Log tag
   * @param snapshot Runtime snapshot
   * @param motion_state Current motion state
   * @param packets_per_publish Number of packets processed per publish cycle
   */
  void log_status(const char *tag,
                  const RuntimeSnapshot &snapshot,
                  uint32_t packets_per_publish);
  
  /**
   * Check if sensors are configured
   */
  bool has_movement_sensor() const { return movement_sensor_ != nullptr; }
  bool has_motion_binary_sensor() const { return motion_binary_sensor_ != nullptr; }
  
  /**
   * Reset rate counter
   */
  void reset_rate_counter() { status_logger_.reset(); }
  
 private:
  sensor::Sensor *movement_sensor_{nullptr};
  binary_sensor::BinarySensor *motion_binary_sensor_{nullptr};
  PeriodicSensingStatusLogger status_logger_{};
};

}  // namespace espectre
}  // namespace esphome
