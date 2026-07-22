/*
 * ESPectre - Sensor Publisher
 *
 * Publishes motion, movement, intensity, and periodic status updates through
 * ESPHome sensors.
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
namespace espectre_component {

using namespace ::espectre;

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
  void set_intensity_sensor(sensor::Sensor *sensor) { intensity_sensor_ = sensor; }
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
   * Publish movement relative to the current threshold as intensity percent.
   *
   * Intensity is min(200, movement / threshold * 100). Values at or above 100
   * mean the movement metric has reached or exceeded the decision threshold.
   *
   * @param movement_metric Movement metric value
   * @param threshold Current probability threshold
   */
  void publish_intensity(float movement_metric, float threshold);
  
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
  bool has_intensity_sensor() const { return intensity_sensor_ != nullptr; }
  bool has_motion_binary_sensor() const { return motion_binary_sensor_ != nullptr; }
  
  /**
   * Reset rate counter
   */
  void reset_rate_counter() { status_logger_.reset(); }
  
 private:
  sensor::Sensor *movement_sensor_{nullptr};
  sensor::Sensor *intensity_sensor_{nullptr};
  binary_sensor::BinarySensor *motion_binary_sensor_{nullptr};
  PeriodicSensingStatusLogger status_logger_{};
};

}  // namespace espectre_component
}  // namespace esphome
