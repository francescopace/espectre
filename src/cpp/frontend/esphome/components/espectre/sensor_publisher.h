/*
 * ESPectre - Sensor Publisher
 *
 * Publishes motion and movement through ESPHome sensors.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/components/sensor/sensor.h"
#include "esphome/components/binary_sensor/binary_sensor.h"
#include "base_detector.h"
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
  void publish_movement_metric(float motion_metric);

  /**
   * Check if sensors are configured
   */
  bool has_movement_sensor() const { return movement_sensor_ != nullptr; }
  bool has_motion_binary_sensor() const { return motion_binary_sensor_ != nullptr; }

 private:
  sensor::Sensor *movement_sensor_{nullptr};
  binary_sensor::BinarySensor *motion_binary_sensor_{nullptr};
};

}  // namespace espectre_component
}  // namespace esphome
