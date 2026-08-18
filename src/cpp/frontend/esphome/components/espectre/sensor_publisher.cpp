/*
 * ESPectre - Sensor Publisher
 *
 * Publishes motion, movement, and periodic status updates through ESPHome
 * sensors.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "sensor_publisher.h"

namespace esphome {
namespace espectre_component {

void SensorPublisher::publish_motion_binary(MotionState motion_state) {
  bool is_motion = (motion_state == MotionState::MOTION);
  if (motion_binary_sensor_) {
    motion_binary_sensor_->publish_state(is_motion);
  }
}

void SensorPublisher::publish_movement_metric(float motion_metric) {
  if (movement_sensor_) {
    movement_sensor_->publish_state(motion_metric);
  }
}

void SensorPublisher::log_status(const char *tag,
                                 const RuntimeSnapshot &snapshot,
                                 uint32_t packets_per_publish,
                                 const RuntimeDiagnosticsSample *diagnostics) {
  status_logger_.log_status(tag, snapshot, packets_per_publish, diagnostics);
}

}  // namespace espectre_component
}  // namespace esphome
