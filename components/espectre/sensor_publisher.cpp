/*
 * ESPectre - Sensor Publisher Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "sensor_publisher.h"
#include "utils.h"
#include "esphome/core/log.h"
#include "esp_timer.h"

namespace esphome {
namespace espectre {

void SensorPublisher::publish_all(const csi_processor_context_t *processor,
                                  csi_motion_state_t motion_state) {
  if (!processor) {
    return;
  }
  
  // Get current values
  float moving_variance = csi_processor_get_moving_variance(processor);
  bool is_motion = (motion_state == CSI_STATE_MOTION);
  
  // Publish motion sensors
  if (motion_binary_sensor_) {
    motion_binary_sensor_->publish_state(is_motion);
  }
  
  if (movement_sensor_) {
    movement_sensor_->publish_state(moving_variance);
  }
}

void SensorPublisher::log_status(const char *tag,
                                 const csi_processor_context_t *processor,
                                 csi_motion_state_t motion_state,
                                 uint32_t packets_per_publish) {
  if (!processor || !tag) {
    return;
  }
  
  // Get current values
  float moving_variance = csi_processor_get_moving_variance(processor);
  float threshold = csi_processor_get_threshold(processor);
  bool is_motion = (motion_state == CSI_STATE_MOTION);
  
  // Calculate CSI rate (packets per second)
  uint32_t now_ms = esp_timer_get_time() / 1000;
  uint32_t rate_pps = 0;
  if (last_log_time_ms_ > 0) {
    uint32_t elapsed_ms = now_ms - last_log_time_ms_;
    if (elapsed_ms > 0) {
      rate_pps = (packets_per_publish * 1000) / elapsed_ms;
    }
  }
  last_log_time_ms_ = now_ms;
  
  // Calculate progress
  float progress = (threshold > 0) ? (moving_variance / threshold) : 0.0f;
  int percent = (int)(progress * 100);
  
  // Log with progress bar and rate
  log_progress_bar(tag, progress, 20, 15,
                   "%d%% | mvmt:%.4f thr:%.4f | %s | %u pkt/s",
                   percent, moving_variance, threshold,
                   is_motion ? "MOTION" : "IDLE",
                   rate_pps);
}

}  // namespace espectre
}  // namespace esphome
