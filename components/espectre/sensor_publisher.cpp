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
#include "esp_wifi.h"

namespace esphome {
namespace espectre {

void SensorPublisher::publish_all(const IDetector *detector,
                                  csi_motion_state_t motion_state) {
  if (!detector) {
    return;
  }
  
  // Get current values
  float motion_metric = detector->get_motion_metric();
  bool is_motion = (motion_state == CSI_STATE_MOTION);
  
  // Publish motion sensors
  if (motion_binary_sensor_) {
    motion_binary_sensor_->publish_state(is_motion);
  }
  
  if (movement_sensor_) {
    movement_sensor_->publish_state(motion_metric);
  }
}

void SensorPublisher::log_status(const char *tag,
                                 const IDetector *detector,
                                 csi_motion_state_t motion_state,
                                 uint32_t packets_per_publish) {
  if (!detector || !tag) {
    return;
  }
  
  // Get current values
  float motion_metric = detector->get_motion_metric();
  float threshold = detector->get_threshold();
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
  
  // Get WiFi info for diagnostics
  wifi_ap_record_t ap_info;
  int8_t rssi = -127;
  uint8_t channel = 0;
  if (esp_wifi_sta_get_ap_info(&ap_info) == ESP_OK) {
    rssi = ap_info.rssi;
    channel = ap_info.primary;
  }
  
  // Calculate progress
  float progress = (threshold > 0) ? (motion_metric / threshold) : 0.0f;
  int percent = (int)(progress * 100);
  
  // Log with progress bar, rate, and WiFi diagnostics
  log_progress_bar(tag, progress, 20, 15,
                   "%d%% | mvmt:%.4f thr:%.4f | %s | %u pkt/s | ch:%d rssi:%d",
                   percent, motion_metric, threshold,
                   is_motion ? "MOTION" : "IDLE",
                   rate_pps, channel, rssi);
}

}  // namespace espectre
}  // namespace esphome
