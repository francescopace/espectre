/*
 * ESPectre - CSI Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_manager.h"
#include "espectre_log.h"
#include "esp_timer.h"

namespace esphome {
namespace espectre {

static const char *TAG = "CSIManager";

static void publish_motion_state_if_changed_(MotionState previous_state,
                                             MotionState current_state,
                                             const motion_state_callback_t &callback) {
  if (callback && previous_state != current_state) {
    callback(current_state);
  }
}

void CSIManager::init(BaseDetector* detector,
                     uint32_t publish_rate,
                     GainLockMode gain_lock_mode,
                     IWiFiCSI* wifi_csi) {
  detector_ = detector;
  publish_rate_ = publish_rate;
  capture_service_.init(gain_lock_mode, wifi_csi);
  capture_service_.set_packet_callback(
      [this](const wifi_csi_info_t *data, const NormalizedCSIPayload &normalized) {
        this->process_normalized_packet_(data, normalized);
      });
  reset_motion_state_filter_();
  
  ESP_LOGD(TAG, "CSI Manager initialized with %s detector", 
           detector_ ? detector_->get_name() : "NULL");
}

bool CSIManager::set_threshold(float threshold) {
  if (detector_ == nullptr) {
    return false;
  }
  if (!detector_->set_threshold(threshold)) {
    ESP_LOGW(TAG, "Rejected invalid threshold: %.3f", threshold);
    return false;
  }
  ESP_LOGD(TAG, "Threshold updated: %.2f", threshold);
  return true;
}

void CSIManager::clear_detector_buffer() {
  if (detector_) {
    MotionState previous_state = effective_motion_state_;
    // Cold reset: clear turbulence history and state.
    // Required after channel switch and post-calibration to avoid stale samples.
    detector_->clear_buffer();
    packets_since_evaluation_ = 0;
    reset_motion_state_filter_();
    publish_motion_state_if_changed_(previous_state, effective_motion_state_, motion_state_callback_);
  }
}

MotionState CSIManager::update_effective_motion_state_(MotionState detector_state) {
  if (detector_state == effective_motion_state_) {
    pending_motion_state_ = effective_motion_state_;
    pending_state_hits_ = 0;
    return effective_motion_state_;
  }

  if (detector_state != pending_motion_state_) {
    pending_motion_state_ = detector_state;
    pending_state_hits_ = 1;
  } else if (pending_state_hits_ < UINT8_MAX) {
    pending_state_hits_++;
  }

  uint8_t required_hits = (pending_motion_state_ == MotionState::MOTION) ? motion_on_hits_ : motion_off_hits_;
  if (pending_state_hits_ >= required_hits) {
    effective_motion_state_ = pending_motion_state_;
    pending_state_hits_ = 0;
  }

  return effective_motion_state_;
}

void CSIManager::reset_motion_state_filter_(MotionState state) {
  effective_motion_state_ = state;
  pending_motion_state_ = state;
  pending_state_hits_ = 0;
}

void CSIManager::process_packet(wifi_csi_info_t* data) {
  if (!data || !detector_) {
    return;
  }
  capture_service_.process_packet(data);
}

void CSIManager::process_normalized_packet_(const wifi_csi_info_t *data, const NormalizedCSIPayload &normalized) {
  if (data == nullptr || detector_ == nullptr || !normalized.valid()) {
    return;
  }

  packets_filtered_ = capture_service_.filtered_packets();
  int8_t *csi_data = const_cast<int8_t *>(normalized.data);
  size_t csi_len = normalized.len;

  if (packet_interceptor_ && packet_interceptor_(csi_data, csi_len)) {
    return;
  }

  const bool should_measure = (packets_total_++ % 1000 == 0);
  int64_t start_us = should_measure ? esp_timer_get_time() : 0;
  
  detector_->process_packet(csi_data, csi_len, DEFAULT_SUBCARRIERS, NUM_SUBCARRIERS);
  
  // Evaluate state on the internal cadence, but always refresh before a periodic publish.
  packets_processed_++;
  packets_since_evaluation_++;
  const bool should_publish = packets_processed_ >= publish_rate_;
  const bool should_evaluate = should_publish || packets_since_evaluation_ >= evaluation_interval_;
  
  if (should_evaluate) {
    // Update detector state on the internal cadence.
    MotionState previous_state = effective_motion_state_;
    detector_->update_state();
    MotionState current_state = update_effective_motion_state_(detector_->get_state());
    publish_motion_state_if_changed_(previous_state, current_state, motion_state_callback_);
    packets_since_evaluation_ = 0;
    
    // Log detection time periodically (every ~10 seconds at 100 pps)
    if (should_measure) {
      int64_t elapsed_us = esp_timer_get_time() - start_us;
      ESP_LOGD(TAG, "[perf] Detection time: %lld us", (long long)elapsed_us);
    }
    
    // Game mode callback: send data every packet for low-latency gameplay
    if (game_mode_callback_) {
      float movement = detector_->get_motion_metric();
      float threshold = detector_->get_threshold();
      game_mode_callback_(movement, threshold);
    }
  
    // Periodic publish callback
    if (should_publish) {
      // Detect WiFi channel changes
      uint8_t packet_channel = data->rx_ctrl.channel;
      if (current_channel_ != 0 && packet_channel != current_channel_) {
        ESP_LOGW(TAG, "WiFi channel changed: %d -> %d, resetting detection buffer",
                 current_channel_, packet_channel);
        clear_detector_buffer();
        current_state = effective_motion_state_;
      }
      current_channel_ = packet_channel;
      
      if (packet_callback_) {
        packet_callback_(current_state, packets_processed_);
      }
      packets_processed_ = 0;
    }
  }
}

esp_err_t CSIManager::enable(csi_processed_callback_t packet_callback) {
  if (enabled_) {
    ESP_LOGW(TAG, "CSI already enabled");
    return ESP_OK;
  }
  
  packet_callback_ = packet_callback;
  capture_service_.set_packet_callback(
      [this](const wifi_csi_info_t *data, const NormalizedCSIPayload &normalized) {
        this->process_normalized_packet_(data, normalized);
      });

  esp_err_t err = capture_service_.enable();
  if (err == ESP_OK) {
    enabled_ = true;
  }
  return err;
}

esp_err_t CSIManager::disable() {
  if (!enabled_) {
    return ESP_OK;
  }
  
  esp_err_t err = capture_service_.disable();
  if (err != ESP_OK) {
    return err;
  }
  
  enabled_ = false;
  packet_callback_ = nullptr;
  capture_service_.set_packet_callback({});
  packets_since_evaluation_ = 0;
  reset_motion_state_filter_();
  return ESP_OK;
}

}  // namespace espectre
}  // namespace esphome
