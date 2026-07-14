/*
 * ESPectre - CSI Pipeline Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_pipeline.h"
#include <algorithm>
#include "espectre_log.h"
#include "esp_timer.h"
#include "csi_frame_identity.h"

namespace espectre {

static const char *TAG = "CsiPipeline";

void CsiPipeline::init(BaseDetector* detector,
                     uint32_t publish_rate,
                     IWiFiCSI* wifi_csi) {
  detector_ = detector;
  publish_rate_ = publish_rate;
  capture_service_.init(wifi_csi);
  capture_service_.set_packet_callback(&CsiPipeline::capture_packet_callback_, this);
  reset_motion_state_filter_();
  
  ESP_LOGD(TAG, "CSI Pipeline initialized with %s detector", 
           detector_ ? detector_->get_name() : "NULL");
}

bool CsiPipeline::set_threshold(float threshold) {
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

void CsiPipeline::clear_detector_buffer() {
  clear_detector_buffer_deferred_();
  loop();
}

void CsiPipeline::clear_detector_buffer_deferred_() {
  if (detector_) {
    MotionState previous_state = effective_motion_state_;
    // Cold reset: clear turbulence history and state.
    // Required after channel switch and post-calibration to avoid stale samples.
    detector_->clear_buffer();
    packets_since_evaluation_ = 0;
    reset_motion_state_filter_();
    request_motion_state_callback_(previous_state, effective_motion_state_);
  }
}

void CsiPipeline::request_motion_state_callback_(MotionState previous_state, MotionState current_state) {
  if (previous_state == current_state) {
    return;
  }
  motion_state_event_.post(current_state);
}

void CsiPipeline::loop() {
  capture_service_.loop();

  uint32_t detection_time_us = 0U;
  if (perf_log_event_.take(detection_time_us)) {
    ESP_LOGD(TAG, "[perf] Detection time: %u us", static_cast<unsigned>(detection_time_us));
  }
  uint8_t previous_channel = 0U;
  uint8_t current_channel = 0U;
  if (channel_change_event_.take(previous_channel, current_channel)) {
    ESP_LOGW(TAG, "WiFi channel changed: %u -> %u, resetting detection buffer",
             static_cast<unsigned>(previous_channel), static_cast<unsigned>(current_channel));
  }
  MotionState motion_state = MotionState::IDLE;
  if (motion_state_event_.take(motion_state) && motion_state_callback_) {
    motion_state_callback_(motion_state);
  }
  float movement = 0.0f;
  float threshold = 0.0f;
  if (live_telemetry_event_.take(movement, threshold) && live_telemetry_callback_) {
    live_telemetry_callback_(movement, threshold);
  }
  MotionState publish_state = MotionState::IDLE;
  uint32_t publish_count = 0U;
  if (packet_publish_event_.take(publish_state, publish_count) && packet_callback_) {
    packet_callback_(publish_state, publish_count);
  }
}

void CsiPipeline::set_local_identity(uint32_t local_ip_addr, const uint8_t *local_mac_addr) {
  local_ip_addr_ = local_ip_addr;
  local_mac_addr_.fill(0U);
  if (local_mac_addr != nullptr) {
    std::copy(local_mac_addr, local_mac_addr + local_mac_addr_.size(), local_mac_addr_.begin());
  }
}

MotionState CsiPipeline::update_effective_motion_state_(MotionState detector_state) {
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

void CsiPipeline::reset_motion_state_filter_(MotionState state) {
  effective_motion_state_ = state;
  pending_motion_state_ = state;
  pending_state_hits_ = 0;
}

void CsiPipeline::process_packet(wifi_csi_info_t* data) {
  if (!data || !detector_) {
    return;
  }
  capture_service_.process_packet(data);
  // Direct callers are synchronous test/host paths. The hardware callback
  // enters through CsiCaptureService and is drained by the runtime loop.
  loop();
}

void CsiPipeline::process_normalized_packet_(const wifi_csi_info_t *data, const NormalizedCSIPayload &normalized) {
  if (data == nullptr || detector_ == nullptr || !normalized.valid()) {
    return;
  }
  if (!csi_frame_matches_local_identity(data, local_ip_addr_, local_mac_addr_.data())) {
    return;
  }

  const int8_t *csi_data = normalized.data;
  const size_t csi_len = normalized.len;

  if (packet_interceptor_ && packet_interceptor_(packet_interceptor_context_, csi_data, csi_len)) {
    return;
  }

  const bool should_measure = (packets_total_++ % 1000 == 0);
  int64_t start_us = should_measure ? esp_timer_get_time() : 0;
  
  detector_->process_packet(csi_data, csi_len, DEFAULT_SUBCARRIERS, NUM_SUBCARRIERS);
  
  // Evaluate state on the internal cadence, but always refresh before a periodic publish.
  const uint32_t processed_count = packets_processed_ + 1U;
  packets_processed_ = processed_count;
  packets_since_evaluation_++;
  const bool should_publish = processed_count >= publish_rate_;
  const bool should_evaluate = should_publish || packets_since_evaluation_ >= evaluation_interval_;
  
  if (should_evaluate) {
    // Update detector state on the internal cadence.
    MotionState previous_state = effective_motion_state_;
    detector_->update_state();
    MotionState current_state = update_effective_motion_state_(detector_->get_state());
    request_motion_state_callback_(previous_state, current_state);
    packets_since_evaluation_ = 0;
    
    // Log detection time periodically (every ~10 seconds at 100 pps)
    if (should_measure) {
      int64_t elapsed_us = esp_timer_get_time() - start_us;
      perf_log_event_.post(static_cast<uint32_t>(elapsed_us));
    }

    // Emit live telemetry on each detector evaluation tick.
    if (live_telemetry_callback_) {
      live_telemetry_event_.post(detector_->get_motion_metric(), detector_->get_threshold());
    }
  
    // Periodic publish callback
    if (should_publish) {
      // Detect WiFi channel changes
      uint8_t packet_channel = data->rx_ctrl.channel;
      if (current_channel_ != 0 && packet_channel != current_channel_) {
        channel_change_event_.post(current_channel_, packet_channel);
        clear_detector_buffer_deferred_();
        current_state = effective_motion_state_;
      }
      current_channel_ = packet_channel;

      if (packet_callback_) {
        packet_publish_event_.post(current_state, packets_processed_);
      }
      packets_processed_ = 0;
    }
  }
}

void CsiPipeline::capture_packet_callback_(void *context,
                                           const wifi_csi_info_t *data,
                                           const NormalizedCSIPayload &normalized) {
  auto *pipeline = static_cast<CsiPipeline *>(context);
  if (pipeline != nullptr) {
    pipeline->process_normalized_packet_(data, normalized);
  }
}

esp_err_t CsiPipeline::enable(csi_processed_callback_t packet_callback) {
  if (enabled_) {
    ESP_LOGW(TAG, "CSI already enabled");
    return ESP_OK;
  }
  
  packet_callback_ = packet_callback;
  capture_service_.set_packet_callback(&CsiPipeline::capture_packet_callback_, this);

  esp_err_t err = capture_service_.enable();
  if (err == ESP_OK) {
    enabled_ = true;
  }
  return err;
}

esp_err_t CsiPipeline::disable() {
  if (!enabled_) {
    return ESP_OK;
  }
  
  esp_err_t err = capture_service_.disable();
  if (err != ESP_OK) {
    return err;
  }
  
  enabled_ = false;
  packet_callback_ = nullptr;
  capture_service_.set_packet_callback(nullptr, nullptr);
  motion_state_event_.clear();
  live_telemetry_event_.clear();
  packet_publish_event_.clear();
  perf_log_event_.clear();
  channel_change_event_.clear();
  packets_since_evaluation_ = 0;
  reset_motion_state_filter_();
  return ESP_OK;
}

}  // namespace espectre
