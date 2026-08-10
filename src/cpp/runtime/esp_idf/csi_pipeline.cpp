/*
 * ESPectre - CSI Pipeline
 *
 * Runs CSI capture, detector evaluation, and motion-state publishing for
 * sensing runtimes.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "csi_pipeline.h"
#include <algorithm>
#include <sdkconfig.h>
#include "espectre_log.h"
#include "esp_timer.h"
#include "csi_frame_identity.h"
#include "csi_phy_filter.h"

namespace espectre {

static const char *TAG = "CsiPipeline";

void CsiPipeline::init(BaseDetector* detector,
                     uint32_t publish_interval_ms,
                     IWiFiCSI* wifi_csi) {
  detector_ = detector;
  publish_interval_ms_ = std::max<uint32_t>(1U, publish_interval_ms);
  last_publish_ms_ = 0U;
  packets_processed_.store(0U, std::memory_order_relaxed);
  capture_service_.init(wifi_csi);
  capture_service_.set_packet_callback(&CsiPipeline::capture_packet_callback_, this);
  capture_service_.set_channel_change_callback(&CsiPipeline::capture_channel_change_callback_, this);
  accepted_packets_total_.store(0U, std::memory_order_relaxed);
  detector_rate_on_hold_ = false;
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

void CsiPipeline::set_motion_hit_thresholds(uint8_t motion_on_hits, uint8_t motion_off_hits, bool reset_filter) {
  set_motion_on_hits(motion_on_hits);
  set_motion_off_hits(motion_off_hits);
  if (reset_filter) {
    reset_motion_state_filter_(effective_motion_state_);
  }
}

void CsiPipeline::set_detector(BaseDetector *detector) {
  detector_ = detector;
  detector_window_event_.clear();
  clear_detector_buffer_deferred_();
  ESP_LOGD(TAG, "Detector updated to %s", detector_ != nullptr ? detector_->get_name() : "NULL");
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
    cadence_.reset_window();
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
  MotionState motion_state = MotionState::IDLE;
  if (motion_state_event_.take(motion_state) && motion_state_callback_) {
    motion_state_callback_(motion_state);
  }
  float movement = 0.0f;
  float threshold = 0.0f;
  if (live_telemetry_event_.take(movement, threshold) && live_telemetry_callback_) {
    live_telemetry_callback_(movement, threshold);
  }
  uint16_t window_packets = 0U;
  if (detector_window_event_.take(window_packets) && detector_window_callback_) {
    detector_window_callback_(window_packets);
  }
}

void CsiPipeline::publish_if_due(uint32_t now_ms) {
  if (!enabled_ || !packet_callback_) {
    return;
  }
  if (last_publish_ms_ == 0U) {
    last_publish_ms_ = now_ms;
    return;
  }
  if (now_ms - last_publish_ms_ < publish_interval_ms_) {
    return;
  }
  last_publish_ms_ = now_ms;
  const uint32_t packets_received =
      packets_processed_.exchange(0U, std::memory_order_relaxed);
  packet_callback_(heartbeat_motion_state_.load(std::memory_order_relaxed), packets_received);
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
  heartbeat_motion_state_.store(state, std::memory_order_relaxed);
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
  const CsiFormatAssessment &assessment = capture_service_.last_assessment();
  if (!assessment.is_sensing_accepted()) {
    return;
  }
  if (!csi_frame_matches_local_identity(data, local_ip_addr_, local_mac_addr_.data())) {
    return;
  }
  if (assessment.reset_detector_before_consume) {
    clear_detector_buffer_deferred_();
  }

  accepted_packets_total_.fetch_add(1U, std::memory_order_relaxed);

  const int8_t *csi_data = normalized.data;
  const size_t csi_len = normalized.len;

  const int8_t rssi_dbm = data->rx_ctrl.rssi;
  last_rssi_dbm_ = rssi_dbm;
  last_channel_ = data->rx_ctrl.channel;

  // The cadence is advanced before the interceptor, not after it. Calibration
  // consumes the packet here, and feeding the estimator only on the detection
  // path starved it for the whole ~1000-packet calibration: the two then ran on
  // different clocks, and the threshold was fitted at a resolution the detector
  // never used.
  const bool cadence_due = cadence_.observe(data->rx_ctrl.timestamp);
  if (!cadence_.detector_rate_supported()) {
    if (!detector_rate_on_hold_) {
      const MotionState previous_state = effective_motion_state_;
      detector_->reset();
      detector_->clear_buffer();
      reset_motion_state_filter_();
      request_motion_state_callback_(previous_state, effective_motion_state_);
      detector_rate_on_hold_ = true;
      ESP_LOGW(TAG, "Detector on hold: measured CSI rate is below %u pps",
               static_cast<unsigned>(DETECTOR_MIN_PACKET_RATE_PPS));
    }
    if (cadence_due) {
      cadence_.after_evaluation();
    }
    return;
  }
  if (detector_rate_on_hold_) {
    detector_->reset();
    detector_->clear_buffer();
    reset_motion_state_filter_();
    detector_rate_on_hold_ = false;
    ESP_LOGI(TAG, "Detector resumed: measured CSI rate recovered");
  }
  if (cadence_.rate_ready()) {
    const uint16_t resolved_window = cadence_.detector_window_packets();
    const uint16_t current_window = detector_->get_window_size();
    const uint16_t minimum_change =
        static_cast<uint16_t>(current_window / 20U > 4U ? current_window / 20U : 4U);
    const uint16_t difference = current_window > resolved_window
                                    ? current_window - resolved_window
                                    : resolved_window - current_window;
    if (difference >= minimum_change) {
      detector_window_event_.post(resolved_window);
    }
  }

  if (packet_interceptor_ &&
      packet_interceptor_(packet_interceptor_context_, csi_data, csi_len, rssi_dbm,
                          cadence_due, cadence_.packets_since_evaluation())) {
    if (cadence_due) {
      cadence_.after_evaluation();
    }
    return;
  }

  detector_->process_packet(csi_data, csi_len, DEFAULT_SUBCARRIERS, NUM_SUBCARRIERS, rssi_dbm);

  packets_processed_.fetch_add(1U, std::memory_order_relaxed);

  if (cadence_due) {
    // The two clock reads exist only to feed the ~10 s [telemetry] DEBUG line,
    // so they compile out with it rather than costing two timer reads per
    // evaluation in a release build.
#if CONFIG_ESPECTRE_DEBUG_TELEMETRY
    const int64_t start_us = esp_timer_get_time();
#endif
    // Update detector state on the internal cadence.
    MotionState previous_state = effective_motion_state_;
    detector_->update_state();
    MotionState current_state = update_effective_motion_state_(detector_->get_state());
    heartbeat_motion_state_.store(current_state, std::memory_order_relaxed);
    request_motion_state_callback_(previous_state, current_state);
    cadence_.after_evaluation();

#if CONFIG_ESPECTRE_DEBUG_TELEMETRY
    const int64_t elapsed_us = esp_timer_get_time() - start_us;
    detection_timing_.record(static_cast<uint32_t>(elapsed_us));
#endif

    // Emit live telemetry on each detector evaluation tick.
    if (live_telemetry_callback_) {
      live_telemetry_event_.post(detector_->get_motion_metric(), detector_->get_threshold());
    }
  }
}

bool CsiPipeline::take_detection_timing(DetectionTimingStats *stats) {
  return detection_timing_.take(stats);
}

void CsiPipeline::capture_packet_callback_(void *context,
                                           const wifi_csi_info_t *data,
                                           const NormalizedCSIPayload &normalized) {
  auto *pipeline = static_cast<CsiPipeline *>(context);
  if (pipeline != nullptr) {
    pipeline->process_normalized_packet_(data, normalized);
  }
}

void CsiPipeline::capture_channel_change_callback_(void *context,
                                                   uint8_t previous_channel,
                                                   uint8_t current_channel) {
  auto *pipeline = static_cast<CsiPipeline *>(context);
  if (pipeline != nullptr && pipeline->channel_change_callback_) {
    pipeline->channel_change_callback_(previous_channel, current_channel);
  }
}

esp_err_t CsiPipeline::enable(csi_processed_callback_t packet_callback) {
  if (enabled_) {
    return ESP_OK;
  }
  
  packet_callback_ = packet_callback;
  capture_service_.set_packet_callback(&CsiPipeline::capture_packet_callback_, this);

  esp_err_t err = capture_service_.enable();
  if (err == ESP_OK) {
    enabled_ = true;
    last_publish_ms_ = 0U;
    packets_processed_.store(0U, std::memory_order_relaxed);
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
  clear_detector_buffer_deferred_();
  motion_state_event_.clear();
  live_telemetry_event_.clear();
  detection_timing_.clear();
  packets_processed_.store(0U, std::memory_order_relaxed);
  last_publish_ms_ = 0U;
  last_rssi_dbm_ = INT8_MIN;
  last_channel_ = 0U;
  cadence_.reset();
  detector_rate_on_hold_ = false;
  reset_motion_state_filter_();
  return ESP_OK;
}

}  // namespace espectre
