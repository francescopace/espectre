/*
 * ESPectre - CSI Pipeline
 *
 * Runs CSI capture, detector evaluation, and motion-state publishing for
 * sensing runtimes.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <functional>

#include "base_detector.h"
#include "csi_capture_service.h"
#include "esp_attr.h"
#include "esp_err.h"
#include "esp_wifi.h"
#include "csi_format.h"
#include "pending_event.h"
#include "runtime_sensing_schema.h"
#include "wifi_csi_interface.h"

namespace espectre {

struct DetectionTimingStats {
  uint64_t duration_sum_us{0U};
  uint32_t samples{0U};
  uint32_t minimum_us{0U};
  uint32_t maximum_us{0U};
};

class PendingDetectionTiming {
 public:
  void record(uint32_t duration_us) {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    stats_.duration_sum_us += duration_us;
    stats_.minimum_us = stats_.samples == 0U ? duration_us : std::min(stats_.minimum_us, duration_us);
    stats_.maximum_us = std::max(stats_.maximum_us, duration_us);
    stats_.samples++;
  }

  bool take(DetectionTimingStats *stats) {
    if (stats == nullptr) {
      return false;
    }
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    if (stats_.samples == 0U) {
      return false;
    }
    *stats = stats_;
    stats_ = {};
    return true;
  }

  void clear() {
    std::lock_guard<detail::PendingEventLock> lock(lock_);
    stats_ = {};
  }

 private:
  detail::PendingEventLock lock_{};
  DetectionTimingStats stats_{};
};

// Callback type for processed CSI data
using csi_processed_callback_t = std::function<void(MotionState, uint32_t)>;

// Callback type for immediate motion-state changes
using motion_state_callback_t = std::function<void(MotionState)>;

// Callback type for live telemetry updates emitted on evaluation ticks.
using live_telemetry_callback_t = std::function<void(float movement, float threshold)>;

// Callback type for intercepting normalized CSI packets before detector processing.
using csi_packet_interceptor_t = bool (*)(void *, const int8_t *, size_t);

/**
 * CSI Pipeline
 * 
 * Manages complete CSI pipeline: hardware configuration, data processing, and motion detection.
 * Handles platform-specific differences between ESP32-C6 and ESP32-S3.
 * Orchestrates CSI packet processing and band calibration.
 */
class CsiPipeline {
 public:
  /**
   * Initialize CSI Pipeline
   * 
   * @param detector Motion detector instance (BaseDetector*)
   * @param publish_rate Number of packets before triggering callback
   * @param wifi_csi WiFi CSI interface (nullptr for real implementation)
   */
  void init(BaseDetector* detector,
            uint32_t publish_rate,
            IWiFiCSI* wifi_csi = nullptr);
  
  /**
   * Update segmentation threshold
   * 
   * @param threshold New threshold value
   */
  bool set_threshold(float threshold);
  void set_detector(BaseDetector *detector);
  void set_evaluation_interval(uint32_t interval) { evaluation_interval_ = interval > 0 ? interval : 1; }
  void set_motion_on_hits(uint8_t hits) { motion_on_hits_ = hits > 0 ? hits : 1; }
  void set_motion_off_hits(uint8_t hits) { motion_off_hits_ = hits > 0 ? hits : 1; }
  
  /**
   * Enable CSI hardware and start processing
   * 
   * @param packet_callback Callback to invoke periodically (every publish_rate packets)
   * @return ESP_OK on success
   */
  esp_err_t enable(csi_processed_callback_t packet_callback = nullptr);
  
  /**
   * Disable CSI hardware
   * 
   * @return ESP_OK on success
   */
  esp_err_t disable();

  /** Drain diagnostics and frontend notifications from the runtime loop. */
  void loop();
  
  /**
   * Process incoming CSI packet
   * 
   * Orchestrates: calibration check → processing → callbacks
   * 
   * @param data CSI packet data
   */
  void process_packet(wifi_csi_info_t* data);
  
  /**
   * Set an optional packet interceptor.
   *
   * When present, normalized CSI packets are offered to the interceptor before
   * the detector sees them. Returning true consumes the packet.
   */
  void set_packet_interceptor(csi_packet_interceptor_t interceptor, void *context = nullptr) {
    packet_interceptor_ = interceptor;
    packet_interceptor_context_ = context;
  }
  
  /**
   * Check if CSI is currently enabled
   */
  bool is_enabled() const { return enabled_; }
  uint64_t accepted_packets_total() const {
    return accepted_packets_total_.load(std::memory_order_relaxed);
  }
  uint64_t rejected_out_of_order_packets_total() const {
    return capture_service_.rejected_out_of_order_packets();
  }
  /**
   * Set callback for live telemetry updates.
   */
  void set_live_telemetry_callback(live_telemetry_callback_t callback) {
    live_telemetry_callback_ = callback;
  }
  
  /**
   * Set callback for immediate motion-state changes.
   */
  void set_motion_state_callback(motion_state_callback_t callback) {
    motion_state_callback_ = callback;
  }
  
  /**
   * Get the detector instance
   */
  BaseDetector* get_detector() { return detector_; }
  
  /**
   * Clear detector buffer (for calibration reset)
   */
  void clear_detector_buffer();
  void set_local_identity(uint32_t local_ip_addr, const uint8_t *local_mac_addr);
  bool take_detection_timing(DetectionTimingStats *stats);
  
 private:
  void process_normalized_packet_(const wifi_csi_info_t *data, const NormalizedCSIPayload &normalized);
  static void capture_packet_callback_(void *context,
                                       const wifi_csi_info_t *data,
                                       const NormalizedCSIPayload &normalized);
  void clear_detector_buffer_deferred_();
  void request_motion_state_callback_(MotionState previous_state, MotionState current_state);
  MotionState update_effective_motion_state_(MotionState detector_state);
  void reset_motion_state_filter_(MotionState state = MotionState::IDLE);
  
  bool enabled_{false};
  BaseDetector* detector_{nullptr};
  csi_packet_interceptor_t packet_interceptor_{nullptr};
  void *packet_interceptor_context_{nullptr};
  csi_processed_callback_t packet_callback_;
  motion_state_callback_t motion_state_callback_;
  live_telemetry_callback_t live_telemetry_callback_;
  uint32_t publish_rate_{100};
  uint32_t evaluation_interval_{25};
  volatile uint32_t packets_processed_{0};
  std::atomic<uint64_t> accepted_packets_total_{0U};
  uint32_t packets_since_evaluation_{0};
  uint8_t current_channel_{0};
  uint8_t motion_on_hits_{RUNTIME_MOTION_ON_HITS_DEFAULT};
  uint8_t motion_off_hits_{3};
  uint8_t pending_state_hits_{0};
  MotionState effective_motion_state_{MotionState::IDLE};
  MotionState pending_motion_state_{MotionState::IDLE};
  uint32_t local_ip_addr_{0U};
  std::array<uint8_t, 6> local_mac_addr_{};

  // Deferred notifications: posted from the CSI callback, drained by loop().
  PendingEvent<MotionState> motion_state_event_;
  PendingEvent<float, float> live_telemetry_event_;
  PendingEvent<MotionState, uint32_t> packet_publish_event_;
  PendingDetectionTiming detection_timing_;
  PendingEvent<uint8_t, uint8_t> channel_change_event_;

  CsiCaptureService capture_service_;

  static constexpr uint8_t NUM_SUBCARRIERS = HT20_SELECTED_BAND_SIZE;
};

}  // namespace espectre
