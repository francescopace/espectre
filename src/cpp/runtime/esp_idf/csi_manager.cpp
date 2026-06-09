/*
 * ESPectre - CSI Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_manager.h"
#include "csi_payload_normalizer.h"
#include "csi_platform_config.h"
#include "gain_controller.h"
#include "espectre_log.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include <cstring>

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

static void log_wrong_sc_packet_(const wifi_csi_info_t* data, size_t csi_len,
                                 uint32_t packets_filtered) {
  const auto &rx = data->rx_ctrl;
#if CONFIG_SOC_WIFI_HE_SUPPORT
  ESP_LOGW(TAG,
           "Filtered %lu packets with wrong SC count (got %zu bytes, expected %d) "
           "[ch=%u bb=%u est_len=%u est_vld=%u]",
           static_cast<unsigned long>(packets_filtered), csi_len, HT20_CSI_LEN,
           static_cast<unsigned>(rx.channel),
           static_cast<unsigned>(rx.cur_bb_format),
           static_cast<unsigned>(rx.rx_channel_estimate_len),
           static_cast<unsigned>(rx.rx_channel_estimate_info_vld));
#else
  ESP_LOGW(TAG,
           "Filtered %lu packets with wrong SC count (got %zu bytes, expected %d) "
           "[ch=%u sig_mode=%u cwb=%u mcs=%u]",
           static_cast<unsigned long>(packets_filtered), csi_len, HT20_CSI_LEN,
           static_cast<unsigned>(rx.channel),
           static_cast<unsigned>(rx.sig_mode),
           static_cast<unsigned>(rx.cwb),
           static_cast<unsigned>(rx.mcs));
#endif
}

void CSIManager::init(BaseDetector* detector,
                     uint32_t publish_rate,
                     GainLockMode gain_lock_mode,
                     IWiFiCSI* wifi_csi) {
  detector_ = detector;
  publish_rate_ = publish_rate;
  
  // Use injected WiFi CSI interface or default real implementation
  wifi_csi_ = wifi_csi ? wifi_csi : &default_wifi_csi_;
  
  // Initialize gain controller for AGC/FFT locking (uses median for robustness)
  gain_controller_.init(gain_lock_mode);
  reset_motion_state_filter_();
  
  ESP_LOGD(TAG, "CSI Manager initialized with %s detector", 
           detector_ ? detector_->get_name() : "NULL");
}

void CSIManager::set_threshold(float threshold) {
  if (detector_) {
    detector_->set_threshold(threshold);
    ESP_LOGD(TAG, "Threshold updated: %.2f", threshold);
  }
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
  
  int8_t *csi_data = data->buf;
  size_t csi_len = data->len;
  
  if (csi_len < 10) {
    ESP_LOGW(TAG, "CSI data too short: %zu bytes", csi_len);
    return;
  }
  
  // Process gain calibration
  if (!gain_controller_.is_locked()) {
    gain_controller_.process_packet(data);
    return;
  }
  
  int8_t csi_remapped[HT20_CSI_LEN];
  const NormalizedCSIPayload normalized =
      normalize_ht20_csi_payload(csi_data, csi_len, csi_remapped, sizeof(csi_remapped));

  static bool double_len_collapse_logged = false;
  if (!double_len_collapse_logged &&
      (normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT20 ||
       normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64)) {
    ESP_LOGI(TAG, "CSI double-length collapse active: 256->128 and/or 228->114");
    double_len_collapse_logged = true;
  }

  static bool remap_logged = false;
  if (!remap_logged &&
      (normalized.tag == NormalizedCSIPayloadTag::HT57_TO_64 ||
       normalized.tag == NormalizedCSIPayloadTag::DOUBLE_HT57_TO_64)) {
    ESP_LOGI(TAG, "CSI remap active: 57->64 SC (left_pad=4, right_pad=3)");
    remap_logged = true;
  }

  csi_data = const_cast<int8_t *>(normalized.data);
  csi_len = normalized.len;
  
  // At this point we expect 128 bytes (64 SC) for HT20. Filter packets with unexpected SC count.
  if (csi_len != HT20_CSI_LEN) {
    if (++packets_filtered_ % 100 == 1) {
      log_wrong_sc_packet_(data, csi_len, packets_filtered_);
    }
    return;
  }
  
  if (packet_interceptor_ && packet_interceptor_(csi_data, csi_len)) {
    return;
  }
  
  // Process CSI packet through detector
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

void IRAM_ATTR CSIManager::csi_rx_callback_wrapper_(void* ctx, wifi_csi_info_t* data) {
  CSIManager* manager = static_cast<CSIManager*>(ctx);
  if (manager && data) {
    manager->process_packet(data);
  }
}

esp_err_t CSIManager::enable(csi_processed_callback_t packet_callback) {
  if (enabled_) {
    ESP_LOGW(TAG, "CSI already enabled");
    return ESP_OK;
  }
  
  packet_callback_ = packet_callback;
    
  esp_err_t err = configure_platform_specific_();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to configure CSI: %s", esp_err_to_name(err));
    return err;
  }
  
  err = wifi_csi_->set_csi_rx_cb(&CSIManager::csi_rx_callback_wrapper_, this);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to set CSI callback: %s", esp_err_to_name(err));
    return err;
  }
  
  err = wifi_csi_->set_csi(true);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to enable CSI: %s", esp_err_to_name(err));
    return err;
  }
  
  enabled_ = true;
  ESP_LOGD(TAG, "CSI enabled successfully");
  
  return ESP_OK;
}

esp_err_t CSIManager::disable() {
  if (!enabled_) {
    return ESP_OK;
  }
  
  esp_err_t err = wifi_csi_->set_csi(false);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to disable CSI: %s", esp_err_to_name(err));
    return err;
  }
  
  err = wifi_csi_->set_csi_rx_cb(nullptr, nullptr);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to unregister CSI callback: %s", esp_err_to_name(err));
    return err;
  }
  
  enabled_ = false;
  packet_callback_ = nullptr;
  motion_state_callback_ = nullptr;
  packets_since_evaluation_ = 0;
  reset_motion_state_filter_();
  ESP_LOGI(TAG, "CSI disabled and callback unregistered");
  
  return ESP_OK;
}

esp_err_t CSIManager::configure_platform_specific_() {
  ESP_LOGI(TAG, "Using %s CSI configuration", CONFIG_IDF_TARGET);
  return configure_ht20_csi(wifi_csi_);
}

}  // namespace espectre
}  // namespace esphome
