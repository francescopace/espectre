/*
 * ESPectre - CSI Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_manager.h"
#include "calibrator_interface.h"
#include "gain_controller.h"
#include "esphome/core/log.h"
#include "esp_timer.h"
#include "esp_attr.h"

namespace esphome {
namespace espectre {

static const char *TAG = "CSIManager";

void CSIManager::init(IDetector* detector,
                     const uint8_t selected_subcarriers[12],
                     uint32_t publish_rate,
                     GainLockMode gain_lock_mode,
                     IWiFiCSI* wifi_csi) {
  detector_ = detector;
  selected_subcarriers_ = selected_subcarriers;
  publish_rate_ = publish_rate;
  
  // Use injected WiFi CSI interface or default real implementation
  wifi_csi_ = wifi_csi ? wifi_csi : &default_wifi_csi_;
  
  // Initialize gain controller for AGC/FFT locking
  gain_controller_.init(300, gain_lock_mode);
  
  ESP_LOGD(TAG, "CSI Manager initialized with %s detector", 
           detector_ ? detector_->get_name() : "NULL");
}

void CSIManager::update_subcarrier_selection(const uint8_t subcarriers[12]) {
  selected_subcarriers_ = subcarriers;
  ESP_LOGD(TAG, "Subcarrier selection updated (%d subcarriers)", NUM_SUBCARRIERS);
}

void CSIManager::set_threshold(float threshold) {
  if (detector_) {
    detector_->set_threshold(threshold);
    ESP_LOGD(TAG, "Threshold updated: %.2f", threshold);
  }
}

void CSIManager::clear_detector_buffer() {
  if (detector_) {
    // Reset detector state (clears buffers for all detector types)
    detector_->reset();
  }
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
  
  // Filter packets with wrong SC count (HT20 only: 64 SC = 128 bytes)
  if (csi_len != HT20_CSI_LEN) {
    packets_filtered_++;
    if (packets_filtered_ % 100 == 1) {
      ESP_LOGW(TAG, "Filtered %lu packets with wrong SC count (got %zu bytes, expected %d)",
               (unsigned long)packets_filtered_, csi_len, HT20_CSI_LEN);
    }
    return;
  }
  
  // If calibration is in progress, delegate to calibrator
  if (calibrator_ != nullptr && calibrator_->is_calibrating()) {
    calibrator_->add_packet(csi_data, csi_len);
    return;
  }
  
  // Process CSI packet through detector
  detector_->process_packet(csi_data, csi_len, selected_subcarriers_, NUM_SUBCARRIERS);
  
  // Handle periodic callback (or game mode which needs every packet)
  packets_processed_++;
  const bool should_publish = packets_processed_ >= publish_rate_;
  
  if (game_mode_callback_ || should_publish) {
    // Update detector state (lazy evaluation)
    detector_->update_state();
    
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
      }
      current_channel_ = packet_channel;
      
      if (packet_callback_) {
        MotionState state = detector_->get_state();
        csi_motion_state_t legacy_state = (state == MotionState::MOTION) ? 
                                          CSI_STATE_MOTION : CSI_STATE_IDLE;
        packet_callback_(legacy_state, packets_processed_);
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
  ESP_LOGI(TAG, "CSI disabled and callback unregistered");
  
  return ESP_OK;
}

esp_err_t CSIManager::configure_platform_specific_() {
#if CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C5
  wifi_csi_config_t csi_config = {
    .enable = 1,
    .acquire_csi_legacy = 1,
    .acquire_csi_ht20 = 1,
    .acquire_csi_ht40 = 0,
    .acquire_csi_su = 0,
    .acquire_csi_mu = 0,
    .acquire_csi_dcm = 0,
    .acquire_csi_beamformed = 0,
#if CONFIG_IDF_TARGET_ESP32C6
    .acquire_csi_he_stbc = 0,
#endif
    .val_scale_cfg = 0,
    .dump_ack_en = 0,
  };
#else
  wifi_csi_config_t csi_config = {
    .lltf_en = false,
    .htltf_en = true,
    .stbc_htltf2_en = false,
    .ltf_merge_en = false,
    .channel_filter_en = false,
    .manu_scale = false,
    .shift = 0,
  };
#endif
  
  ESP_LOGI(TAG, "Using %s CSI configuration", CONFIG_IDF_TARGET);
  return wifi_csi_->set_csi_config(&csi_config);
}

}  // namespace espectre
}  // namespace esphome
