/*
 * ESPectre - CSI Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_manager.h"
#include "calibration_manager.h"
#include "esphome/core/log.h"
#include "esp_timer.h"

namespace esphome {
namespace espectre {

static const char *TAG = "CSIManager";

void CSIManager::init(csi_processor_context_t* processor,
                     const uint8_t selected_subcarriers[12],
                     float segmentation_threshold,
                     uint16_t segmentation_window_size,
                     uint32_t publish_rate,
                     bool hampel_enabled,
                     uint8_t hampel_window,
                     float hampel_threshold) {
  processor_ = processor;
  selected_subcarriers_ = selected_subcarriers;
  publish_rate_ = publish_rate;
  
  // Set subcarrier selection
  csi_set_subcarrier_selection(selected_subcarriers_, NUM_SUBCARRIERS);
  
  // Configure Hampel filter
  hampel_turbulence_init(&processor_->hampel_state, hampel_window, hampel_threshold, hampel_enabled);
  
  ESP_LOGD(TAG, "CSI Manager initialized (threshold: %.2f, window: %d, hampel: %s, window: %d)",
           segmentation_threshold, segmentation_window_size, hampel_enabled ? "ON" : "OFF", hampel_window);
}

void CSIManager::update_subcarrier_selection(const uint8_t subcarriers[12]) {
  selected_subcarriers_ = subcarriers;
  csi_set_subcarrier_selection(subcarriers, NUM_SUBCARRIERS);
  ESP_LOGD(TAG, "Subcarrier selection updated (%d subcarriers)", NUM_SUBCARRIERS);
}

void CSIManager::set_threshold(float threshold) {
  csi_processor_set_threshold(processor_, threshold);
  ESP_LOGD(TAG, "Threshold updated: %.2f", threshold);
}

void CSIManager::process_packet(wifi_csi_info_t* data,
                                csi_motion_state_t& motion_state) {
  if (!data || !processor_) {
    return;
  }
  
  int8_t *csi_data = data->buf;
  size_t csi_len = data->len;
  
  if (csi_len < 10) {
    ESP_LOGW(TAG, "CSI data too short: %zu bytes", csi_len);
    return;
  }
  
  // If calibration is in progress, delegate to calibration manager
  if (calibrator_ != nullptr && calibrator_->is_calibrating()) {
    calibrator_->add_packet(csi_data, csi_len);
    return;
  }
  
  // Process CSI packet
  csi_process_packet(processor_,
                    csi_data, csi_len,
                    selected_subcarriers_,
                    NUM_SUBCARRIERS);
  
  // Update motion state
  motion_state = csi_processor_get_state(processor_);
  
  // Handle periodic callback
  packets_processed_++;
  if (packets_processed_ >= publish_rate_) {
    if (packet_callback_) {
      packet_callback_(motion_state);
    }
    packets_processed_ = 0;
  }
}

// Static wrapper for ESP-IDF C callback
void CSIManager::csi_rx_callback_wrapper_(void* ctx, wifi_csi_info_t* data) {
  
  CSIManager* manager = static_cast<CSIManager*>(ctx);
  if (manager && data) {
    // Process packet directly in the manager
    csi_motion_state_t dummy_state;
    manager->process_packet(data, dummy_state);
  }
}

esp_err_t CSIManager::enable(csi_processed_callback_t packet_callback) {
  if (enabled_) {
    ESP_LOGW(TAG, "CSI already enabled");
    return ESP_OK;
  }
  
  packet_callback_ = packet_callback;
    
  // Configure platform-specific CSI settings
  esp_err_t err = configure_platform_specific_();
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to configure CSI: %s", esp_err_to_name(err));
    return err;
  }
  
  // Register internal wrapper callback
  err = esp_wifi_set_csi_rx_cb(&CSIManager::csi_rx_callback_wrapper_, this);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to set CSI callback: %s", esp_err_to_name(err));
    return err;
  }
  
  // Enable CSI
  err = esp_wifi_set_csi(true);
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
  
  esp_err_t err = esp_wifi_set_csi(false);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to disable CSI: %s", esp_err_to_name(err));
    return err;
  }
  
  enabled_ = false;
  ESP_LOGI(TAG, "CSI disabled");
  
  return ESP_OK;
}

esp_err_t CSIManager::configure_platform_specific_() {
#if CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C5
  // ESP32-C5/C6: Modern CSI API with WiFi 6 support
  wifi_csi_config_t csi_config = {
    .enable = 1,                    // Master enable (REQUIRED)
    .acquire_csi_legacy = 1,        // L-LTF from 802.11a/g (fallback for legacy routers)
    .acquire_csi_ht20 = 1,          // HT-LTF from 802.11n HT20 (PRIMARY - best SNR)
    .acquire_csi_ht40 = 0,          // HT40 disabled (less stable, enable only if router uses HT40)
    .acquire_csi_su = 1,            // HE-LTF from 802.11ax SU (WiFi 6 - better precision if supported)
    .acquire_csi_mu = 0,            // MU-MIMO disabled (rarely used in home environments)
    .acquire_csi_dcm = 0,           // DCM disabled (long-range feature, not needed)
    .acquire_csi_beamformed = 0,    // Beamformed disabled (alters channel estimation)
#if CONFIG_IDF_TARGET_ESP32C6
    .acquire_csi_he_stbc = 0,       // HE-STBC disabled (requires multiple antennas) - C6 only
#endif
    .val_scale_cfg = 0,             // Auto-scaling (0 for auto)
    .dump_ack_en = 0,               // ACK frames disabled (adds noise, not useful)
  };
#else
  // ESP32, ESP32-S2, ESP32-S3, ESP32-C3: Legacy CSI API
  wifi_csi_config_t csi_config = {
    .lltf_en = false,               // Disabled - HT-LTF only
    .htltf_en = true,               // HT-LTF only (PRIMARY - best SNR)
    .stbc_htltf2_en = false,        // Disabled for consistency
    .ltf_merge_en = false,          // No merge (only HT-LTF enabled)
    .channel_filter_en = false,     // Raw subcarriers
    .manu_scale = true,             // Manual scaling
    .shift = 4,                     // Shift=4 → values/16, calibrated to match C6 auto-scale range
  };
#endif
  
  ESP_LOGI(TAG, "Using %s CSI configuration", CONFIG_IDF_TARGET);

  return esp_wifi_set_csi_config(&csi_config);
}

}  // namespace espectre
}  // namespace esphome
