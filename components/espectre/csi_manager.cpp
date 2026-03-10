/*
 * ESPectre - CSI Manager Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "csi_manager.h"
#include "nbvi_calibrator.h"
#include "gain_controller.h"
#include "esphome/core/log.h"
#include "esp_timer.h"
#include "esp_attr.h"
#include <cstring>

namespace esphome {
namespace espectre {

static const char *TAG = "CSIManager";

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
                     const uint8_t selected_subcarriers[12],
                     uint32_t publish_rate,
                     GainLockMode gain_lock_mode,
                     IWiFiCSI* wifi_csi) {
  detector_ = detector;
  selected_subcarriers_ = selected_subcarriers;
  publish_rate_ = publish_rate;
  
  // Use injected WiFi CSI interface or default real implementation
  wifi_csi_ = wifi_csi ? wifi_csi : &default_wifi_csi_;
  
  // Initialize gain controller for AGC/FFT locking (uses median for robustness)
  gain_controller_.init(gain_lock_mode);
  
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
    // Cold reset: clear turbulence history and state.
    // Required after channel switch and post-calibration to avoid stale samples.
    detector_->clear_buffer();
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
  
  // STBC workaround (GitHub issue #76, #93, espressif/esp-csi#238):
  // some Multi-antenna routers can expose doubled HT CSI blocks (256->128 for HT20, or 228->114 for short 57-SC).
  if (csi_len == HT20_CSI_LEN_DOUBLE || csi_len == HT20_CSI_LEN_SHORT_DOUBLE) {
    // The two LTFs share the same HT20 subcarrier layout, so we keep the first block as a valid channel estimate.
    csi_len = (csi_len == HT20_CSI_LEN_DOUBLE) ? HT20_CSI_LEN : HT20_CSI_LEN_SHORT;

    static bool double_len_collapse_logged = false;
    if (!double_len_collapse_logged) {
      ESP_LOGI(TAG, "CSI double-length collapse active: 256->128 and/or 228->114");
      double_len_collapse_logged = true;
    }
  }

  // Fallback for short HT20 seen on C5 and potentially on other targets/AP combinations: 114 bytes maps to 57 complex samples with DC already present
  // We pad guards to fit our internal HT20 layout (64 SC, 128 bytes).
  int8_t csi_remapped[HT20_CSI_LEN];
  if (csi_len == HT20_CSI_LEN_SHORT) {
    std::memset(csi_remapped, 0, sizeof(csi_remapped));
    std::memcpy(&csi_remapped[HT20_CSI_LEN_SHORT_LEFT_PAD], csi_data, HT20_CSI_LEN_SHORT);
    csi_data = csi_remapped;
    csi_len = HT20_CSI_LEN;

    static bool remap_logged = false;
    if (!remap_logged) {
      ESP_LOGI(TAG, "CSI remap active: 57->64 SC (left_pad=4, right_pad=3)");
      remap_logged = true;
    }
  }
  
  // At this point we expect 128 bytes (64 SC) for HT20. Filter packets with unexpected SC count.
  if (csi_len != HT20_CSI_LEN) {
    if (++packets_filtered_ % 100 == 1) {
      log_wrong_sc_packet_(data, csi_len, packets_filtered_);
    }
    return;
  }
  
  // If calibration is in progress, delegate to calibrator
  if (calibrator_ != nullptr && calibrator_->is_calibrating()) {
    calibrator_->add_packet(csi_data, csi_len);
    return;
  }
  
  // Process CSI packet through detector
  const bool should_measure = (packets_total_++ % 1000 == 0);
  int64_t start_us = should_measure ? esp_timer_get_time() : 0;
  
  detector_->process_packet(csi_data, csi_len, selected_subcarriers_, NUM_SUBCARRIERS);
  
  // Handle periodic callback (or game mode which needs every packet)
  packets_processed_++;
  const bool should_publish = packets_processed_ >= publish_rate_;
  
  if (game_mode_callback_ || should_publish) {
    // Update detector state (lazy evaluation)
    detector_->update_state();
    
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
      }
      current_channel_ = packet_channel;
      
      if (packet_callback_) {
        packet_callback_(detector_->get_state(), packets_processed_);
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
#if CONFIG_IDF_TARGET_ESP32C5
  // ESP32-C5: MAC_VERSION_NUM = 3, WiFi 6 capable
  wifi_csi_config_t csi_config = {
    .enable = 1,
    .acquire_csi_legacy = 0,
    .acquire_csi_force_lltf = 0,
    .acquire_csi_ht20 = 1,
    .acquire_csi_ht40 = 0,
    .acquire_csi_vht = 0,
    .acquire_csi_su = 0,
    .acquire_csi_mu = 0,
    .acquire_csi_dcm = 0,
    .acquire_csi_beamformed = 0,
    .acquire_csi_he_stbc_mode = 0,
    .val_scale_cfg = 0,
    .dump_ack_en = 0,
  };
#elif CONFIG_IDF_TARGET_ESP32C6
  // ESP32-C6: MAC_VERSION_NUM = 2, WiFi 6 capable
  wifi_csi_config_t csi_config = {
    .enable = 1,
    .acquire_csi_legacy = 0,
    .acquire_csi_ht20 = 1,
    .acquire_csi_ht40 = 0,
    .acquire_csi_su = 0,
    .acquire_csi_mu = 0,
    .acquire_csi_dcm = 0,
    .acquire_csi_beamformed = 0,
    .acquire_csi_he_stbc = 0,
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
