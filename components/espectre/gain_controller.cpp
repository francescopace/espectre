/*
 * ESPectre - Gain Controller Implementation
 * 
 * Manages AGC/FFT gain locking for stable CSI measurements.
 * Uses median calculation for robust baseline (matches Espressif implementation).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "gain_controller.h"
#include "esphome/core/log.h"
#include <algorithm>
#include <cmath>

namespace esphome {
namespace espectre {

static const char *TAG = "GainController";

// Helper function to calculate median of uint8_t array
uint8_t GainController::calculate_median_u8(uint8_t* arr, uint16_t size) {
  if (size == 0) return 0;
  std::sort(arr, arr + size);
  if (size % 2 == 0) {
    // Even number: average of two middle values
    return (arr[size / 2 - 1] + arr[size / 2]) / 2;
  } else {
    // Odd number: middle value
    return arr[size / 2];
  }
}

// Helper function to calculate median of int8_t array
int8_t GainController::calculate_median_i8(int8_t* arr, uint16_t size) {
  if (size == 0) return 0;
  std::sort(arr, arr + size);
  if (size % 2 == 0) {
    // Even number: average of two middle values
    return (arr[size / 2 - 1] + arr[size / 2]) / 2;
  } else {
    // Odd number: middle value
    return arr[size / 2];
  }
}

void GainController::init(GainLockMode mode) {
  mode_ = mode;
  packet_count_ = 0;
  agc_gain_locked_ = 0;
  fft_gain_locked_ = 0;
  skipped_strong_signal_ = false;
  
#if ESPECTRE_GAIN_LOCK_SUPPORTED
  // All modes need calibration phase to establish baseline for compensation
  locked_ = false;
  skip_gain_lock_ = false;
  
  const char* mode_str;
  switch (mode) {
    case GainLockMode::AUTO: mode_str = "auto"; break;
    case GainLockMode::ENABLED: mode_str = "enabled"; break;
    case GainLockMode::DISABLED: mode_str = "disabled (with compensation)"; break;
  }
  ESP_LOGD(TAG, "Gain controller initialized (mode: %s, %d packets, using median)", 
           mode_str, CALIBRATION_PACKETS);
#else
  // On unsupported platforms, mark as locked immediately (no calibration phase)
  locked_ = true;
  skip_gain_lock_ = true;
  ESP_LOGD(TAG, "Gain lock not supported on this platform (skipping)");
#endif
}

void GainController::process_packet(const wifi_csi_info_t* info) {
#if ESPECTRE_GAIN_LOCK_SUPPORTED
  if (locked_ || info == nullptr) {
    return;
  }
  
  // Cast to PHY structure to access hidden gain fields
  const wifi_pkt_rx_ctrl_phy_t* phy_info = reinterpret_cast<const wifi_pkt_rx_ctrl_phy_t*>(info);
  
  if (packet_count_ < CALIBRATION_PACKETS) {
    // Store gain values for median calculation
    agc_samples_[packet_count_] = phy_info->agc_gain;
    fft_samples_[packet_count_] = phy_info->fft_gain;
    packet_count_++;
    
    // Log progress every 25% (useful for debugging)
    if (packet_count_ == CALIBRATION_PACKETS / 4 ||
        packet_count_ == CALIBRATION_PACKETS / 2 ||
        packet_count_ == (CALIBRATION_PACKETS * 3) / 4) {
      ESP_LOGD(TAG, "Gain calibration %d%% (%d/%d packets)", 
               (packet_count_ * 100) / CALIBRATION_PACKETS,
               packet_count_, CALIBRATION_PACKETS);
    }
  } else if (packet_count_ == CALIBRATION_PACKETS) {
    // Calculate medians (more robust than mean against outliers)
    agc_gain_locked_ = calculate_median_u8(agc_samples_, CALIBRATION_PACKETS);
    fft_gain_locked_ = calculate_median_i8(fft_samples_, CALIBRATION_PACKETS);
    
    locked_ = true;
    packet_count_++;  // Prevent re-entry
    
    // Handle different modes
    if (mode_ == GainLockMode::DISABLED) {
      // DISABLED mode: baseline collected for compensation, but no gain lock
      ESP_LOGI(TAG, "Gain baseline: AGC=%d, FFT=%d (compensation enabled, no lock)", 
               agc_gain_locked_, fft_gain_locked_);
    } else if (mode_ == GainLockMode::AUTO && agc_gain_locked_ < MIN_SAFE_AGC) {
      // AUTO mode with strong signal: skip gain lock to prevent CSI freeze
      skipped_strong_signal_ = true;
      ESP_LOGW(TAG, "Signal too strong (AGC=%d < %d) - skipping gain lock (compensation enabled)", 
               agc_gain_locked_, MIN_SAFE_AGC);
      ESP_LOGW(TAG, "Move sensor 2-3 meters from AP for optimal performance");
    } else {
      // AUTO/ENABLED mode: force gain lock
      phy_fft_scale_force(true, fft_gain_locked_);
      phy_force_rx_gain(1, agc_gain_locked_);
      
      ESP_LOGI(TAG, "Gain locked: AGC=%d, FFT=%d (median of %d packets)", 
               agc_gain_locked_, fft_gain_locked_, CALIBRATION_PACKETS);
    }
    
    ESP_LOGI(TAG, "HT20 mode: 64 subcarriers");
    
    // Notify callback that calibration is complete (triggers band calibration)
    if (lock_complete_callback_) {
      lock_complete_callback_();
    }
  }
#else
  // On unsupported platforms, gain lock is not available
  // The lock is already set to true in init() on unsupported platforms
  (void)info;
#endif
}

float GainController::calculate_compensation(const wifi_csi_info_t* info) const {
#if ESPECTRE_GAIN_LOCK_SUPPORTED
  // No compensation needed if gain is properly locked
  if (!needs_compensation() || info == nullptr) {
    return 1.0f;
  }
  
  // Cast to PHY structure to access current gain values
  const wifi_pkt_rx_ctrl_phy_t* phy_info = reinterpret_cast<const wifi_pkt_rx_ctrl_phy_t*>(info);
  
  uint8_t current_agc = phy_info->agc_gain;
  int8_t current_fft = phy_info->fft_gain;
  
  // Calculate compensation factor (Espressif formula)
  // Gain values are in dB, so we convert delta to linear scale
  float agc_delta = static_cast<float>(agc_gain_locked_) - static_cast<float>(current_agc);
  float fft_delta = static_cast<float>(fft_gain_locked_) - static_cast<float>(current_fft);
  
  float compensation = std::pow(10.0f, agc_delta / 20.0f) * 
                       std::pow(10.0f, fft_delta / 20.0f);
  
  // Clamp to reasonable range to avoid extreme values
  if (compensation < 0.1f) compensation = 0.1f;
  if (compensation > 10.0f) compensation = 10.0f;
  
  return compensation;
#else
  (void)info;
  return 1.0f;
#endif
}

}  // namespace espectre
}  // namespace esphome
