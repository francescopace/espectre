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
#include "utils.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "GainController";

void GainController::init(GainLockMode mode) {
  mode_ = mode;
  packet_count_ = 0;
  agc_gain_locked_ = 0;
  fft_gain_locked_ = 0;
  skipped_strong_signal_ = false;
  
#if ESPECTRE_GAIN_LOCK_SUPPORTED
  // All modes need calibration phase to establish baseline
  locked_ = false;
  skip_gain_lock_ = false;
  
  const char* mode_str;
  switch (mode) {
    case GainLockMode::AUTO: mode_str = "auto"; break;
    case GainLockMode::ENABLED: mode_str = "enabled"; break;
    case GainLockMode::DISABLED: mode_str = "disabled"; break;
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
    agc_gain_locked_ = esphome::espectre::calculate_median_u8(agc_samples_, CALIBRATION_PACKETS);
    fft_gain_locked_ = esphome::espectre::calculate_median_i8(fft_samples_, CALIBRATION_PACKETS);
    
    locked_ = true;
    packet_count_++;  // Prevent re-entry
    
    // Handle different modes
    if (mode_ == GainLockMode::DISABLED) {
      // DISABLED mode: no gain lock, will use CV normalization
      ESP_LOGI(TAG, "Gain baseline: AGC=%d, FFT=%d (no lock, CV normalization enabled)", 
               agc_gain_locked_, fft_gain_locked_);
    } else if (mode_ == GainLockMode::AUTO && agc_gain_locked_ < MIN_SAFE_AGC) {
      // AUTO mode with strong signal: skip gain lock to prevent CSI freeze
      skipped_strong_signal_ = true;
      ESP_LOGW(TAG, "Signal too strong (AGC=%d < %d) - skipping gain lock, using CV normalization", 
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

}  // namespace espectre
}  // namespace esphome
