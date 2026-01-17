/*
 * ESPectre - Gain Controller Implementation
 * 
 * Manages AGC/FFT gain locking for stable CSI measurements.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "gain_controller.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "GainController";

void GainController::init(uint16_t calibration_packets, GainLockMode mode) {
  calibration_packets_ = calibration_packets;
  mode_ = mode;
  packet_count_ = 0;
  agc_gain_sum_ = 0;
  fft_gain_sum_ = 0;
  agc_gain_locked_ = 0;
  fft_gain_locked_ = 0;
  skipped_strong_signal_ = false;
  
#if ESPECTRE_GAIN_LOCK_SUPPORTED
  if (mode == GainLockMode::DISABLED) {
    // User explicitly disabled gain lock
    locked_ = true;
    skip_gain_lock_ = true;
    ESP_LOGD(TAG, "Gain lock disabled by configuration");
  } else {
    locked_ = false;
    const char* mode_str = (mode == GainLockMode::AUTO) ? "auto" : "enabled";
    ESP_LOGD(TAG, "Gain controller initialized (mode: %s, calibration packets: %d)", 
             mode_str, calibration_packets);
  }
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
  
  if (packet_count_ < calibration_packets_) {
    // Accumulate gain values
    agc_gain_sum_ += phy_info->agc_gain;
    fft_gain_sum_ += phy_info->fft_gain;
    packet_count_++;
    
    // Log current average every 25% (useful for debugging)
    if (packet_count_ == calibration_packets_ / 4 ||
        packet_count_ == calibration_packets_ / 2 ||
        packet_count_ == (calibration_packets_ * 3) / 4) {
      uint8_t avg_agc = static_cast<uint8_t>(agc_gain_sum_ / packet_count_);
      uint8_t avg_fft = static_cast<uint8_t>(fft_gain_sum_ / packet_count_);
      ESP_LOGD(TAG, "Gain calibration %d%%: AGC~%d, FFT~%d (%d/%d packets)", 
               (packet_count_ * 100) / calibration_packets_, avg_agc, avg_fft,
               packet_count_, calibration_packets_);
    }
  } else if (packet_count_ == calibration_packets_) {
    // Calculate averages
    agc_gain_locked_ = static_cast<uint8_t>(agc_gain_sum_ / calibration_packets_);
    fft_gain_locked_ = static_cast<uint8_t>(fft_gain_sum_ / calibration_packets_);
    
    // Check if we should skip gain lock due to strong signal (AUTO mode only)
    if (mode_ == GainLockMode::AUTO && agc_gain_locked_ < MIN_SAFE_AGC) {
      // Signal too strong - skip gain lock to prevent CSI freeze
      locked_ = true;
      skipped_strong_signal_ = true;
      ESP_LOGW(TAG, "Signal too strong (AGC=%d < %d) - skipping gain lock", 
               agc_gain_locked_, MIN_SAFE_AGC);
      ESP_LOGW(TAG, "Move sensor 2-3 meters from AP for optimal performance");
      
      // Notify callback (calibration will proceed without gain lock)
      if (lock_complete_callback_) {
        lock_complete_callback_();
      }
      return;
    }
    
    // Force the gain values
    phy_fft_scale_force(true, fft_gain_locked_);
    phy_force_rx_gain(1, agc_gain_locked_);
    
    locked_ = true;
    ESP_LOGI(TAG, "Gain locked: AGC=%d, FFT=%d (after %d packets)", 
             agc_gain_locked_, fft_gain_locked_, calibration_packets_);
    ESP_LOGI(TAG, "HT20 mode: 64 subcarriers");
    
    // Notify callback that gain is now locked (triggers band calibration)
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

