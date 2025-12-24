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

void GainController::init(uint16_t calibration_packets) {
  calibration_packets_ = calibration_packets;
  packet_count_ = 0;
  agc_gain_sum_ = 0;
  fft_gain_sum_ = 0;
  agc_gain_locked_ = 0;
  fft_gain_locked_ = 0;
  
#if ESPECTRE_GAIN_LOCK_SUPPORTED
  locked_ = false;
  ESP_LOGD(TAG, "Gain controller initialized (calibration packets: %d)", calibration_packets);
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
    // Calculate averages and lock
    agc_gain_locked_ = static_cast<uint8_t>(agc_gain_sum_ / calibration_packets_);
    fft_gain_locked_ = static_cast<uint8_t>(fft_gain_sum_ / calibration_packets_);
    
    // Force the gain values
    phy_fft_scale_force(true, fft_gain_locked_);
    phy_force_rx_gain(1, agc_gain_locked_);
    
    locked_ = true;
    ESP_LOGI(TAG, "Gain locked: AGC=%d, FFT=%d (after %d packets)", 
             agc_gain_locked_, fft_gain_locked_, calibration_packets_);
    
    // Notify callback that gain is now locked (triggers NBVI calibration)
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

