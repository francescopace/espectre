/*
 * ESPectre - Gain Controller
 * 
 * Manages AGC (Automatic Gain Control) and FFT gain locking for stable CSI measurements.
 * Based on Espressif esp-csi recommendations for improved CSI quality.
 * 
 * The ESP32 WiFi hardware has automatic gain control that can cause CSI amplitude
 * variations even in static environments. This controller:
 * 1. Collects gain statistics from the first N packets after boot
 * 2. Calculates average AGC and FFT gain values
 * 3. Locks (forces) these values to eliminate gain-induced variations
 * 
 * Supported platforms: ESP32-S3, ESP32-C3, ESP32-C5, ESP32-C6
 * (ESP32 and ESP32-S2 do not expose these PHY functions)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "sdkconfig.h"
#include "esp_wifi.h"
#include <cstdint>
#include <functional>

namespace esphome {
namespace espectre {

// Gain lock is only available on newer ESP32 variants
#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6
#define ESPECTRE_GAIN_LOCK_SUPPORTED 1
#else
#define ESPECTRE_GAIN_LOCK_SUPPORTED 0
#endif

/**
 * PHY RX Control structure with gain fields
 * 
 * This structure overlays wifi_csi_info_t to access undocumented
 * PHY fields (agc_gain and fft_gain) that are present on newer ESP32 variants.
 * 
 * Based on Espressif esp-csi example:
 * https://github.com/espressif/esp-csi/blob/master/examples/get-started/csi_recv_router/main/app_main.c
 */
typedef struct {
    unsigned : 32;  // reserved
    unsigned : 32;  // reserved
    unsigned : 32;  // reserved
    unsigned : 32;  // reserved
    unsigned : 32;  // reserved
#if CONFIG_IDF_TARGET_ESP32S2
    unsigned : 32;  // reserved
#elif ESPECTRE_GAIN_LOCK_SUPPORTED
    unsigned : 16;  // reserved
    unsigned fft_gain : 8;   // FFT scaling gain
    unsigned agc_gain : 8;   // Automatic Gain Control value
    unsigned : 32;  // reserved
#endif
    unsigned : 32;  // reserved
#if CONFIG_IDF_TARGET_ESP32S2
    signed : 8;     // reserved
    unsigned : 24;  // reserved
#elif CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5
    unsigned : 32;  // reserved
    unsigned : 32;  // reserved
    unsigned : 32;  // reserved
#endif
    unsigned : 32;  // reserved
} wifi_pkt_rx_ctrl_phy_t;

#if ESPECTRE_GAIN_LOCK_SUPPORTED
// External PHY functions (from ESP-IDF PHY blob, not in public headers)
extern "C" {
    /**
     * Enable/disable automatic FFT gain control and set its value
     * @param force_en true to disable automatic FFT gain control
     * @param force_value forced FFT gain value
     */
    void phy_fft_scale_force(bool force_en, uint8_t force_value);

    /**
     * Enable/disable automatic gain control and set its value
     * @param force_en true to disable automatic gain control
     * @param force_value forced gain value
     */
    void phy_force_rx_gain(int force_en, int force_value);
}
#endif

/**
 * Gain Controller
 * 
 * Collects AGC/FFT gain statistics and locks them for stable CSI measurements.
 * This eliminates amplitude variations caused by the WiFi hardware's automatic
 * gain control, which can otherwise cause false motion detections.
 * 
 * The gain lock phase happens BEFORE NBVI calibration to ensure clean data:
 * - Phase 1: Gain Lock (~3 seconds, 300 packets) - locks AGC/FFT
 * - Phase 2: NBVI Calibration (~7 seconds, 700 packets) - with stable gain
 */
class GainController {
 public:
  // Callback type for when gain lock completes
  using lock_complete_callback_t = std::function<void()>;
  
  /**
   * Initialize the gain controller
   * 
   * @param calibration_packets Number of packets to collect before locking (default: 300, ~3 seconds)
   */
  void init(uint16_t calibration_packets = 300);
  
  /**
   * Set callback for when gain lock completes
   * 
   * If gain lock is not supported on this platform, the callback is
   * invoked immediately since gain is already considered "locked".
   * 
   * @param callback Function to call when gain is locked
   */
  void set_lock_complete_callback(lock_complete_callback_t callback) {
    lock_complete_callback_ = callback;
    // If gain lock was skipped (unsupported platform), call callback immediately
    if (skip_gain_lock_ && callback) {
      callback();
    }
  }
  
  /**
   * Process a CSI packet for gain calibration
   * 
   * Should be called for every CSI packet until is_locked() returns true.
   * After calibration_packets, automatically locks the gain values.
   * Extracts AGC/FFT internally from the packet structure.
   * 
   * @param info CSI packet info
   */
  void process_packet(const wifi_csi_info_t* info);
  
  /**
   * Check if gain values have been locked
   * 
   * @return true if gain is locked, false if still calibrating
   */
  bool is_locked() const { return locked_; }
  
  /**
   * Check if gain lock is supported on this platform
   * 
   * @return true if supported (S3/C3/C5/C6), false otherwise
   */
  static constexpr bool is_supported() {
#if ESPECTRE_GAIN_LOCK_SUPPORTED
    return true;
#else
    return false;
#endif
  }
  
  /**
   * Get the locked AGC gain value
   * 
   * @return AGC gain value (only valid after is_locked() == true)
   */
  uint8_t get_agc_gain() const { return agc_gain_locked_; }
  
  /**
   * Get the locked FFT gain value
   * 
   * @return FFT gain value (only valid after is_locked() == true)
   */
  uint8_t get_fft_gain() const { return fft_gain_locked_; }
  
  /**
   * Get the number of packets processed so far
   * 
   * @return Packet count
   */
  uint16_t get_packet_count() const { return packet_count_; }
  
  /**
   * Get the number of packets used for gain lock calibration
   * 
   * @return Calibration packet count (default: 300)
   */
  uint16_t get_calibration_packets() const { return calibration_packets_; }
  
 private:
  uint16_t calibration_packets_{300};
  uint16_t packet_count_{0};
  uint32_t agc_gain_sum_{0};
  uint32_t fft_gain_sum_{0};
  uint8_t agc_gain_locked_{0};
  uint8_t fft_gain_locked_{0};
  bool locked_{false};
  bool skip_gain_lock_{false};  // Set true on platforms without gain lock support
  lock_complete_callback_t lock_complete_callback_{nullptr};
};

}  // namespace espectre
}  // namespace esphome

