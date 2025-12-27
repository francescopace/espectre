/*
 * ESPectre - CSI Manager
 * 
 * Manages ESP32 CSI (Channel State Information) hardware configuration.
 * Handles platform-specific differences (ESP32-C6 vs ESP32-S3).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_wifi.h"
#include "esp_err.h"
#include "esp_attr.h"  // For IRAM_ATTR
#include "csi_processor.h"
#include "wifi_csi_interface.h"
#include "gain_controller.h"
#include <functional>

namespace esphome {
namespace espectre {

// Forward declaration
class CalibrationManager;

// Callback type for processed CSI data
using csi_processed_callback_t = std::function<void(csi_motion_state_t)>;

// Callback type for game mode (called every packet with movement and threshold)
using game_mode_callback_t = std::function<void(float movement, float threshold)>;

/**
 * CSI Manager
 * 
 * Manages complete CSI pipeline: hardware configuration, data processing, and motion detection.
 * Handles platform-specific differences between ESP32-C6 and ESP32-S3.
 * Orchestrates CSI packet processing and NBVI calibration.
 */
class CSIManager {
 public:
  /**
   * Initialize CSI Manager
   * 
   * @param processor CSI processor context
   * @param selected_subcarriers Initial subcarrier selection (array of 12 subcarriers)
   * @param segmentation_threshold Motion detection threshold
   * @param segmentation_window_size Moving variance window size
   * @param publish_rate Number of packets before triggering callback
   * @param lowpass_enabled Whether low-pass filter is enabled
   * @param lowpass_cutoff Low-pass filter cutoff frequency in Hz
   * @param hampel_enabled Whether Hampel filter is enabled
   * @param hampel_window Hampel window size (3-11)
   * @param hampel_threshold Hampel threshold (MAD multiplier)
   * @param wifi_csi WiFi CSI interface (nullptr for real implementation)
   */
  void init(csi_processor_context_t* processor,
            const uint8_t selected_subcarriers[12],
            float segmentation_threshold,
            uint16_t segmentation_window_size,
            uint32_t publish_rate,
            bool lowpass_enabled,
            float lowpass_cutoff,
            bool hampel_enabled,
            uint8_t hampel_window,
            float hampel_threshold,
            IWiFiCSI* wifi_csi = nullptr);
  
  /**
   * Update subcarrier selection
   * 
   * @param subcarriers New subcarrier selection (array of 12 subcarriers)
   */
  void update_subcarrier_selection(const uint8_t subcarriers[12]);
  
  /**
   * Update segmentation threshold
   * 
   * @param threshold New threshold value
   */
  void set_threshold(float threshold);
  
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
  
  /**
   * Process incoming CSI packet
   * 
   * Orchestrates: calibration check → processing → callbacks
   * 
   * @param data CSI packet data
   */
  void process_packet(wifi_csi_info_t* data);
  
  /**
   * Set calibration mode
   * 
   * @param calibrator Calibration manager instance (nullptr to disable calibration mode)
   */
  void set_calibration_mode(CalibrationManager* calibrator) { calibrator_ = calibrator; }
  
  /**
   * Check if CSI is currently enabled
   * 
   * @return true if enabled, false otherwise
   */
  bool is_enabled() const { return enabled_; }
  
  /**
   * Check if gain is locked
   * 
   * @return true if gain calibration is complete
   */
  bool is_gain_locked() const { return gain_controller_.is_locked(); }
  
  /**
   * Get the number of packets used for gain lock calibration
   * 
   * @return Calibration packet count (default: 300)
   */
  uint16_t get_gain_lock_packets() const { return gain_controller_.get_calibration_packets(); }
  
  /**
   * Get the gain controller (for status reporting)
   * 
   * @return Reference to gain controller
   */
  const GainController& get_gain_controller() const { return gain_controller_; }
  
  /**
   * Set callback for when gain lock completes
   * 
   * Use this to trigger NBVI calibration after gain is locked.
   * 
   * @param callback Function to call when gain is locked
   */
  void set_gain_lock_callback(GainController::lock_complete_callback_t callback) {
    gain_controller_.set_lock_complete_callback(callback);
  }
  
  /**
   * Set game mode callback
   * 
   * When set, this callback is called every CSI packet with movement and threshold.
   * Used for low-latency game mode communication.
   * 
   * @param callback Function to call every packet (nullptr to disable)
   */
  void set_game_mode_callback(game_mode_callback_t callback) {
    game_mode_callback_ = callback;
  }
  
 private:
  // Static wrapper for ESP-IDF C callback
  // IRAM_ATTR: Keep in IRAM for consistent low-latency execution from ISR context
  static void IRAM_ATTR csi_rx_callback_wrapper_(void* ctx, wifi_csi_info_t* data);
  
  bool enabled_{false};
  csi_processor_context_t* processor_{nullptr};
  const uint8_t* selected_subcarriers_{nullptr};
  CalibrationManager* calibrator_{nullptr};
  csi_processed_callback_t packet_callback_;
  game_mode_callback_t game_mode_callback_;
  uint32_t publish_rate_{100};
  volatile uint32_t packets_processed_{0};  // volatile: modified from ISR callback
  uint8_t current_channel_{0};  // Track WiFi channel for change detection
  
  // WiFi CSI interface (injected or default real implementation)
  IWiFiCSI* wifi_csi_{nullptr};
  WiFiCSIReal default_wifi_csi_;
  
  // Gain controller for AGC/FFT locking
  GainController gain_controller_;
  
  static constexpr uint8_t NUM_SUBCARRIERS = 12;
  
  /**
   * Configure CSI based on platform
   * 
   * @return ESP_OK on success
   */
  esp_err_t configure_platform_specific_();
};

}  // namespace espectre
}  // namespace esphome
