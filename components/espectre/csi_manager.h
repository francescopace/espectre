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
#include "esp_attr.h"
#include "utils.h"
#include "detector_interface.h"
#include "wifi_csi_interface.h"
#include "gain_controller.h"
#include <functional>

namespace esphome {
namespace espectre {

// Forward declaration
class ICalibrator;

// Callback type for processed CSI data
using csi_processed_callback_t = std::function<void(csi_motion_state_t, uint32_t)>;

// Callback type for game mode (called every packet with movement and threshold)
using game_mode_callback_t = std::function<void(float movement, float threshold)>;

/**
 * CSI Manager
 * 
 * Manages complete CSI pipeline: hardware configuration, data processing, and motion detection.
 * Handles platform-specific differences between ESP32-C6 and ESP32-S3.
 * Orchestrates CSI packet processing and band calibration.
 */
class CSIManager {
 public:
  /**
   * Initialize CSI Manager
   * 
   * @param detector Motion detector instance (IDetector*)
   * @param selected_subcarriers Initial subcarrier selection (array of 12 subcarriers)
   * @param publish_rate Number of packets before triggering callback
   * @param gain_lock_mode Gain lock mode (auto/enabled/disabled)
   * @param wifi_csi WiFi CSI interface (nullptr for real implementation)
   */
  void init(IDetector* detector,
            const uint8_t selected_subcarriers[12],
            uint32_t publish_rate,
            GainLockMode gain_lock_mode = GainLockMode::AUTO,
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
   * When a calibrator is set, CSI packets are routed to it during calibration.
   * 
   * @param calibrator Calibrator instance (nullptr to disable calibration mode)
   */
  void set_calibration_mode(ICalibrator* calibrator) { calibrator_ = calibrator; }
  
  /**
   * Check if CSI is currently enabled
   */
  bool is_enabled() const { return enabled_; }
  
  /**
   * Check if gain is locked
   */
  bool is_gain_locked() const { return gain_controller_.is_locked(); }
  
  /**
   * Get the number of packets used for gain lock calibration
   */
  uint16_t get_gain_lock_packets() const { return gain_controller_.get_calibration_packets(); }
  
  /**
   * Get the gain controller (for status reporting)
   */
  const GainController& get_gain_controller() const { return gain_controller_; }
  
  /**
   * Set callback for when gain lock completes
   */
  void set_gain_lock_callback(GainController::lock_complete_callback_t callback) {
    gain_controller_.set_lock_complete_callback(callback);
  }
  
  /**
   * Set game mode callback
   */
  void set_game_mode_callback(game_mode_callback_t callback) {
    game_mode_callback_ = callback;
  }
  
  /**
   * Get the detector instance
   */
  IDetector* get_detector() { return detector_; }
  
  /**
   * Clear detector buffer (for calibration reset)
   */
  void clear_detector_buffer();
  
 private:
  static void IRAM_ATTR csi_rx_callback_wrapper_(void* ctx, wifi_csi_info_t* data);
  
  bool enabled_{false};
  IDetector* detector_{nullptr};
  const uint8_t* selected_subcarriers_{nullptr};
  ICalibrator* calibrator_{nullptr};
  csi_processed_callback_t packet_callback_;
  game_mode_callback_t game_mode_callback_;
  uint32_t publish_rate_{100};
  volatile uint32_t packets_processed_{0};
  volatile uint32_t packets_filtered_{0};
  uint8_t current_channel_{0};
  
  IWiFiCSI* wifi_csi_{nullptr};
  WiFiCSIReal default_wifi_csi_;
  GainController gain_controller_;
  
  static constexpr uint8_t NUM_SUBCARRIERS = HT20_SELECTED_BAND_SIZE;
  
  esp_err_t configure_platform_specific_();
};

}  // namespace espectre
}  // namespace esphome
