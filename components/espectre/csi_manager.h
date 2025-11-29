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
#include "csi_processor.h"
#include <functional>

namespace esphome {
namespace espectre {

// Forward declaration
class CalibrationManager;

// Callback type for processed CSI data
using csi_processed_callback_t = std::function<void(csi_motion_state_t)>;

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
   * @param hampel_enabled Whether Hampel filter is enabled
   * @param hampel_window Hampel window size (3-11)
   * @param hampel_threshold Hampel threshold (MAD multiplier)
   */
  void init(csi_processor_context_t* processor,
            const uint8_t selected_subcarriers[12],
            float segmentation_threshold,
            uint16_t segmentation_window_size,
            uint32_t publish_rate,
            bool hampel_enabled,
            uint8_t hampel_window,
            float hampel_threshold);
  
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
   * Orchestrates: calibration check → processing
   * 
   * @param data CSI packet data
   * @param motion_state Output for motion state
   */
  void process_packet(wifi_csi_info_t* data,
                     csi_motion_state_t& motion_state);
  
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
  
 private:
  // Static wrapper for ESP-IDF C callback
  static void csi_rx_callback_wrapper_(void* ctx, wifi_csi_info_t* data);
  
  bool enabled_{false};
  csi_processor_context_t* processor_{nullptr};
  const uint8_t* selected_subcarriers_{nullptr};
  CalibrationManager* calibrator_{nullptr};
  csi_processed_callback_t packet_callback_;
  uint32_t publish_rate_{100};
  uint32_t packets_processed_{0};
  
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
