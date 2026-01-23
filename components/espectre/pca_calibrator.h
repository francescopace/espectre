/*
 * ESPectre - PCA Calibrator
 * 
 * Calibrator for PCA-based motion detection.
 * Collects baseline correlation values during quiet period.
 * 
 * Algorithm:
 * 1. Collect baseline CSI packets (quiet room)
 * 2. For each packet: extract amplitudes, compute PCA, calculate correlation
 * 3. Store correlation values (cal_values)
 * 4. Return band (PCA subcarriers) and cal_values for threshold calculation
 * 
 * Threshold formula (calculated externally): 1 - min(correlation)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "calibrator_interface.h"
#include "pca_detector.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>

namespace esphome {
namespace espectre {

// Forward declarations
class CSIManager;

/**
 * PCA Calibrator
 * 
 * Calibrator for PCA-based motion detection.
 * Collects correlation values during baseline for threshold calculation.
 */
class PCACalibrator : public ICalibrator {
 public:
  // Use callback types from interface
  using result_callback_t = ICalibrator::result_callback_t;
  using collection_complete_callback_t = ICalibrator::collection_complete_callback_t;
  
  PCACalibrator();
  ~PCACalibrator();
  
  /**
   * Initialize PCA calibrator (ICalibrator interface)
   * 
   * @param csi_manager CSI manager instance
   */
  void init(CSIManager* csi_manager) override;
  
  /**
   * Start calibration
   * 
   * @param current_band Current subcarrier selection (ignored for PCA)
   * @param current_band_size Size of current band
   * @param callback Callback to invoke with results
   * @return ESP_OK on success
   */
  esp_err_t start_calibration(const uint8_t* current_band,
                              uint8_t current_band_size,
                              result_callback_t callback) override;
  
  /**
   * Add CSI packet to calibration buffer
   * 
   * @param csi_data Raw CSI data (I/Q pairs)
   * @param csi_len Length of CSI data
   * @return true if buffer is full and calibration should proceed
   */
  bool add_packet(const int8_t* csi_data, size_t csi_len) override;
  
  /**
   * Check if calibration is in progress
   */
  bool is_calibrating() const override { return calibrating_; }
  
  /**
   * Set callback for collection complete notification
   */
  void set_collection_complete_callback(collection_complete_callback_t callback) override {
    collection_complete_callback_ = callback;
  }
  
  /**
   * Configuration setters
   */
  void set_buffer_size(uint16_t size) { buffer_size_ = size; }
  uint16_t get_buffer_size() const { return buffer_size_; }
  
 private:
  // Extract amplitudes from CSI data (every PCA_SUBCARRIER_STEP-th)
  void extract_amplitudes(const int8_t* csi_data, size_t len, float* amplitudes);
  
  // Compute PCA on current window
  bool compute_pca(float* pca_output);
  
  // Calculate correlation with past PCA vectors
  float compute_correlation(const float* pca_current);
  
  // Internal methods
  void on_collection_complete_();
  void finish_calibration_(bool success);
  
  // Members
  CSIManager* csi_manager_{nullptr};
  bool calibrating_{false};
  result_callback_t result_callback_;
  collection_complete_callback_t collection_complete_callback_;
  
  // CSI amplitude buffer for PCA [PCA_WINDOW_SIZE][PCA_NUM_SUBCARRIERS]
  float* csi_buffer_[PCA_WINDOW_SIZE];
  size_t csi_buffer_count_{0};
  
  // PCA output buffer [PCA_BUFFER_SIZE][PCA_NUM_SUBCARRIERS]
  float* pca_buffer_[PCA_BUFFER_SIZE];
  size_t pca_buffer_count_{0};
  size_t pca_buffer_idx_{0};
  
  // Collected correlation values
  std::vector<float> correlation_values_;
  
  // Configuration parameters
  uint16_t buffer_size_{700};  // Same as NBVI/P95 (~7 seconds at 100 Hz)
  uint16_t packet_count_{0};
  
  // Results - PCA uses fixed subcarriers (every PCA_SUBCARRIER_STEP-th)
  uint8_t selected_band_[16];  // Up to 16 subcarriers for PCA
  uint8_t selected_band_size_{0};
};

}  // namespace espectre
}  // namespace esphome
