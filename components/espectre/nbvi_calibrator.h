/*
 * ESPectre - NBVI Calibrator
 * 
 * NBVI (Normalized Baseline Variability Index) automatic subcarrier selection.
 * Selects optimal 12 non-consecutive subcarriers based on baseline stability.
 * 
 * Algorithm:
 * 1. Collect baseline CSI packets (quiet room)
 * 2. Find candidate baseline windows using percentile-based detection
 * 3. For each candidate, calculate NBVI for all subcarriers
 * 4. Select 12 subcarriers with lowest NBVI and spectral spacing
 * 5. Validate using MVS false positive rate
 * 
 * Returns (band, mv_values). Adaptive threshold is calculated externally.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "calibrator_interface.h"
#include "utils.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstddef>
#include <cstdint>
#include <functional>
#include <vector>
#include <cstdio>

namespace esphome {
namespace espectre {

// Forward declarations
class CSIManager;

/**
 * NBVI Calibrator
 * 
 * Automatic subcarrier selection using NBVI algorithm.
 * Selects 12 non-consecutive subcarriers with lowest baseline variability.
 */
class NBVICalibrator : public ICalibrator {
 public:
  // Use callback types from interface
  using result_callback_t = ICalibrator::result_callback_t;
  using collection_complete_callback_t = ICalibrator::collection_complete_callback_t;
  
  /**
   * Initialize NBVI calibrator (ICalibrator interface)
   * 
   * @param csi_manager CSI manager instance
   */
  void init(CSIManager* csi_manager) override {
    init(csi_manager, "/spiffs/nbvi_buffer.bin");
  }
  
  /**
   * Initialize NBVI calibrator with custom buffer path
   * 
   * @param csi_manager CSI manager instance
   * @param buffer_path Path for temporary calibration file
   */
  void init(CSIManager* csi_manager, const char* buffer_path);
  
  /**
   * Start automatic calibration
   * 
   * @param current_band Current subcarrier selection (for fallback)
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
  void set_window_size(uint16_t size) { window_size_ = size; }
  void set_window_step(uint16_t step) { window_step_ = step; }
  void set_percentile(uint8_t percentile) { percentile_ = percentile; }
  void set_alpha(float alpha) { alpha_ = alpha; }
  void set_min_spacing(uint8_t spacing) { min_spacing_ = spacing; }
  void set_noise_gate_percentile(uint8_t percentile) { noise_gate_percentile_ = percentile; }
  
  uint16_t get_buffer_size() const { return buffer_size_; }
  
 private:
  // Internal structures
  struct NBVIMetrics {
    uint8_t subcarrier;
    float nbvi;
    float mean;
    float std;
  };
  
  struct WindowVariance {
    uint16_t start_idx;
    float variance;
  };
  
  // Internal methods
  void on_collection_complete_();
  static void calibration_task_(void* arg);
  void finish_calibration_(bool success);
  esp_err_t run_calibration_();
  esp_err_t find_candidate_windows_(std::vector<WindowVariance>& candidates);
  void calculate_nbvi_metrics_(uint16_t baseline_start, std::vector<NBVIMetrics>& metrics);
  uint8_t apply_noise_gate_(std::vector<NBVIMetrics>& metrics);
  void select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                           uint8_t* output_band, uint8_t* output_size);
  bool validate_subcarriers_(const uint8_t* band, uint8_t band_size, 
                            float* out_fp_rate, std::vector<float>& out_mv_values);
  
  // Utility methods
  float calculate_percentile_(const std::vector<float>& sorted_values, uint8_t percentile) const;
  void calculate_nbvi_weighted_(const std::vector<float>& magnitudes, NBVIMetrics& out_metrics) const;
  
  // File I/O helpers
  bool ensure_spiffs_mounted_();
  bool open_buffer_file_for_writing_();
  bool open_buffer_file_for_reading_();
  void close_buffer_file_();
  void remove_buffer_file_();
  std::vector<uint8_t> read_window_(uint16_t start_idx, uint16_t window_size);
  
  // Members
  CSIManager* csi_manager_{nullptr};
  bool calibrating_{false};
  result_callback_t result_callback_;
  collection_complete_callback_t collection_complete_callback_;
  TaskHandle_t calibration_task_handle_{nullptr};
  
  // File-based storage
  FILE* buffer_file_{nullptr};
  uint16_t buffer_count_{0};
  const char* buffer_path_{"/spiffs/nbvi_buffer.bin"};
  
  // Configuration parameters
  uint16_t buffer_size_{700};
  uint16_t window_size_{200};
  uint16_t window_step_{50};
  uint8_t percentile_{10};
  float alpha_{0.5f};
  uint8_t min_spacing_{1};
  uint8_t noise_gate_percentile_{25};
  
  // Current calibration context
  std::vector<uint8_t> current_band_;
  uint8_t last_progress_{0};
  
  // Results
  uint8_t selected_band_[12];
  uint8_t selected_band_size_{0};
  std::vector<float> mv_values_;
  
  // Constants
  static constexpr uint8_t SELECTED_SUBCARRIERS_COUNT = 12;
  static constexpr uint16_t MVS_WINDOW_SIZE = 50;
  static constexpr float MVS_THRESHOLD = 1.0f;
  static constexpr float NULL_SUBCARRIER_THRESHOLD = 1.0f;
};

}  // namespace espectre
}  // namespace esphome
