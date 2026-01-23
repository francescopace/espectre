/*
 * ESPectre - P95 Calibrator
 * 
 * Automatic subcarrier band selection using P95 moving variance optimization.
 * Selects optimal 12-subcarrier band for motion detection by minimizing
 * false positive rate.
 * 
 * Returns (band, mv_values). Adaptive threshold is calculated externally
 * using threshold.h after band selection.
 * 
 * Uses file-based storage to avoid RAM limitations. Magnitudes stored as
 * uint8 (max CSI magnitude ~181 fits in 1 byte).
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
 * P95 Calibrator
 * 
 * Orchestrates P95-based band calibration process:
 * 1. Collects CSI packets during baseline period
 * 2. Evaluates candidate bands using P95 of moving variance
 * 3. Selects optimal 12-subcarrier band for motion detection
 * 
 * Uses P95 optimization to minimize false positive rate.
 */

class P95Calibrator : public ICalibrator {
 public:
  // Use callback types from interface
  using result_callback_t = ICalibrator::result_callback_t;
  using collection_complete_callback_t = ICalibrator::collection_complete_callback_t;
  
  /**
   * Initialize calibrator (ICalibrator interface)
   * 
   * @param csi_manager CSI manager instance
   */
  void init(CSIManager* csi_manager) override { 
    init(csi_manager, "/spiffs/band_buffer.bin"); 
  }
  
  /**
   * Initialize calibrator with custom buffer path
   * 
   * @param csi_manager CSI manager instance
   * @param buffer_path Path for temporary calibration file
   */
  void init(CSIManager* csi_manager, const char* buffer_path);
  
  /**
   * Start automatic calibration
   * 
   * Begins collecting CSI packets for calibration analysis.
   * CSI Manager will be set to calibration mode.
   * 
   * @param current_band Current subcarrier selection (for baseline detection)
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
   * Called by CSI Manager during calibration mode.
   * 
   * @param csi_data Raw CSI data (I/Q pairs)
   * @param csi_len Length of CSI data
   * @return true if buffer is full and calibration should proceed
   */
  bool add_packet(const int8_t* csi_data, size_t csi_len) override;
  
  /**
   * Check if calibration is in progress
   * 
   * @return true if calibrating, false otherwise
   */
  bool is_calibrating() const override { return calibrating_; }
  
  /**
   * Configuration setters (optional, use before start_calibration)
   */
  void set_buffer_size(uint16_t size) { buffer_size_ = size; }
  
  /**
   * Initialize subcarrier configuration (HT20 only)
   * 
   * Sets up guard bands for 64 subcarriers. Called automatically
   * before calibration starts.
   */
  void init_subcarrier_config();
  
  uint16_t get_buffer_size() const { return buffer_size_; }
  void set_skip_subcarrier_selection(bool skip) { skip_subcarrier_selection_ = skip; }
  void set_collection_complete_callback(collection_complete_callback_t callback) override { 
    collection_complete_callback_ = callback; 
  }
  
  /**
   * Get the moving variance values from calibration
   * 
   * @return MV values (only valid after calibration completes)
   */
  const std::vector<float>& get_mv_values() const { return mv_values_; }
  
 private:
  // Internal structures
  struct BandResult {
    uint8_t start_sc;                // First subcarrier in band
    float p95;                       // P95 of moving variance (for band selection)
    float fp_estimate;               // Estimated FP rate
    std::vector<float> mv_values;    // Moving variance values for threshold calc
  };
  
  // Internal methods
  void on_collection_complete_();
  esp_err_t run_calibration_();
  static void calibration_task_(void* arg);
  void finish_calibration_(bool success);
  void get_candidate_bands_(std::vector<std::vector<uint8_t>>& candidates);
  BandResult evaluate_band_(const std::vector<uint8_t>& window_data, const uint8_t* band, uint8_t band_size);
  float calculate_percentile_(const std::vector<float>& values, uint8_t percentile) const;
  
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
  
  // File-based storage (saves RAM - magnitudes stored as uint8)
  FILE* buffer_file_{nullptr};
  uint16_t buffer_count_{0};
  const char* buffer_path_{"/spiffs/band_buffer.bin"};
  
  // Configuration parameters
  uint16_t buffer_size_{700};         // Number of packets to collect (~7 seconds at 100 Hz)
  bool skip_subcarrier_selection_{false}; // Skip band selection, only calculate baseline
  
  // P95 algorithm constants
  static constexpr uint16_t MVS_WINDOW_SIZE = 50;  // Window size for moving variance
  static constexpr float MVS_THRESHOLD = 1.0f;     // Detection threshold
  static constexpr float SAFE_MARGIN = 0.15f;      // Safety margin below threshold
  static constexpr uint8_t BAND_SELECTION_PERCENTILE = 95;  // Fixed P95 for band selection
  
  // Current calibration context
  std::vector<uint8_t> current_band_;
  uint8_t last_progress_{0};
  
  // Results
  uint8_t selected_band_[12];
  uint8_t selected_band_size_{0};
  std::vector<float> mv_values_;     // MV values for threshold calculation
  
  // Use central HT20 constants from csi_processor.h
  // (HT20_NUM_SUBCARRIERS, HT20_GUARD_BAND_LOW, HT20_GUARD_BAND_HIGH, HT20_DC_SUBCARRIER)
  
  // Constants
  static constexpr uint8_t SELECTED_SUBCARRIERS_COUNT = 12;
  
  // Threshold for null subcarrier detection (mean amplitude below this = null)
  static constexpr float NULL_SUBCARRIER_THRESHOLD = 1.0f;
  
};

// Backward compatibility alias
using CalibrationManager = P95Calibrator;

}  // namespace espectre
}  // namespace esphome
