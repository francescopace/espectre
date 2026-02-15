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

#include "calibration_file_buffer.h"
#include "base_detector.h"  // For DETECTOR_DEFAULT_WINDOW_SIZE
#include "utils.h"
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
 * NBVI Calibrator
 * 
 * Automatic subcarrier selection using NBVI algorithm.
 * Selects 12 non-consecutive subcarriers with lowest baseline variability.
 * 
 * Lifecycle:
 * 1. init() - initialize with CSI manager and buffer path
 * 2. start_calibration() - begin collecting packets
 * 3. add_packet() - called by CSI manager for each packet
 * 4. When buffer full: run calibration algorithm in background task
 * 5. Invoke callback with results (band, mv_values, success)
 */
class NBVICalibrator {
 public:
  // Callback types
  using result_callback_t = std::function<void(const uint8_t* band, uint8_t size, 
                                               const std::vector<float>& cal_values, bool success)>;
  using collection_complete_callback_t = std::function<void()>;
  
  // ========================================================================
  // Initialization
  // ========================================================================
  
  /**
   * Initialize calibrator with default buffer path
   */
  void init(CSIManager* csi_manager) {
    init(csi_manager, "/spiffs/nbvi_buffer.bin");
  }
  
  /**
   * Initialize calibrator with custom buffer path
   */
  void init(CSIManager* csi_manager, const char* buffer_path);
  
  // ========================================================================
  // Calibration API
  // ========================================================================
  
  /**
   * Start calibration process
   * 
   * @param current_band Current/fallback subcarrier selection
   * @param current_band_size Size of current band
   * @param callback Callback to invoke with results
   * @return ESP_OK on success
   */
  esp_err_t start_calibration(const uint8_t* current_band,
                              uint8_t current_band_size,
                              result_callback_t callback);
  
  /**
   * Add CSI packet to calibration buffer
   * 
   * @param csi_data Raw CSI data (I/Q pairs)
   * @param csi_len Length of CSI data
   * @return true if buffer is full and calibration should proceed
   */
  bool add_packet(const int8_t* csi_data, size_t csi_len);
  
  /**
   * Check if calibration is in progress
   */
  bool is_calibrating() const { return calibrating_; }
  
  /**
   * Set callback for collection complete notification
   */
  void set_collection_complete_callback(collection_complete_callback_t callback) {
    collection_complete_callback_ = callback;
  }
  
  // ========================================================================
  // Configuration
  // ========================================================================
  
  void set_buffer_size(uint16_t size) { file_buffer_.set_size(size); }
  uint16_t get_buffer_size() const { return file_buffer_.get_size(); }
  
  void set_mvs_window_size(uint16_t size) { mvs_window_size_ = size; }
  uint16_t get_mvs_window_size() const { return mvs_window_size_; }
  
  void set_window_size(uint16_t size) { window_size_ = size; }
  void set_window_step(uint16_t step) { window_step_ = step; }
  void set_percentile(uint8_t percentile) { percentile_ = percentile; }
  void set_alpha(float alpha) { alpha_ = alpha; }
  void set_min_spacing(uint8_t spacing) { min_spacing_ = spacing; }
  void set_noise_gate_percentile(uint8_t percentile) { noise_gate_percentile_ = percentile; }
  
  /**
   * Set CV normalization mode for turbulence calculation during calibration.
   * Must match the detector's normalization mode.
   */
  void set_cv_normalization(bool enabled) { use_cv_normalization_ = enabled; }
  bool is_cv_normalization_enabled() const { return use_cv_normalization_; }
  
 private:
  // ========================================================================
  // Internal structures
  // ========================================================================
  
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
  
  // ========================================================================
  // Lifecycle management
  // ========================================================================
  
  void on_collection_complete_();
  static void calibration_task_wrapper_(void* arg);
  void finish_calibration_(bool success);
  
  // ========================================================================
  // Calibration algorithm
  // ========================================================================
  
  esp_err_t run_calibration_();
  esp_err_t find_candidate_windows_(std::vector<WindowVariance>& candidates);
  void calculate_nbvi_metrics_(uint16_t baseline_start, std::vector<NBVIMetrics>& metrics);
  uint8_t apply_noise_gate_(std::vector<NBVIMetrics>& metrics);
  void select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                           uint8_t* output_band, uint8_t* output_size);
  bool validate_subcarriers_(const uint8_t* band, uint8_t band_size, 
                            float* out_fp_rate, std::vector<float>& out_mv_values);
  void calculate_nbvi_weighted_(const std::vector<float>& magnitudes, NBVIMetrics& out_metrics) const;
  
  // ========================================================================
  // State
  // ========================================================================
  
  CSIManager* csi_manager_{nullptr};
  CalibrationFileBuffer file_buffer_;
  std::vector<uint8_t> current_band_;
  
  // Results
  uint8_t selected_band_[12]{};
  uint8_t selected_band_size_{0};
  std::vector<float> mv_values_;
  
  // Calibration state
  bool calibrating_{false};
  result_callback_t result_callback_;
  collection_complete_callback_t collection_complete_callback_;
  TaskHandle_t calibration_task_handle_{nullptr};
  
  // MVS window size for validation (uses DETECTOR_DEFAULT_WINDOW_SIZE)
  uint16_t mvs_window_size_{DETECTOR_DEFAULT_WINDOW_SIZE};
  
  // CV normalization mode
  bool use_cv_normalization_{false};
  
  // NBVI algorithm parameters
  uint16_t window_size_{200};
  uint16_t window_step_{50};
  uint8_t percentile_{10};
  float alpha_{0.5f};
  uint8_t min_spacing_{1};
  uint8_t noise_gate_percentile_{25};
  
  // Shared constants
  static constexpr float MVS_THRESHOLD = 1.0f;
  static constexpr float NULL_SUBCARRIER_THRESHOLD = 1.0f;
};

}  // namespace espectre
}  // namespace esphome
