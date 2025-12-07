/*
 * ESPectre - Calibration Manager
 * 
 * Manages NBVI (Normalized Baseline Variability Index) auto-calibration.
 * Automatically selects optimal 12 subcarriers for motion detection.
 * 
 * Uses file-based storage to avoid RAM limitations. Magnitudes stored as
 * uint8 (max CSI magnitude ~181 fits in 1 byte). This allows collecting
 * thousands of packets without memory issues.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esp_err.h"
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
 * Calibration Manager
 * 
 * Orchestrates NBVI auto-calibration process:
 * 1. Collects CSI packets during baseline period
 * 2. Analyzes subcarrier stability using NBVI algorithm
 * 3. Selects optimal 12 subcarriers for motion detection
 * 
 * Uses adaptive baseline detection and spectral spacing for robust selection.
 */
class CalibrationManager {
 public:
  // Callback type for calibration results
  using result_callback_t = std::function<void(const uint8_t* band, uint8_t size, bool success)>;
  
  /**
   * Initialize calibration manager
   * 
   * @param csi_manager CSI manager instance
   * @param buffer_path Path for temporary calibration file (default: /spiffs/nbvi_buffer.bin)
   */
  void init(CSIManager* csi_manager, const char* buffer_path = "/spiffs/nbvi_buffer.bin");
  
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
  esp_err_t start_auto_calibration(const uint8_t* current_band,
                                   uint8_t current_band_size,
                                   result_callback_t callback);
  
  /**
   * Add CSI packet to calibration buffer
   * 
   * Called by CSI Manager during calibration mode.
   * 
   * @param csi_data Raw CSI data (I/Q pairs)
   * @param csi_len Length of CSI data
   * @return true if buffer is full and calibration should proceed
   */
  bool add_packet(const int8_t* csi_data, size_t csi_len);
  
  /**
   * Check if calibration is in progress
   * 
   * @return true if calibrating, false otherwise
   */
  bool is_calibrating() const { return calibrating_; }
  
  /**
   * Configuration setters (optional, use before start_auto_calibration)
   */
  void set_buffer_size(uint16_t size) { buffer_size_ = size; }
  void set_window_size(uint16_t size) { window_size_ = size; }
  void set_window_step(uint16_t step) { window_step_ = step; }
  void set_percentile(uint8_t percentile) { percentile_ = percentile; }
  void set_alpha(float alpha) { alpha_ = alpha; }
  void set_min_spacing(uint8_t spacing) { min_spacing_ = spacing; }
  void set_noise_gate_percentile(uint8_t percentile) { noise_gate_percentile_ = percentile; }
  
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
  esp_err_t run_calibration_();
  esp_err_t find_baseline_window_(uint16_t* out_window_start);
  void calculate_nbvi_metrics_(uint16_t baseline_start, std::vector<NBVIMetrics>& metrics);
  uint8_t apply_noise_gate_(std::vector<NBVIMetrics>& metrics);
  void select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                           uint8_t* output_band,
                           uint8_t* output_size);
  
  // Utility methods
  float calculate_percentile_(const std::vector<float>& sorted_values, uint8_t percentile) const;
  void calculate_nbvi_weighted_(const std::vector<float>& magnitudes,
                                NBVIMetrics& out_metrics) const;
  
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
  
  // File-based storage (saves RAM - magnitudes stored as uint8)
  FILE* buffer_file_{nullptr};
  uint16_t buffer_count_{0};
  const char* buffer_path_{"/spiffs/nbvi_buffer.bin"};
  
  // Configuration parameters
  uint16_t buffer_size_{1000};        // Number of packets to collect
  uint16_t window_size_{100};        // Window size for baseline detection
  uint16_t window_step_{50};         // Step size for sliding window
  uint8_t percentile_{10};           // Percentile for baseline detection
  float alpha_{0.3f};                // NBVI weighting factor
  uint8_t min_spacing_{3};           // Minimum spectral spacing
  uint8_t noise_gate_percentile_{10}; // Noise gate threshold
  
  // Current calibration context
  std::vector<uint8_t> current_band_;
  uint8_t last_progress_{0};
  
  // Results
  uint8_t selected_band_[12];
  uint8_t selected_band_size_{0};
  
  // Constants
  static constexpr uint8_t NUM_SUBCARRIERS = 64;
  static constexpr uint8_t SELECTED_SUBCARRIERS_COUNT = 12;
};

}  // namespace espectre
}  // namespace esphome
