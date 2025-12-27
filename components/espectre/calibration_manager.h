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
  // Parameters: band, size, normalization_scale, success
  using result_callback_t = std::function<void(const uint8_t* band, uint8_t size, float normalization_scale, bool success)>;
  
  // Callback type for collection complete notification
  // Called when all packets have been collected, before NBVI processing starts.
  // Caller can use this to pause traffic generation during the processing phase.
  using collection_complete_callback_t = std::function<void()>;
  
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
  
  uint16_t get_buffer_size() const { return buffer_size_; }
  uint16_t get_window_size() const { return window_size_; }
  uint16_t get_window_step() const { return window_step_; }
  void set_percentile(uint8_t percentile) { percentile_ = percentile; }
  void set_alpha(float alpha) { alpha_ = alpha; }
  void set_min_spacing(uint8_t spacing) { min_spacing_ = spacing; }
  void set_noise_gate_percentile(uint8_t percentile) { noise_gate_percentile_ = percentile; }
  void set_skip_subcarrier_selection(bool skip) { skip_subcarrier_selection_ = skip; }
  void set_collection_complete_callback(collection_complete_callback_t callback) { 
    collection_complete_callback_ = callback; 
  }
  
  /**
   * Get the baseline variance calculated during calibration
   * 
   * @return Baseline variance (only valid after calibration completes)
   */
  float get_baseline_variance() const { return baseline_variance_; }
  
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
  esp_err_t find_candidate_windows_(std::vector<WindowVariance>& candidates);
  void calculate_nbvi_metrics_(uint16_t baseline_start, std::vector<NBVIMetrics>& metrics);
  uint8_t apply_noise_gate_(std::vector<NBVIMetrics>& metrics);
  void select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                           uint8_t* output_band,
                           uint8_t* output_size);
  bool validate_subcarriers_(const uint8_t* band, uint8_t band_size, float* out_fp_rate);
  
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
  collection_complete_callback_t collection_complete_callback_;
  
  // File-based storage (saves RAM - magnitudes stored as uint8)
  FILE* buffer_file_{nullptr};
  uint16_t buffer_count_{0};
  const char* buffer_path_{"/spiffs/nbvi_buffer.bin"};
  
  // Configuration parameters
  uint16_t buffer_size_{700};         // Number of packets to collect (~7 seconds at 100 Hz)
  uint16_t window_size_{200};        // Window size for baseline detection (2s @ 100Hz)
  uint16_t window_step_{50};         // Step size for sliding window
  uint8_t percentile_{10};           // Percentile for baseline detection
  float alpha_{0.5f};                // NBVI weighting factor (higher = more weight on signal strength)
  uint8_t min_spacing_{1};           // Minimum spectral spacing (1 = adjacent allowed)
  uint8_t noise_gate_percentile_{25}; // Noise gate threshold
  bool skip_subcarrier_selection_{false}; // Skip NBVI, only calculate baseline
  
  // Current calibration context
  std::vector<uint8_t> current_band_;
  uint8_t last_progress_{0};
  
  // Results
  uint8_t selected_band_[12];
  uint8_t selected_band_size_{0};
  float normalization_scale_{1.0f};  // Calculated normalization factor
  float baseline_variance_{1.0f};    // Baseline variance for normalization
  
  // Constants
  static constexpr uint8_t NUM_SUBCARRIERS = 64;
  static constexpr uint8_t SELECTED_SUBCARRIERS_COUNT = 12;
  
  // OFDM 20MHz subcarrier limits for NBVI selection
  // Standard guard bands: [0-5] and [59-63], DC null: [32]
  // We use a more conservative range [11-52] to avoid edge subcarriers
  // which tend to be noisier and cause false positives
  static constexpr uint8_t GUARD_BAND_LOW = 11;  // First valid subcarrier (conservative)
  static constexpr uint8_t GUARD_BAND_HIGH = 52; // Last valid subcarrier (conservative)
  static constexpr uint8_t DC_SUBCARRIER = 32;   // DC null (always excluded)
  
  // Threshold for null subcarrier detection (mean amplitude below this = null)
  static constexpr float NULL_SUBCARRIER_THRESHOLD = 1.0f;
  
  // Helper methods
  float calculate_baseline_variance_(uint16_t baseline_start);
  void calculate_normalization_scale_();
  void log_normalization_status_();
};

}  // namespace espectre
}  // namespace esphome
