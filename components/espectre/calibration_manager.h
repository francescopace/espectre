/*
 * ESPectre - Calibration Manager
 * 
 * Automatic subcarrier band selection using P95 moving variance optimization.
 * Selects optimal 12-subcarrier band for motion detection by minimizing
 * false positive rate.
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
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <atomic>
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
 * Orchestrates P95-based band calibration process:
 * 1. Collects CSI packets during baseline period
 * 2. Evaluates candidate bands using P95 of moving variance
 * 3. Selects optimal 12-subcarrier band for motion detection
 * 
 * Uses P95 optimization to minimize false positive rate.
 */

class CalibrationManager {
 public:
  // Callback type for calibration results
  // Parameters: band, size, adaptive_threshold, success
  using result_callback_t = std::function<void(const uint8_t* band, uint8_t size, float adaptive_threshold, bool success)>;
  
  // Callback type for collection complete notification
  // Called when all packets have been collected, before P95 processing starts.
  // Caller can use this to pause traffic generation during the processing phase.
  using collection_complete_callback_t = std::function<void()>;
  
  /**
   * Initialize calibration manager
   * 
   * @param csi_manager CSI manager instance
   * @param buffer_path Path for temporary calibration file (default: /spiffs/band_buffer.bin)
   */
  void init(CSIManager* csi_manager, const char* buffer_path = "/spiffs/band_buffer.bin");
  
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
  
  /**
   * Set expected number of subcarriers (from GainController stats)
   * 
   * Should be called before start_auto_calibration() to ensure consistent
   * packet handling. Packets with different SC count will be discarded.
   * 
   * @param num_sc Expected subcarrier count (64, 128, or 256)
   */
  void set_expected_subcarriers(uint16_t num_sc);
  
  uint16_t get_buffer_size() const { return buffer_size_; }
  uint16_t get_guard_band_low() const { return guard_band_low_; }
  uint16_t get_guard_band_high() const { return guard_band_high_; }
  void set_skip_subcarrier_selection(bool skip) { skip_subcarrier_selection_ = skip; }
  void set_collection_complete_callback(collection_complete_callback_t callback) { 
    collection_complete_callback_ = callback; 
  }
  
  /**
   * Get the P95 calculated during calibration
   * 
   * @return Best P95 value (only valid after calibration completes)
   */
  float get_best_p95() const { return best_p95_; }
  
 private:
  // Internal structures
  struct BandResult {
    uint8_t start_sc;      // First subcarrier in band
    float p95;             // P95 of moving variance
    float mean_mv;         // Mean moving variance
    float fp_estimate;     // Estimated FP rate
  };
  
  // Internal methods
  void on_collection_complete_();
  esp_err_t run_calibration_();
  static void calibration_task_(void* arg);
  void finish_calibration_(bool success);
  void get_candidate_bands_(std::vector<std::vector<uint8_t>>& candidates);
  BandResult evaluate_band_(const std::vector<uint8_t>& window_data, const uint8_t* band, uint8_t band_size);
  float calculate_p95_(const std::vector<float>& values) const;
  
  // File I/O helpers
  bool ensure_spiffs_mounted_();
  bool open_buffer_file_for_writing_();
  bool open_buffer_file_for_reading_();
  void close_buffer_file_();
  void remove_buffer_file_();
  std::vector<uint8_t> read_window_(uint16_t start_idx, uint16_t window_size);
  
  // Members
  CSIManager* csi_manager_{nullptr};
  std::atomic<bool> calibrating_{false};
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
  static constexpr float ADAPTIVE_THRESHOLD_FACTOR = 1.4f;  // P95 × factor = adaptive threshold
  
  // Current calibration context
  std::vector<uint8_t> current_band_;
  uint8_t last_progress_{0};
  
  // Results
  uint8_t selected_band_[12];
  uint8_t selected_band_size_{0};
  float adaptive_threshold_{1.0f};   // Calculated adaptive threshold (P95 × 1.4)
  float best_p95_{0.0f};             // Best P95 from calibration
  
  // Dynamic subcarrier configuration (determined from first CSI packet)
  uint16_t num_subcarriers_{64};     // Number of subcarriers (64, 128, or 256)
  uint16_t guard_band_low_{11};      // First valid subcarrier
  uint16_t guard_band_high_{52};     // Last valid subcarrier
  uint16_t dc_low_{32};              // DC zone start (for 64/128 SC: single point)
  uint16_t dc_high_{32};             // DC zone end (for 256 SC: range [dc_low_, dc_high_])
  
  // Constants
  static constexpr uint8_t SELECTED_SUBCARRIERS_COUNT = 12;
  
  // Threshold for null subcarrier detection (mean amplitude below this = null)
  static constexpr float NULL_SUBCARRIER_THRESHOLD = 1.0f;
  
  // Helper methods
  void calculate_adaptive_threshold_(float p95);
  void log_adaptive_threshold_status_();
};

}  // namespace espectre
}  // namespace esphome
