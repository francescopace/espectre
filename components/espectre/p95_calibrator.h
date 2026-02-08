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
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "base_calibrator.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace esphome {
namespace espectre {

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
class P95Calibrator : public BaseCalibrator {
 public:
  /**
   * Initialize calibrator (ICalibrator interface)
   */
  void init(CSIManager* csi_manager) override { 
    BaseCalibrator::init(csi_manager, "/spiffs/band_buffer.bin"); 
  }
  
  /**
   * Initialize calibrator with custom buffer path
   */
  void init(CSIManager* csi_manager, const char* buffer_path) {
    BaseCalibrator::init(csi_manager, buffer_path);
  }
  
  /**
   * Initialize subcarrier configuration (HT20 only)
   */
  void init_subcarrier_config();
  
  void set_skip_subcarrier_selection(bool skip) { skip_subcarrier_selection_ = skip; }
  
  /**
   * Get the moving variance values from calibration
   */
  const std::vector<float>& get_mv_values() const { return mv_values_; }
  
 protected:
  const char* get_tag_() const override { return "P95Calibrator"; }
  const char* get_task_name_() const override { return "p95_cal"; }
  esp_err_t run_calibration_() override;
  
 private:
  // Internal structures
  struct BandResult {
    uint8_t start_sc;
    float p95;
    float fp_estimate;
    std::vector<float> mv_values;
  };
  
  // Internal methods
  void get_candidate_bands_(std::vector<std::vector<uint8_t>>& candidates);
  BandResult evaluate_band_(const std::vector<uint8_t>& window_data, const uint8_t* band, uint8_t band_size);
  
  // Configuration
  bool skip_subcarrier_selection_{false};
  
  // P95 algorithm constants
  static constexpr float SAFE_MARGIN = 0.15f;
  static constexpr uint8_t BAND_SELECTION_PERCENTILE = 95;
};

// Backward compatibility alias
using CalibrationManager = P95Calibrator;

}  // namespace espectre
}  // namespace esphome
