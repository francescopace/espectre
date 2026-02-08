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

#include "base_calibrator.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>
#include <vector>

namespace esphome {
namespace espectre {

/**
 * NBVI Calibrator
 * 
 * Automatic subcarrier selection using NBVI algorithm.
 * Selects 12 non-consecutive subcarriers with lowest baseline variability.
 */
class NBVICalibrator : public BaseCalibrator {
 public:
  /**
   * Initialize NBVI calibrator (ICalibrator interface)
   */
  void init(CSIManager* csi_manager) override {
    BaseCalibrator::init(csi_manager, "/spiffs/nbvi_buffer.bin");
  }
  
  /**
   * Initialize NBVI calibrator with custom buffer path
   */
  void init(CSIManager* csi_manager, const char* buffer_path) {
    BaseCalibrator::init(csi_manager, buffer_path);
  }
  
  /**
   * Configuration setters
   */
  void set_window_size(uint16_t size) { window_size_ = size; }
  void set_window_step(uint16_t step) { window_step_ = step; }
  void set_percentile(uint8_t percentile) { percentile_ = percentile; }
  void set_alpha(float alpha) { alpha_ = alpha; }
  void set_min_spacing(uint8_t spacing) { min_spacing_ = spacing; }
  void set_noise_gate_percentile(uint8_t percentile) { noise_gate_percentile_ = percentile; }
  
 protected:
  const char* get_tag_() const override { return "NBVI"; }
  const char* get_task_name_() const override { return "nbvi_cal"; }
  esp_err_t run_calibration_() override;
  
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
  esp_err_t find_candidate_windows_(std::vector<WindowVariance>& candidates);
  void calculate_nbvi_metrics_(uint16_t baseline_start, std::vector<NBVIMetrics>& metrics);
  uint8_t apply_noise_gate_(std::vector<NBVIMetrics>& metrics);
  void select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                           uint8_t* output_band, uint8_t* output_size);
  bool validate_subcarriers_(const uint8_t* band, uint8_t band_size, 
                            float* out_fp_rate, std::vector<float>& out_mv_values);
  void calculate_nbvi_weighted_(const std::vector<float>& magnitudes, NBVIMetrics& out_metrics) const;
  
  // Configuration parameters
  uint16_t window_size_{200};
  uint16_t window_step_{50};
  uint8_t percentile_{10};
  float alpha_{0.5f};
  uint8_t min_spacing_{1};
  uint8_t noise_gate_percentile_{25};
};

}  // namespace espectre
}  // namespace esphome
