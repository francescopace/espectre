/*
 * ESPectre - P95 Calibrator Implementation
 * 
 * P95-based band selection algorithm for optimal subcarrier selection.
 * Inherits common lifecycle from BaseCalibrator.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "p95_calibrator.h"
#include "threshold.h"
#include "csi_manager.h"
#include "utils.h"
#include "esphome/core/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace esphome {
namespace espectre {

static const char *TAG = "P95Calibrator";

// ============================================================================
// PUBLIC API
// ============================================================================

void P95Calibrator::init_subcarrier_config() {
  ESP_LOGI(TAG, "P95: HT20 mode, %d subcarriers, valid range [%d-%d], DC=%d",
           HT20_NUM_SUBCARRIERS, HT20_GUARD_BAND_LOW, HT20_GUARD_BAND_HIGH, HT20_DC_SUBCARRIER);
}

// ============================================================================
// CALIBRATION ALGORITHM
// ============================================================================

esp_err_t P95Calibrator::run_calibration_() {
  uint16_t buffer_count = file_buffer_.get_count();
  
  if (buffer_count < MVS_WINDOW_SIZE + 10) {
    ESP_LOGE(TAG, "Not enough packets for calibration (%d < %d)", 
             buffer_count, MVS_WINDOW_SIZE + 10);
    return ESP_FAIL;
  }
  
  ESP_LOGD(TAG, "Starting P95-based band calibration...");
  ESP_LOGD(TAG, "  Buffer: %d packets, MVS window: %d", buffer_count, MVS_WINDOW_SIZE);
  
  // Read all packets from file
  std::vector<uint8_t> all_data = file_buffer_.read_window(0, buffer_count);
  if (all_data.size() != buffer_count * HT20_NUM_SUBCARRIERS) {
    ESP_LOGE(TAG, "Failed to read calibration data");
    return ESP_FAIL;
  }
  
  // If skipping subcarrier selection, just calculate baseline
  if (skip_subcarrier_selection_) {
    selected_band_size_ = current_band_.size();
    std::memcpy(selected_band_, current_band_.data(), selected_band_size_);
    
    BandResult result = evaluate_band_(all_data, selected_band_, selected_band_size_);
    mv_values_ = std::move(result.mv_values);
    
    ESP_LOGI(TAG, "Baseline calibration complete (fixed subcarriers)");
    ESP_LOGD(TAG, "  P95: %.4f", result.p95);
    
    return ESP_OK;
  }
  
  // Step 1: Generate all candidate bands (12 consecutive subcarriers)
  std::vector<std::vector<uint8_t>> candidates;
  get_candidate_bands_(candidates);
  
  if (candidates.empty()) {
    ESP_LOGE(TAG, "No valid candidate bands found");
    return ESP_FAIL;
  }
  
  ESP_LOGI(TAG, "Evaluating %zu candidate bands...", candidates.size());
  
  // Step 2: Evaluate each candidate band
  std::vector<BandResult> results;
  results.reserve(candidates.size());
  
  for (size_t i = 0; i < candidates.size(); i++) {
    BandResult result = evaluate_band_(all_data, candidates[i].data(), 
                                       static_cast<uint8_t>(candidates[i].size()));
    result.start_sc = candidates[i][0];
    results.push_back(std::move(result));
    
    ESP_LOGV(TAG, "  Band [%d-%d]: P95=%.4f, FP=%.1f%%",
             candidates[i][0], candidates[i].back(), results.back().p95, results.back().fp_estimate * 100.0f);
    
    if (i % 10 == 0) {
      vTaskDelay(1);
    }
  }
  
  // Step 3: Select optimal band
  float p95_limit = MVS_THRESHOLD - SAFE_MARGIN;
  
  size_t best_idx = 0;
  float best_p95 = -1.0f;
  bool found_safe = false;
  
  for (size_t i = 0; i < results.size(); i++) {
    if (results[i].p95 < p95_limit) {
      if (results[i].p95 > best_p95) {
        best_p95 = results[i].p95;
        best_idx = i;
        found_safe = true;
      }
    }
  }
  
  if (!found_safe) {
    ESP_LOGW(TAG, "No bands with P95 < %.2f, using lowest", p95_limit);
    best_p95 = std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < results.size(); i++) {
      if (results[i].p95 < best_p95) {
        best_p95 = results[i].p95;
        best_idx = i;
      }
    }
  } else {
    ESP_LOGD(TAG, "Found safe bands (P95 < %.2f)", p95_limit);
  }
  
  // Copy selected band and mv_values
  std::memcpy(selected_band_, candidates[best_idx].data(), HT20_SELECTED_BAND_SIZE);
  selected_band_size_ = HT20_SELECTED_BAND_SIZE;
  mv_values_ = std::move(results[best_idx].mv_values);
  
  ESP_LOGI(TAG, "P95: Band selection successful");
  ESP_LOGD(TAG, "  Selected: [%d-%d]",
           selected_band_[0], selected_band_[HT20_SELECTED_BAND_SIZE - 1]);
  ESP_LOGD(TAG, "  P95 MV: %.4f", best_p95);
  ESP_LOGD(TAG, "  Est. FP rate: %.1f%%", results[best_idx].fp_estimate * 100.0f);
  
  return ESP_OK;
}

void P95Calibrator::get_candidate_bands_(std::vector<std::vector<uint8_t>>& candidates) {
  candidates.clear();
  
  // HT20: Generate valid bands of 12 consecutive subcarriers
  // Zone before DC (subcarriers 11-31)
  for (uint16_t start = HT20_GUARD_BAND_LOW; start + HT20_SELECTED_BAND_SIZE <= HT20_DC_SUBCARRIER; start++) {
    std::vector<uint8_t> band;
    bool valid = true;
    
    for (uint8_t i = 0; i < HT20_SELECTED_BAND_SIZE; i++) {
      uint8_t sc = start + i;
      if (sc < HT20_GUARD_BAND_LOW || sc > HT20_GUARD_BAND_HIGH || sc == HT20_DC_SUBCARRIER) {
        valid = false;
        break;
      }
      band.push_back(sc);
    }
    
    if (valid && band.size() == HT20_SELECTED_BAND_SIZE) {
      candidates.push_back(band);
    }
  }
  
  // Zone after DC (subcarriers 33-52)
  for (uint16_t start = HT20_DC_SUBCARRIER + 1; start + HT20_SELECTED_BAND_SIZE <= HT20_GUARD_BAND_HIGH + 1; start++) {
    std::vector<uint8_t> band;
    bool valid = true;
    
    for (uint8_t i = 0; i < HT20_SELECTED_BAND_SIZE; i++) {
      uint8_t sc = start + i;
      if (sc < HT20_GUARD_BAND_LOW || sc > HT20_GUARD_BAND_HIGH || sc == HT20_DC_SUBCARRIER) {
        valid = false;
        break;
      }
      band.push_back(sc);
    }
    
    if (valid && band.size() == HT20_SELECTED_BAND_SIZE) {
      candidates.push_back(band);
    }
  }
  
  ESP_LOGD(TAG, "Generated %zu candidate bands (HT20)", candidates.size());
}

P95Calibrator::BandResult P95Calibrator::evaluate_band_(
    const std::vector<uint8_t>& all_data, 
    const uint8_t* band, 
    uint8_t band_size) {
  
  BandResult result = {0, std::numeric_limits<float>::infinity(), 1.0f, {}};
  
  if (all_data.size() < MVS_WINDOW_SIZE * HT20_NUM_SUBCARRIERS) {
    return result;
  }
  
  uint16_t num_packets = all_data.size() / HT20_NUM_SUBCARRIERS;
  
  // Calculate turbulence for each packet
  std::vector<float> turbulences(num_packets);
  float valid_mags[HT20_SELECTED_BAND_SIZE];
  
  for (uint16_t pkt = 0; pkt < num_packets; pkt++) {
    const uint8_t* packet_data = &all_data[pkt * HT20_NUM_SUBCARRIERS];
    
    for (uint8_t i = 0; i < band_size && i < HT20_SELECTED_BAND_SIZE; i++) {
      valid_mags[i] = static_cast<float>(packet_data[band[i]]);
    }
    
    turbulences[pkt] = std::sqrt(calculate_variance_two_pass(valid_mags, band_size));
  }
  
  // Calculate moving variance series
  result.mv_values.reserve(num_packets - MVS_WINDOW_SIZE);
  
  for (uint16_t i = MVS_WINDOW_SIZE; i < num_packets; i++) {
    float variance = calculate_variance_two_pass(&turbulences[i - MVS_WINDOW_SIZE], MVS_WINDOW_SIZE);
    result.mv_values.push_back(variance);
  }
  
  if (result.mv_values.empty()) {
    return result;
  }
  
  // Calculate P95 for band selection
  result.p95 = calculate_percentile(result.mv_values, BAND_SELECTION_PERCENTILE);
  
  // Estimate FP rate
  uint32_t fp_count = 0;
  for (float v : result.mv_values) {
    if (v > MVS_THRESHOLD) {
      fp_count++;
    }
  }
  result.fp_estimate = static_cast<float>(fp_count) / result.mv_values.size();
  
  return result;
}


}  // namespace espectre
}  // namespace esphome
