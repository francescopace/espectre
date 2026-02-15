/*
 * ESPectre - NBVI Calibrator Implementation
 * 
 * NBVI algorithm for non-consecutive subcarrier selection.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "nbvi_calibrator.h"
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

static const char *TAG = "NBVI";

// ============================================================================
// PUBLIC API
// ============================================================================

void NBVICalibrator::init(CSIManager* csi_manager, const char* buffer_path) {
  csi_manager_ = csi_manager;
  file_buffer_.init(buffer_path);
}

esp_err_t NBVICalibrator::start_calibration(const uint8_t* current_band,
                                            uint8_t current_band_size,
                                            result_callback_t callback) {
  if (!csi_manager_) {
    ESP_LOGE(TAG, "CSI Manager not initialized");
    return ESP_ERR_INVALID_STATE;
  }
  
  if (calibrating_) {
    ESP_LOGW(TAG, "Calibration already in progress");
    return ESP_ERR_INVALID_STATE;
  }
  
  // Store context
  result_callback_ = callback;
  current_band_.assign(current_band, current_band + current_band_size);
  
  // Prepare file buffer
  file_buffer_.remove_file();
  if (!file_buffer_.open_for_writing()) {
    ESP_LOGE(TAG, "Failed to open buffer file for writing");
    return ESP_ERR_NO_MEM;
  }
  
  file_buffer_.reset();
  mv_values_.clear();
  
  calibrating_ = true;
  csi_manager_->set_calibration_mode(this);
  
  ESP_LOGI(TAG, "Calibration starting");
  
  return ESP_OK;
}

bool NBVICalibrator::add_packet(const int8_t* csi_data, size_t csi_len) {
  if (!calibrating_ || file_buffer_.is_full() || !file_buffer_.is_open()) {
    return file_buffer_.is_full();
  }
  
  bool full = file_buffer_.write_packet(csi_data, csi_len);
  
  if (full) {
    on_collection_complete_();
  }
  
  return full;
}

// ============================================================================
// LIFECYCLE MANAGEMENT
// ============================================================================

void NBVICalibrator::on_collection_complete_() {
  ESP_LOGD(TAG, "Collection complete, processing...");
  
  // Notify caller that collection is complete (can pause traffic generator)
  if (collection_complete_callback_) {
    collection_complete_callback_();
  }
  
  // Stop receiving CSI packets during processing
  csi_manager_->set_calibration_mode(nullptr);
  
  // Close write mode - file will be reopened for reading in calibration task
  file_buffer_.close();
  
  // Launch calibration in a separate task to avoid blocking the CSI callback
  BaseType_t result = xTaskCreate(
      calibration_task_wrapper_,
      "nbvi_cal",
      8192,  // 8KB stack for calibration calculations
      this,
      1,     // Low priority
      &calibration_task_handle_
  );
  
  if (result != pdPASS) {
    ESP_LOGE(TAG, "Failed to create calibration task");
    finish_calibration_(false);
  }
}

void NBVICalibrator::calibration_task_wrapper_(void* arg) {
  NBVICalibrator* self = static_cast<NBVICalibrator*>(arg);
  
  // Open buffer file for reading
  if (!self->file_buffer_.open_for_reading()) {
    ESP_LOGE(TAG, "Failed to open buffer file for reading");
    self->finish_calibration_(false);
    vTaskDelete(NULL);
    return;
  }
  
  // Run calibration algorithm
  esp_err_t err = self->run_calibration_();
  
  bool success = (err == ESP_OK && self->selected_band_size_ == HT20_SELECTED_BAND_SIZE);
  
  // Cleanup file
  self->file_buffer_.close();
  self->file_buffer_.remove_file();
  
  // Notify completion
  self->finish_calibration_(success);
  
  // Self-terminate
  vTaskDelete(NULL);
}

void NBVICalibrator::finish_calibration_(bool success) {
  calibrating_ = false;
  calibration_task_handle_ = nullptr;
  
  if (result_callback_) {
    result_callback_(selected_band_, selected_band_size_, mv_values_, success);
  }
}

// ============================================================================
// CALIBRATION ALGORITHM
// ============================================================================

esp_err_t NBVICalibrator::run_calibration_() {
  uint16_t buffer_count = file_buffer_.get_count();
  
  if (buffer_count < mvs_window_size_ + 10) {
    ESP_LOGE(TAG, "Not enough packets for calibration");
    return ESP_FAIL;
  }
  
  ESP_LOGD(TAG, "Starting NBVI calibration...");
  
  // Step 1: Find candidate baseline windows
  std::vector<WindowVariance> candidates;
  esp_err_t err = find_candidate_windows_(candidates);
  if (err != ESP_OK || candidates.empty()) {
    ESP_LOGE(TAG, "Failed to find candidate windows");
    return ESP_FAIL;
  }
  
  ESP_LOGD(TAG, "Found %zu candidate windows", candidates.size());
  
  // Step 2: Evaluate each candidate window
  float best_fp_rate = 1.0f;
  bool found_valid = false;
  uint8_t best_band[HT20_SELECTED_BAND_SIZE] = {0};
  std::vector<float> best_mv_values;
  
  for (size_t idx = 0; idx < candidates.size(); idx++) {
    uint16_t baseline_start = candidates[idx].start_idx;
    
    // Calculate NBVI for all subcarriers
    std::vector<NBVIMetrics> all_metrics(HT20_NUM_SUBCARRIERS);
    calculate_nbvi_metrics_(baseline_start, all_metrics);
    
    // Apply Noise Gate
    uint8_t filtered_count = apply_noise_gate_(all_metrics);
    
    if (filtered_count < HT20_SELECTED_BAND_SIZE) {
      continue;
    }
    
    // Sort by NBVI (ascending)
    std::sort(all_metrics.begin(), all_metrics.begin() + filtered_count,
              [](const NBVIMetrics& a, const NBVIMetrics& b) {
                return a.nbvi < b.nbvi;
              });
    
    // Select with spacing
    uint8_t temp_band[HT20_SELECTED_BAND_SIZE] = {0};
    uint8_t temp_band_size = 0;
    select_with_spacing_(all_metrics, temp_band, &temp_band_size);
    
    if (temp_band_size != HT20_SELECTED_BAND_SIZE) {
      continue;
    }
    
    // Validate
    float fp_rate = 0.0f;
    std::vector<float> temp_mv_values;
    validate_subcarriers_(temp_band, temp_band_size, &fp_rate, temp_mv_values);
    
    ESP_LOGV(TAG, "Window %zu: FP rate %.1f%%", idx + 1, fp_rate * 100.0f);
    
    if (fp_rate < best_fp_rate) {
      best_fp_rate = fp_rate;
      std::memcpy(best_band, temp_band, HT20_SELECTED_BAND_SIZE);
      best_mv_values = std::move(temp_mv_values);
      found_valid = true;
    }
    
    vTaskDelay(1);  // Yield
  }
  
  if (!found_valid) {
    ESP_LOGW(TAG, "All candidate windows failed - using default subcarriers");
    
    selected_band_size_ = current_band_.size();
    std::memcpy(selected_band_, current_band_.data(), selected_band_size_);
    
    // Get MV values for default band
    float fp_rate;
    validate_subcarriers_(selected_band_, selected_band_size_, &fp_rate, mv_values_);
    
    ESP_LOGI(TAG, "Fallback to default band");
    return ESP_OK;
  }
  
  // Store results
  std::memcpy(selected_band_, best_band, HT20_SELECTED_BAND_SIZE);
  selected_band_size_ = HT20_SELECTED_BAND_SIZE;
  mv_values_ = std::move(best_mv_values);
  
  ESP_LOGI(TAG, "NBVI Calibration successful");
  ESP_LOGD(TAG, "  Band: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
           selected_band_[0], selected_band_[1], selected_band_[2], selected_band_[3],
           selected_band_[4], selected_band_[5], selected_band_[6], selected_band_[7],
           selected_band_[8], selected_band_[9], selected_band_[10], selected_band_[11]);
  ESP_LOGD(TAG, "  Est. FP rate: %.1f%%", best_fp_rate * 100.0f);
  
  return ESP_OK;
}

esp_err_t NBVICalibrator::find_candidate_windows_(std::vector<WindowVariance>& candidates) {
  candidates.clear();
  
  uint16_t buffer_count = file_buffer_.get_count();
  
  if (buffer_count < window_size_) {
    return ESP_FAIL;
  }
  
  std::vector<WindowVariance> all_windows;
  
  for (uint16_t start = 0; start + window_size_ <= buffer_count; start += window_step_) {
    std::vector<uint8_t> window_data = file_buffer_.read_window(start, window_size_);
    if (window_data.size() != window_size_ * HT20_NUM_SUBCARRIERS) {
      continue;
    }
    
    // Calculate turbulence for each packet using current band
    std::vector<float> turbulences(window_size_);
    
    for (uint16_t pkt = 0; pkt < window_size_; pkt++) {
      const uint8_t* packet_magnitudes = &window_data[pkt * HT20_NUM_SUBCARRIERS];
      
      float float_mags[HT20_NUM_SUBCARRIERS];
      for (uint8_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
        float_mags[sc] = static_cast<float>(packet_magnitudes[sc]);
      }
      
      turbulences[pkt] = calculate_spatial_turbulence(float_mags, current_band_.data(),
                                                       current_band_.size(), 64,
                                                       use_cv_normalization_);
    }
    
    float variance = calculate_variance_two_pass(turbulences.data(), window_size_);
    
    WindowVariance wv;
    wv.start_idx = start;
    wv.variance = variance;
    all_windows.push_back(wv);
    
    vTaskDelay(1);
  }
  
  if (all_windows.empty()) {
    return ESP_FAIL;
  }
  
  // Sort by variance and select best windows
  std::sort(all_windows.begin(), all_windows.end(),
            [](const WindowVariance& a, const WindowVariance& b) {
              return a.variance < b.variance;
            });
  
  // Get percentile threshold
  std::vector<float> variances;
  for (const auto& w : all_windows) {
    variances.push_back(w.variance);
  }
  float p_threshold = calculate_percentile(variances, percentile_);
  
  // Select windows below threshold
  for (const auto& w : all_windows) {
    if (w.variance <= p_threshold) {
      candidates.push_back(w);
    }
  }
  
  return ESP_OK;
}

void NBVICalibrator::calculate_nbvi_metrics_(uint16_t baseline_start,
                                             std::vector<NBVIMetrics>& metrics) {
  std::vector<uint8_t> window_data = file_buffer_.read_window(baseline_start, window_size_);
  if (window_data.size() != window_size_ * HT20_NUM_SUBCARRIERS) {
    return;
  }
  
  for (uint8_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
    std::vector<float> magnitudes(window_size_);
    
    for (uint16_t pkt = 0; pkt < window_size_; pkt++) {
      magnitudes[pkt] = static_cast<float>(window_data[pkt * HT20_NUM_SUBCARRIERS + sc]);
    }
    
    metrics[sc].subcarrier = sc;
    calculate_nbvi_weighted_(magnitudes, metrics[sc]);
    
    // Exclude guard bands and DC
    if (sc < HT20_GUARD_BAND_LOW || sc > HT20_GUARD_BAND_HIGH || sc == HT20_DC_SUBCARRIER) {
      metrics[sc].nbvi = std::numeric_limits<float>::infinity();
    } else if (metrics[sc].mean < NULL_SUBCARRIER_THRESHOLD) {
      metrics[sc].nbvi = std::numeric_limits<float>::infinity();
    }
  }
}

uint8_t NBVICalibrator::apply_noise_gate_(std::vector<NBVIMetrics>& metrics) {
  // Collect valid means
  std::vector<float> valid_means;
  for (const auto& m : metrics) {
    if (m.mean >= NULL_SUBCARRIER_THRESHOLD && !std::isinf(m.nbvi)) {
      valid_means.push_back(m.mean);
    }
  }
  
  if (valid_means.empty()) {
    return 0;
  }
  
  float threshold = calculate_percentile(valid_means, noise_gate_percentile_);
  
  // Move filtered subcarriers to front
  uint8_t count = 0;
  for (size_t i = 0; i < metrics.size(); i++) {
    if (metrics[i].mean >= threshold && !std::isinf(metrics[i].nbvi)) {
      if (i != count) {
        std::swap(metrics[count], metrics[i]);
      }
      count++;
    }
  }
  
  return count;
}

void NBVICalibrator::select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                                          uint8_t* output_band,
                                          uint8_t* output_size) {
  std::vector<uint8_t> selected;
  
  // Always include top 5 (best NBVI)
  for (size_t i = 0; i < sorted_metrics.size() && selected.size() < 5; i++) {
    if (!std::isinf(sorted_metrics[i].nbvi)) {
      selected.push_back(sorted_metrics[i].subcarrier);
    }
  }
  
  // Remaining with spacing
  for (size_t i = 5; i < sorted_metrics.size() && selected.size() < HT20_SELECTED_BAND_SIZE; i++) {
    if (std::isinf(sorted_metrics[i].nbvi)) {
      continue;
    }
    
    uint8_t candidate = sorted_metrics[i].subcarrier;
    bool valid = true;
    
    for (uint8_t existing : selected) {
      if (std::abs(static_cast<int>(candidate) - static_cast<int>(existing)) < min_spacing_) {
        valid = false;
        break;
      }
    }
    
    if (valid) {
      selected.push_back(candidate);
    }
  }
  
  // Fallback: add remaining without spacing constraint
  if (selected.size() < HT20_SELECTED_BAND_SIZE) {
    for (size_t i = 0; i < sorted_metrics.size() && selected.size() < HT20_SELECTED_BAND_SIZE; i++) {
      if (std::isinf(sorted_metrics[i].nbvi)) {
        continue;
      }
      uint8_t sc = sorted_metrics[i].subcarrier;
      bool already_selected = false;
      for (uint8_t existing : selected) {
        if (existing == sc) {
          already_selected = true;
          break;
        }
      }
      if (!already_selected) {
        selected.push_back(sc);
      }
    }
  }
  
  std::sort(selected.begin(), selected.end());
  
  *output_size = selected.size();
  std::memcpy(output_band, selected.data(), selected.size());
}

bool NBVICalibrator::validate_subcarriers_(const uint8_t* band, uint8_t band_size,
                                           float* out_fp_rate,
                                           std::vector<float>& out_mv_values) {
  out_mv_values.clear();
  
  uint16_t buffer_count = file_buffer_.get_count();
  
  std::vector<float> turbulence_buffer(mvs_window_size_);
  uint16_t motion_count = 0;
  uint16_t total_packets = 0;
  
  for (uint16_t pkt = 0; pkt < buffer_count; pkt++) {
    std::vector<uint8_t> packet_data = file_buffer_.read_window(pkt, 1);
    if (packet_data.size() != HT20_NUM_SUBCARRIERS) {
      continue;
    }
    
    float float_mags[HT20_NUM_SUBCARRIERS];
    for (uint8_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
      float_mags[sc] = static_cast<float>(packet_data[sc]);
    }
    
    float turbulence = calculate_spatial_turbulence(float_mags, band, band_size, 64,
                                                     use_cv_normalization_);
    
    // Shift buffer
    for (uint16_t i = 0; i < mvs_window_size_ - 1; i++) {
      turbulence_buffer[i] = turbulence_buffer[i + 1];
    }
    turbulence_buffer[mvs_window_size_ - 1] = turbulence;
    
    if (pkt < mvs_window_size_) {
      continue;
    }
    
    float variance = calculate_variance_two_pass(turbulence_buffer.data(), mvs_window_size_);
    out_mv_values.push_back(variance);
    
    if (variance > MVS_THRESHOLD) {
      motion_count++;
    }
    total_packets++;
  }
  
  *out_fp_rate = (total_packets > 0) ? static_cast<float>(motion_count) / total_packets : 0.0f;
  
  return true;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

void NBVICalibrator::calculate_nbvi_weighted_(const std::vector<float>& magnitudes,
                                              NBVIMetrics& out_metrics) const {
  size_t count = magnitudes.size();
  if (count == 0) {
    out_metrics.nbvi = std::numeric_limits<float>::infinity();
    out_metrics.mean = 0.0f;
    out_metrics.std = 0.0f;
    return;
  }
  
  float sum = 0.0f;
  for (float mag : magnitudes) {
    sum += mag;
  }
  float mean = sum / count;
  
  if (mean < 1e-6f) {
    out_metrics.nbvi = std::numeric_limits<float>::infinity();
    out_metrics.mean = mean;
    out_metrics.std = 0.0f;
    return;
  }
  
  float variance = calculate_variance_two_pass(magnitudes.data(), count);
  float stddev = std::sqrt(variance);
  
  float cv = stddev / mean;
  float nbvi_energy = stddev / (mean * mean);
  float nbvi_weighted = alpha_ * nbvi_energy + (1.0f - alpha_) * cv;
  
  out_metrics.nbvi = nbvi_weighted;
  out_metrics.mean = mean;
  out_metrics.std = stddev;
}

}  // namespace espectre
}  // namespace esphome
