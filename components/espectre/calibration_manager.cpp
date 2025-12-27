/*
 * ESPectre - Calibration Manager Implementation
 * 
 * Uses file-based storage to avoid RAM limitations.
 * Magnitudes stored as uint8 (max CSI magnitude ~181 fits in 1 byte).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "calibration_manager.h"
#include "csi_manager.h"
#include "utils.h"
#include "esphome/core/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cerrno>
#include <limits>
#include <unistd.h>
#include "esp_spiffs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

namespace esphome {
namespace espectre {

static const char *TAG = "Calibration";

// ============================================================================
// PUBLIC API
// ============================================================================

void CalibrationManager::init(CSIManager* csi_manager, const char* buffer_path) {
  csi_manager_ = csi_manager;
  buffer_path_ = buffer_path;
  ESP_LOGD(TAG, "Calibration Manager initialized (buffer: %s)", buffer_path_);
}

esp_err_t CalibrationManager::start_auto_calibration(const uint8_t* current_band,
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
  
  // Remove old buffer file and open new one for writing
  remove_buffer_file_();
  if (!open_buffer_file_for_writing_()) {
    ESP_LOGE(TAG, "Failed to open buffer file for writing");
    return ESP_ERR_NO_MEM;
  }
  
  buffer_count_ = 0;
  last_progress_ = 0;
  
  // Set calibration mode in CSI Manager
  calibrating_ = true;
  csi_manager_->set_calibration_mode(this);
  
  ESP_LOGI(TAG, "Auto-Calibration Starting (file-based storage)");
  
  return ESP_OK;
}

bool CalibrationManager::add_packet(const int8_t* csi_data, size_t csi_len) {
  if (!calibrating_ || buffer_count_ >= buffer_size_ || !buffer_file_) {
    return buffer_count_ >= buffer_size_;
  }
  
  if (csi_len < 128) {
    ESP_LOGW(TAG, "CSI data too short: %zu bytes (need 128)", csi_len);
    return false;
  }
  
  // Calculate magnitudes and write directly to file as uint8
  // (max CSI magnitude ~181 fits in 1 byte, saves RAM)
  uint8_t magnitudes[NUM_SUBCARRIERS];
  
  for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
    int8_t i_val = csi_data[sc * 2];
    int8_t q_val = csi_data[sc * 2 + 1];
    float mag = calculate_magnitude(i_val, q_val);
    magnitudes[sc] = static_cast<uint8_t>(std::min(mag, 255.0f));
  }
  
  // Write to file
  size_t written = fwrite(magnitudes, 1, NUM_SUBCARRIERS, buffer_file_);
  if (written != NUM_SUBCARRIERS) {
    ESP_LOGE(TAG, "Failed to write magnitudes to file");
    return false;
  }
  
  buffer_count_++;
  
  // Flush periodically to ensure data is written
  // Also yield to allow other tasks (especially WiFi/LwIP) to process
  // This helps prevent ENOMEM errors in the traffic generator on ESP32-S3
  // where PSRAM, SPIFFS, and WiFi compete for bus access
  if (buffer_count_ % 100 == 0) {
    fflush(buffer_file_);
    vTaskDelay(1);  // Minimal yield to prevent WiFi starvation
  }
  
  // Log progress bar every 10%
  uint8_t progress = (buffer_count_ * 100) / buffer_size_;
  if (progress >= last_progress_ + 10 || buffer_count_ == buffer_size_) {
    log_progress_bar(TAG, progress / 100.0f, 20, -1,
                     "%d%% (%d/%d)",
                     progress, buffer_count_, buffer_size_);
    last_progress_ = progress;
  }
  
  // Check if buffer is full
  bool buffer_full = (buffer_count_ >= buffer_size_);
  
  if (buffer_full) {
    on_collection_complete_();
  }
  
  return buffer_full;
}

// ============================================================================
// INTERNAL METHODS
// ============================================================================

void CalibrationManager::on_collection_complete_() {
  ESP_LOGD(TAG, "NBVI: Collection complete, processing...");
  
  // Notify caller that collection is complete (can pause traffic generator)
  if (collection_complete_callback_) {
    collection_complete_callback_();
  }
  
  // Close write mode and reopen for reading
  close_buffer_file_();
  if (!open_buffer_file_for_reading_()) {
    ESP_LOGE(TAG, "Failed to open buffer file for reading");
    if (result_callback_) {
      result_callback_(nullptr, 0, 1.0f, false);
    }
    calibrating_ = false;
    csi_manager_->set_calibration_mode(nullptr);
    return;
  }
  
  // Run calibration
  esp_err_t err = run_calibration_();
  
  bool success = (err == ESP_OK && selected_band_size_ == SELECTED_SUBCARRIERS_COUNT);
  
  // Cleanup
  calibrating_ = false;
  csi_manager_->set_calibration_mode(nullptr);
  close_buffer_file_();
  remove_buffer_file_();
  
  // Call user callback with results (including normalization scale)
  if (result_callback_) {
    result_callback_(selected_band_, selected_band_size_, normalization_scale_, success);
  }
}

esp_err_t CalibrationManager::run_calibration_() {
  if (buffer_count_ < buffer_size_) {
    ESP_LOGE(TAG, "Buffer not full (%d < %d)", buffer_count_, buffer_size_);
    return ESP_FAIL;
  }
  
  ESP_LOGD(TAG, "Starting calibration...");
  ESP_LOGV(TAG, "  Window size: %d packets", window_size_);
  ESP_LOGV(TAG, "  Step size: %d packets", window_step_);
  
  // Step 1: Find all candidate baseline windows (sorted by variance, best first)
  std::vector<WindowVariance> candidates;
  esp_err_t err = find_candidate_windows_(candidates);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "Failed to find candidate windows");
    return err;
  }
  
  // If skipping subcarrier selection (user specified subcarriers), use first candidate
  if (skip_subcarrier_selection_) {
    // Use the current band as-is (already set in current_band_)
    selected_band_size_ = current_band_.size();
    std::memcpy(selected_band_, current_band_.data(), selected_band_size_);
    
    // Variance was calculated with current_band_, which is correct for fixed subcarriers
    baseline_variance_ = candidates[0].variance;
    calculate_normalization_scale_();
    
    ESP_LOGI(TAG, "✓ Baseline calibration complete (fixed subcarriers)");
    ESP_LOGD(TAG, "  Baseline variance: %.4f", baseline_variance_);
    log_normalization_status_();
    
    return ESP_OK;
  }
  
  // Step 2: Try all candidate windows and select the one with minimum FP rate
  float best_fp_rate = 1.0f;
  size_t best_idx = 0;
  bool found_valid = false;
  uint8_t best_band[SELECTED_SUBCARRIERS_COUNT] = {0};
  uint16_t best_baseline_start = 0;
  std::vector<NBVIMetrics> best_metrics;
  uint8_t best_filtered_count = 0;
  
  for (size_t candidate_idx = 0; candidate_idx < candidates.size(); candidate_idx++) {
    uint16_t baseline_start = candidates[candidate_idx].start_idx;
    float window_variance = candidates[candidate_idx].variance;
    
    ESP_LOGV(TAG, "Evaluating candidate window %zu/%zu (start=%d, variance=%.4f)",
             candidate_idx + 1, candidates.size(), baseline_start, window_variance);
    
    // Step 2a: Calculate NBVI for all subcarriers using this window
    std::vector<NBVIMetrics> all_metrics(NUM_SUBCARRIERS);
    calculate_nbvi_metrics_(baseline_start, all_metrics);
    
    // Step 2b: Apply Noise Gate
    uint8_t filtered_count = apply_noise_gate_(all_metrics);
    
    if (filtered_count < SELECTED_SUBCARRIERS_COUNT) {
      ESP_LOGV(TAG, "  Skipping: not enough subcarriers after Noise Gate (%d < %d)",
               filtered_count, SELECTED_SUBCARRIERS_COUNT);
      continue;
    }
    
    // Step 2c: Sort by NBVI (ascending - lower is better)
    std::sort(all_metrics.begin(), all_metrics.begin() + filtered_count,
              [](const NBVIMetrics& a, const NBVIMetrics& b) {
                return a.nbvi < b.nbvi;
              });
    
    // Step 2d: Select with spectral spacing
    uint8_t temp_band[SELECTED_SUBCARRIERS_COUNT] = {0};
    uint8_t temp_band_size = 0;
    select_with_spacing_(all_metrics, temp_band, &temp_band_size);
    
    if (temp_band_size != SELECTED_SUBCARRIERS_COUNT) {
      ESP_LOGV(TAG, "  Skipping: invalid band size (%d != %d)",
               temp_band_size, SELECTED_SUBCARRIERS_COUNT);
      continue;
    }
    
    // Step 2e: Validate - run MVS on entire buffer with selected subcarriers
    float fp_rate = 0.0f;
    validate_subcarriers_(temp_band, temp_band_size, &fp_rate);
    
    ESP_LOGV(TAG, "  Window %zu: FP rate %.1f%%", candidate_idx + 1, fp_rate * 100.0f);
    
    // Track best result (minimum FP rate)
    if (fp_rate < best_fp_rate) {
      best_fp_rate = fp_rate;
      best_idx = candidate_idx;
      best_baseline_start = baseline_start;
      std::memcpy(best_band, temp_band, SELECTED_SUBCARRIERS_COUNT);
      best_metrics = all_metrics;
      best_filtered_count = filtered_count;
      found_valid = true;
    }
  }
  
  if (!found_valid) {
    ESP_LOGW(TAG, "All %zu candidate windows failed - using default subcarriers with normalization fallback", candidates.size());
    
    // Fallback: keep default subcarriers but still calculate normalization
    // Use the first (best) candidate window for baseline variance
    selected_band_size_ = current_band_.size();
    std::memcpy(selected_band_, current_band_.data(), selected_band_size_);
    
    // Calculate baseline variance using current (default) subcarriers
    baseline_variance_ = calculate_baseline_variance_(candidates[0].start_idx);
    if (baseline_variance_ < 0.01f) {
      baseline_variance_ = 1.0f;  // Fallback
    }
    calculate_normalization_scale_();
    
    ESP_LOGI(TAG, "⚠ Fallback calibration: default subcarriers with normalization");
    ESP_LOGD(TAG, "  Baseline variance: %.4f", baseline_variance_);
    log_normalization_status_();
    
    // Return ESP_FAIL to signal subcarrier selection failed,
    // but normalization_scale_ is now set correctly
    return ESP_FAIL;
  }
  
  // Use the best result
  std::memcpy(selected_band_, best_band, SELECTED_SUBCARRIERS_COUNT);
  selected_band_size_ = SELECTED_SUBCARRIERS_COUNT;
  
  ESP_LOGD(TAG, "Selected window %zu/%zu with minimum FP rate %.1f%%",
           best_idx + 1, candidates.size(), best_fp_rate * 100.0f);
  
  // Step 3: Calculate baseline variance using the SELECTED subcarriers
  baseline_variance_ = calculate_baseline_variance_(best_baseline_start);
  if (baseline_variance_ < 0.01f) {
    baseline_variance_ = 1.0f;  // Fallback
  }
  calculate_normalization_scale_();
  
  // Calculate average metrics for selected band
  float avg_nbvi = 0.0f;
  float avg_mean = 0.0f;
  for (uint8_t i = 0; i < SELECTED_SUBCARRIERS_COUNT; i++) {
    for (uint8_t j = 0; j < best_filtered_count; j++) {
      if (best_metrics[j].subcarrier == selected_band_[i]) {
        avg_nbvi += best_metrics[j].nbvi;
        avg_mean += best_metrics[j].mean;
        break;
      }
    }
  }
  avg_nbvi /= SELECTED_SUBCARRIERS_COUNT;
  avg_mean /= SELECTED_SUBCARRIERS_COUNT;
  
  ESP_LOGI(TAG, "✓ Calibration successful: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
           selected_band_[0], selected_band_[1], selected_band_[2], selected_band_[3],
           selected_band_[4], selected_band_[5], selected_band_[6], selected_band_[7],
           selected_band_[8], selected_band_[9], selected_band_[10], selected_band_[11]);
  ESP_LOGD(TAG, "  Average NBVI: %.6f", avg_nbvi);
  ESP_LOGD(TAG, "  Average magnitude: %.2f", avg_mean);
  ESP_LOGD(TAG, "  Baseline variance: %.4f", baseline_variance_);
  ESP_LOGD(TAG, "  Best of %zu windows (FP rate: %.1f%%)", candidates.size(), best_fp_rate * 100.0f);
  log_normalization_status_();
  
  return ESP_OK;
}

esp_err_t CalibrationManager::find_candidate_windows_(std::vector<WindowVariance>& candidates) {
  if (buffer_count_ < window_size_) {
    ESP_LOGE(TAG, "Not enough packets for baseline detection (%d < %d)",
             buffer_count_, window_size_);
    return ESP_FAIL;
  }
  
  // Calculate number of windows
  uint16_t num_windows = 0;
  for (uint16_t i = 0; i <= buffer_count_ - window_size_; i += window_step_) {
    num_windows++;
  }
  
  if (num_windows == 0) {
    ESP_LOGE(TAG, "No windows to analyze");
    return ESP_FAIL;
  }
  
  ESP_LOGV(TAG, "Analyzing %d windows (size=%d, step=%d)",
           num_windows, window_size_, window_step_);
  
  candidates.resize(num_windows);
  std::vector<float> turbulence_buffer(window_size_);
  
  // Note: CalibrationManager expects data collected AFTER gain lock completes.
  // The gain lock period (first 300 packets after boot) is handled by CSIManager
  // which only starts feeding CalibrationManager after AGC stabilization.
  
  // Analyze each window - read from file
  uint16_t window_idx = 0;
  for (uint16_t i = 0; i <= buffer_count_ - window_size_; i += window_step_) {
    // Read window from file
    std::vector<uint8_t> window_data = read_window_(i, window_size_);
    if (window_data.size() != window_size_ * NUM_SUBCARRIERS) {
      ESP_LOGW(TAG, "Failed to read window at %d", i);
      continue;
    }
    
    // Calculate turbulence for each packet in window
    for (uint16_t j = 0; j < window_size_; j++) {
      const uint8_t* packet_magnitudes = &window_data[j * NUM_SUBCARRIERS];
      
      // Convert uint8 magnitudes to float for turbulence calculation
      float float_mags[NUM_SUBCARRIERS];
      for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
        float_mags[sc] = static_cast<float>(packet_magnitudes[sc]);
      }
      
      turbulence_buffer[j] = calculate_spatial_turbulence(float_mags,
                                                          current_band_.data(),
                                                          static_cast<uint8_t>(current_band_.size()));
    }
    
    // Calculate variance of turbulence
    float variance = calculate_variance_two_pass(turbulence_buffer.data(), window_size_);
    
    candidates[window_idx].start_idx = i;
    candidates[window_idx].variance = variance;
    window_idx++;
  }
  
  // Resize to actual count
  candidates.resize(window_idx);
  
  // Calculate percentile threshold (variances must be sorted!)
  std::vector<float> variances(window_idx);
  for (uint16_t i = 0; i < window_idx; i++) {
    variances[i] = candidates[i].variance;
  }
  std::sort(variances.begin(), variances.end());
  
  float p_threshold = calculate_percentile_(variances, percentile_);
  
  // Filter: keep only windows below percentile threshold
  auto new_end = std::remove_if(candidates.begin(), candidates.end(),
                                [p_threshold](const WindowVariance& w) {
                                  return w.variance > p_threshold;
                                });
  candidates.erase(new_end, candidates.end());
  
  // Sort by variance (ascending - best first)
  std::sort(candidates.begin(), candidates.end(),
            [](const WindowVariance& a, const WindowVariance& b) {
              return a.variance < b.variance;
            });
  
  ESP_LOGD(TAG, "Found %zu candidate windows (p%d threshold: %.4f)",
           candidates.size(), percentile_, p_threshold);
  
  return candidates.empty() ? ESP_FAIL : ESP_OK;
}

void CalibrationManager::calculate_nbvi_metrics_(uint16_t baseline_start,
                                                std::vector<NBVIMetrics>& metrics) {
  // Read baseline window from file
  std::vector<uint8_t> window_data = read_window_(baseline_start, window_size_);
  if (window_data.size() != window_size_ * NUM_SUBCARRIERS) {
    ESP_LOGE(TAG, "Failed to read baseline window");
    return;
  }
  
  std::vector<float> subcarrier_magnitudes(window_size_);
  uint8_t null_count = 0;
  
  // Process ALL subcarriers - detect null ones dynamically based on signal strength
  // This is environment-aware: works with any chip and adapts to local RF conditions
  for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
    // Extract magnitude series for this subcarrier from baseline window
    for (uint16_t i = 0; i < window_size_; i++) {
      subcarrier_magnitudes[i] = static_cast<float>(window_data[i * NUM_SUBCARRIERS + sc]);
    }
    
    // Calculate NBVI
    metrics[sc].subcarrier = sc;
    calculate_nbvi_weighted_(subcarrier_magnitudes, metrics[sc]);
    
    // Exclude guard bands and DC subcarrier (always invalid regardless of signal)
    // OFDM 20MHz: [0-5] lower guard, [32] DC, [59-63] upper guard
    if (sc < GUARD_BAND_LOW || sc > GUARD_BAND_HIGH || sc == DC_SUBCARRIER) {
      metrics[sc].nbvi = std::numeric_limits<float>::infinity();
      null_count++;
    }
    // Auto-detect weak subcarriers: if mean < threshold, mark as invalid
    else if (metrics[sc].mean < NULL_SUBCARRIER_THRESHOLD) {
      metrics[sc].nbvi = std::numeric_limits<float>::infinity();
      null_count++;
    }
  }
  
  ESP_LOGD(TAG, "Excluded %d subcarriers (guard bands + weak signals)", null_count);
}

uint8_t CalibrationManager::apply_noise_gate_(std::vector<NBVIMetrics>& metrics) {
  if (metrics.empty()) return 0;
  
  // Extract NON-ZERO means only (skip invalid subcarriers with mean=0)
  std::vector<float> valid_means;
  valid_means.reserve(metrics.size());
  for (size_t i = 0; i < metrics.size(); i++) {
    if (metrics[i].mean > 1.0f) {  // Only consider subcarriers with actual signal
      valid_means.push_back(metrics[i].mean);
    }
  }
  
  if (valid_means.empty()) {
    ESP_LOGW(TAG, "Noise Gate: No valid subcarriers found");
    return 0;
  }
  
  std::sort(valid_means.begin(), valid_means.end());
  
  float threshold = calculate_percentile_(valid_means, noise_gate_percentile_);
  
  // Filter metrics (move valid ones to front)
  auto new_end = std::remove_if(metrics.begin(), metrics.end(),
                                [threshold](const NBVIMetrics& m) {
                                  return m.mean < threshold;
                                });
  
  uint8_t filtered_count = std::distance(metrics.begin(), new_end);
  uint8_t excluded = metrics.size() - filtered_count;
  
  ESP_LOGD(TAG, "Noise Gate: %d subcarriers excluded (threshold: %.2f, valid: %zu)",
           excluded, threshold, valid_means.size());
  
  return filtered_count;
}

void CalibrationManager::select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                                             uint8_t* output_band,
                                             uint8_t* output_size) {
  std::vector<uint8_t> selected;
  selected.reserve(SELECTED_SUBCARRIERS_COUNT);
  
  // Phase 1: Top 5 absolute best
  for (uint8_t i = 0; i < 5 && i < sorted_metrics.size(); i++) {
    selected.push_back(sorted_metrics[i].subcarrier);
  }
  
  ESP_LOGD(TAG, "Top 5 selected: [%d, %d, %d, %d, %d]",
           selected[0], selected[1], selected[2], selected[3], selected[4]);
  
  // Phase 2: Remaining 7 with spacing
  for (size_t i = 5; i < sorted_metrics.size() && selected.size() < SELECTED_SUBCARRIERS_COUNT; i++) {
    uint8_t candidate = sorted_metrics[i].subcarrier;
    
    // Check spacing with already selected
    bool spacing_ok = true;
    for (uint8_t sel : selected) {
      uint8_t dist = (candidate > sel) ? (candidate - sel) : (sel - candidate);
      if (dist < min_spacing_) {
        spacing_ok = false;
        break;
      }
    }
    
    if (spacing_ok) {
      selected.push_back(candidate);
    }
  }
  
  // If not enough with spacing, add best remaining regardless
  if (selected.size() < SELECTED_SUBCARRIERS_COUNT) {
    for (size_t i = 5; i < sorted_metrics.size() && selected.size() < SELECTED_SUBCARRIERS_COUNT; i++) {
      uint8_t candidate = sorted_metrics[i].subcarrier;
      
      // Check if already selected
      if (std::find(selected.begin(), selected.end(), candidate) == selected.end()) {
        selected.push_back(candidate);
      }
    }
  }
  
  // Sort output band
  std::sort(selected.begin(), selected.end());
  
  std::memcpy(output_band, selected.data(), selected.size());
  *output_size = selected.size();
  
  ESP_LOGD(TAG, "Selected %zu subcarriers with spacing Δf≥%d",
           selected.size(), min_spacing_);
}

// ============================================================================
// UTILITY METHODS
// ============================================================================


float CalibrationManager::calculate_percentile_(const std::vector<float>& sorted_values,
                                               uint8_t percentile) const {
  size_t n = sorted_values.size();
  if (n == 0) return 0.0f;
  if (n == 1) return sorted_values[0];
  
  // Linear interpolation between closest ranks
  float k = (n - 1) * percentile / 100.0f;
  size_t f = static_cast<size_t>(k);
  size_t c = f + 1;
  
  if (c >= n) {
    return sorted_values[n - 1];
  }
  
  float d0 = sorted_values[f] * (c - k);
  float d1 = sorted_values[c] * (k - f);
  return d0 + d1;
}

void CalibrationManager::calculate_nbvi_weighted_(const std::vector<float>& magnitudes,
                                                 NBVIMetrics& out_metrics) const {
  size_t count = magnitudes.size();
  if (count == 0) {
    out_metrics.nbvi = INFINITY;
    out_metrics.mean = 0.0f;
    out_metrics.std = 0.0f;
    return;
  }
  
  // Calculate mean
  float sum = 0.0f;
  for (float mag : magnitudes) {
    sum += mag;
  }
  float mean = sum / count;
  
  if (mean < 1e-6f) {
    out_metrics.nbvi = INFINITY;
    out_metrics.mean = mean;
    out_metrics.std = 0.0f;
    return;
  }
  
  // Calculate standard deviation
  float variance = calculate_variance_two_pass(magnitudes.data(), count);
  float stddev = std::sqrt(variance);
  
  // NBVI Weighted (configurable alpha, default 0.5)
  float cv = stddev / mean;                      // Coefficient of variation
  float nbvi_energy = stddev / (mean * mean);    // Energy normalization
  float nbvi_weighted = alpha_ * nbvi_energy + (1.0f - alpha_) * cv;
  
  out_metrics.nbvi = nbvi_weighted;
  out_metrics.mean = mean;
  out_metrics.std = stddev;
}

// ============================================================================
// FILE I/O METHODS
// ============================================================================

bool CalibrationManager::ensure_spiffs_mounted_() {
  // Check if already mounted
  FILE* test = fopen(buffer_path_, "rb");
  if (test) {
    fclose(test);
    return true;  // SPIFFS is working
  }
  
  // Try to mount SPIFFS
  esp_vfs_spiffs_conf_t conf = {
    .base_path = "/spiffs",
    .partition_label = NULL,  // Use default partition
    .max_files = 2,
    .format_if_mount_failed = true
  };
  
  esp_err_t ret = esp_vfs_spiffs_register(&conf);
  if (ret != ESP_OK) {
    if (ret == ESP_ERR_NOT_FOUND) {
      ESP_LOGE(TAG, "SPIFFS partition not found! ESPectre requires SPIFFS for calibration.");
    } else if (ret == ESP_FAIL) {
      ESP_LOGE(TAG, "Failed to mount or format SPIFFS");
    } else {
      ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
    }
    return false;
  }
  
  size_t total = 0, used = 0;
  esp_spiffs_info(NULL, &total, &used);
  ESP_LOGI(TAG, "SPIFFS mounted: %zu KB total, %zu KB used", total / 1024, used / 1024);
  
  return true;
}

bool CalibrationManager::open_buffer_file_for_writing_() {
  // Ensure SPIFFS is mounted before opening file
  if (!ensure_spiffs_mounted_()) {
    return false;
  }
  
  buffer_file_ = fopen(buffer_path_, "wb");
  if (!buffer_file_) {
    ESP_LOGE(TAG, "Failed to open %s for writing", buffer_path_);
    return false;
  }
  return true;
}

bool CalibrationManager::open_buffer_file_for_reading_() {
  buffer_file_ = fopen(buffer_path_, "rb");
  if (!buffer_file_) {
    ESP_LOGE(TAG, "Failed to open %s for reading", buffer_path_);
    return false;
  }
  return true;
}

void CalibrationManager::close_buffer_file_() {
  if (buffer_file_) {
    fclose(buffer_file_);
    buffer_file_ = nullptr;
  }
}

void CalibrationManager::remove_buffer_file_() {
  // truncate the file
  FILE* f = fopen(buffer_path_, "wb");
  if (f) {
    fclose(f);
  }
}

std::vector<uint8_t> CalibrationManager::read_window_(uint16_t start_idx, uint16_t window_size) {
  std::vector<uint8_t> data;
  
  if (!buffer_file_) {
    ESP_LOGE(TAG, "Buffer file not open for reading");
    return data;
  }
  
  size_t bytes_to_read = window_size * NUM_SUBCARRIERS;
  data.resize(bytes_to_read);
  
  // Seek to window start
  long offset = static_cast<long>(start_idx) * NUM_SUBCARRIERS;
  if (fseek(buffer_file_, offset, SEEK_SET) != 0) {
    ESP_LOGE(TAG, "Failed to seek to offset %ld", offset);
    data.clear();
    return data;
  }
  
  // Read window data
  size_t bytes_read = fread(data.data(), 1, bytes_to_read, buffer_file_);
  if (bytes_read != bytes_to_read) {
    ESP_LOGW(TAG, "Read %zu bytes, expected %zu", bytes_read, bytes_to_read);
    data.resize(bytes_read);
  }
  
  return data;
}

bool CalibrationManager::validate_subcarriers_(const uint8_t* band, uint8_t band_size, float* out_fp_rate) {
  // Validate selected subcarriers by running MVS detection on the ENTIRE buffer
  // and checking for false positives (motion detected in baseline data)
  //
  // The calibration buffer contains ONLY baseline data (quiet room), so any
  // motion detection is a false positive.
  //
  // Returns true always (caller decides based on fp_rate)
  
  constexpr uint16_t MVS_WINDOW_SIZE = SEGMENTATION_DEFAULT_WINDOW_SIZE;  // Use production MVS window (50)
  constexpr float MVS_THRESHOLD = SEGMENTATION_DEFAULT_THRESHOLD;  // Use production threshold (1.0)
  
  if (buffer_count_ < MVS_WINDOW_SIZE) {
    ESP_LOGW(TAG, "Not enough packets for validation");
    *out_fp_rate = 0.0f;
    return true;  // Skip validation if not enough data
  }
  
  // Create a temporary turbulence buffer for MVS (filled sequentially, not circular)
  std::vector<float> turbulence_buffer(MVS_WINDOW_SIZE);
  uint16_t motion_count = 0;
  uint16_t total_packets = 0;
  
  // Process entire buffer packet by packet
  for (uint16_t pkt = 0; pkt < buffer_count_; pkt++) {
    // Read single packet
    std::vector<uint8_t> packet_data = read_window_(pkt, 1);
    if (packet_data.size() != NUM_SUBCARRIERS) {
      continue;
    }
    
    // Convert to float
    float float_mags[NUM_SUBCARRIERS];
    for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
      float_mags[sc] = static_cast<float>(packet_data[sc]);
    }
    
    // Calculate turbulence with the candidate subcarriers
    float turbulence = calculate_spatial_turbulence(float_mags, band, band_size);
    
    // Shift buffer and add new value (like real MVS does)
    for (uint16_t i = 0; i < MVS_WINDOW_SIZE - 1; i++) {
      turbulence_buffer[i] = turbulence_buffer[i + 1];
    }
    turbulence_buffer[MVS_WINDOW_SIZE - 1] = turbulence;
    
    // Skip warmup (need full window before calculating variance)
    if (pkt < MVS_WINDOW_SIZE) {
      continue;
    }
    
    // Calculate variance (MVS) - this is the moving variance
    float variance = calculate_variance_two_pass(turbulence_buffer.data(), MVS_WINDOW_SIZE);
    
    // Check if this would trigger motion detection
    if (variance > MVS_THRESHOLD) {
      motion_count++;
    }
    total_packets++;
  }
  
  *out_fp_rate = (total_packets > 0) ? (float)motion_count / total_packets : 0.0f;
  
  ESP_LOGV(TAG, "Validation: %d/%d packets detected as motion (FP rate: %.1f%%)",
           motion_count, total_packets, *out_fp_rate * 100.0f);
  
  return true;  // Always succeed, caller selects best based on fp_rate
}

float CalibrationManager::calculate_baseline_variance_(uint16_t baseline_start) {
  // Recalculate baseline variance using the SELECTED subcarriers (selected_band_)
  // This is called after NBVI selection to get accurate variance for the actual band used
  
  std::vector<uint8_t> window_data = read_window_(baseline_start, window_size_);
  if (window_data.size() != window_size_ * NUM_SUBCARRIERS) {
    ESP_LOGW(TAG, "Failed to read window for variance recalculation");
    return 0.0f;
  }
  
  std::vector<float> turbulence_buffer(window_size_);
  
  // Calculate turbulence for each packet using the SELECTED subcarriers
  for (uint16_t j = 0; j < window_size_; j++) {
    const uint8_t* packet_magnitudes = &window_data[j * NUM_SUBCARRIERS];
    
    // Convert uint8 magnitudes to float for turbulence calculation
    float float_mags[NUM_SUBCARRIERS];
    for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
      float_mags[sc] = static_cast<float>(packet_magnitudes[sc]);
    }
    
    // Use selected_band_ instead of current_band_
    turbulence_buffer[j] = calculate_spatial_turbulence(float_mags,
                                                        selected_band_,
                                                        selected_band_size_);
  }
  
  // Calculate variance of turbulence
  return calculate_variance_two_pass(turbulence_buffer.data(), window_size_);
}

void CalibrationManager::calculate_normalization_scale_() {
  // Normalize only if baseline variance is ABOVE target (0.25)
  // If baseline <= 0.25, no scaling needed (already in good range)
  // If baseline > 0.25, attenuate to bring it down to 0.25
  // This prevents over-amplification of weak signals
  constexpr float TARGET_BASELINE = 0.25f;
  
  if (baseline_variance_ > TARGET_BASELINE) {
    // Baseline too high - attenuate to reach 0.25
    normalization_scale_ = TARGET_BASELINE / baseline_variance_;
    // Clamp to min 0.1 to avoid extreme attenuation
    if (normalization_scale_ < 0.1f) normalization_scale_ = 0.1f;
  } else {
    // Baseline already at or below target - no scaling
    normalization_scale_ = 1.0f;
  }
}

void CalibrationManager::log_normalization_status_() {
  ESP_LOGI(TAG, "  Normalization scale: %.4f", normalization_scale_);
}

}  // namespace espectre
}  // namespace esphome
