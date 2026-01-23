/*
 * ESPectre - NBVI Calibrator Implementation
 * 
 * NBVI algorithm for non-consecutive subcarrier selection.
 * Uses file-based storage to avoid RAM limitations.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "nbvi_calibrator.h"
#include "csi_manager.h"
#include "utils.h"
#include "utils.h"
#include "esphome/core/log.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include "esp_spiffs.h"
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
  buffer_path_ = buffer_path;
  ESP_LOGD(TAG, "NBVI Calibrator initialized (buffer: %s)", buffer_path_);
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
  
  result_callback_ = callback;
  current_band_.assign(current_band, current_band + current_band_size);
  
  remove_buffer_file_();
  if (!open_buffer_file_for_writing_()) {
    ESP_LOGE(TAG, "Failed to open buffer file for writing");
    return ESP_ERR_NO_MEM;
  }
  
  buffer_count_ = 0;
  last_progress_ = 0;
  mv_values_.clear();
  
  calibrating_ = true;
  csi_manager_->set_calibration_mode(this);
  
  ESP_LOGI(TAG, "NBVI Calibration Starting");
  
  return ESP_OK;
}

bool NBVICalibrator::add_packet(const int8_t* csi_data, size_t csi_len) {
  if (!calibrating_ || buffer_count_ >= buffer_size_ || !buffer_file_) {
    return buffer_count_ >= buffer_size_;
  }
  
  uint16_t packet_sc = csi_len / 2;
  if (packet_sc != HT20_NUM_SUBCARRIERS) {
    return false;
  }
  
  uint8_t magnitudes[HT20_NUM_SUBCARRIERS];
  
  for (uint16_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
    // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
    int8_t q_val = csi_data[sc * 2];      // Imaginary first
    int8_t i_val = csi_data[sc * 2 + 1];  // Real second
    float mag = calculate_magnitude(i_val, q_val);
    magnitudes[sc] = static_cast<uint8_t>(std::min(mag, 255.0f));
  }
  
  size_t written = fwrite(magnitudes, 1, HT20_NUM_SUBCARRIERS, buffer_file_);
  if (written != HT20_NUM_SUBCARRIERS) {
    ESP_LOGE(TAG, "Failed to write magnitudes to file");
    return false;
  }
  
  buffer_count_++;
  
  if (buffer_count_ % 100 == 0) {
    fflush(buffer_file_);
    vTaskDelay(1);
  }
  
  uint8_t progress = (buffer_count_ * 100) / buffer_size_;
  if (progress >= last_progress_ + 10 || buffer_count_ == buffer_size_) {
    log_progress_bar(TAG, progress / 100.0f, 20, -1,
                     "%d%% (%d/%d)", progress, buffer_count_, buffer_size_);
    last_progress_ = progress;
  }
  
  bool buffer_full = (buffer_count_ >= buffer_size_);
  
  if (buffer_full) {
    on_collection_complete_();
  }
  
  return buffer_full;
}

// ============================================================================
// INTERNAL METHODS
// ============================================================================

void NBVICalibrator::on_collection_complete_() {
  ESP_LOGD(TAG, "Collection complete, processing...");
  
  if (collection_complete_callback_) {
    collection_complete_callback_();
  }
  
  // Stop receiving CSI packets during processing
  csi_manager_->set_calibration_mode(nullptr);
  
  close_buffer_file_();
  
  BaseType_t result = xTaskCreate(
      calibration_task_,
      "nbvi_cal",
      8192,
      this,
      1,
      &calibration_task_handle_
  );
  
  if (result != pdPASS) {
    ESP_LOGE(TAG, "Failed to create calibration task");
    finish_calibration_(false);
  }
}

void NBVICalibrator::calibration_task_(void* arg) {
  NBVICalibrator* self = static_cast<NBVICalibrator*>(arg);
  
  if (!self->open_buffer_file_for_reading_()) {
    ESP_LOGE(TAG, "Failed to open buffer file for reading");
    self->finish_calibration_(false);
    vTaskDelete(NULL);
    return;
  }
  
  esp_err_t err = self->run_calibration_();
  
  bool success = (err == ESP_OK && self->selected_band_size_ == SELECTED_SUBCARRIERS_COUNT);
  
  self->close_buffer_file_();
  self->remove_buffer_file_();
  
  self->finish_calibration_(success);
  
  vTaskDelete(NULL);
}

void NBVICalibrator::finish_calibration_(bool success) {
  calibrating_ = false;
  calibration_task_handle_ = nullptr;
  
  if (result_callback_) {
    result_callback_(selected_band_, selected_band_size_, mv_values_, success);
  }
}

esp_err_t NBVICalibrator::run_calibration_() {
  if (buffer_count_ < MVS_WINDOW_SIZE + 10) {
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
  uint8_t best_band[SELECTED_SUBCARRIERS_COUNT] = {0};
  std::vector<float> best_mv_values;
  
  for (size_t idx = 0; idx < candidates.size(); idx++) {
    uint16_t baseline_start = candidates[idx].start_idx;
    
    // Calculate NBVI for all subcarriers
    std::vector<NBVIMetrics> all_metrics(HT20_NUM_SUBCARRIERS);
    calculate_nbvi_metrics_(baseline_start, all_metrics);
    
    // Apply Noise Gate
    uint8_t filtered_count = apply_noise_gate_(all_metrics);
    
    if (filtered_count < SELECTED_SUBCARRIERS_COUNT) {
      continue;
    }
    
    // Sort by NBVI (ascending)
    std::sort(all_metrics.begin(), all_metrics.begin() + filtered_count,
              [](const NBVIMetrics& a, const NBVIMetrics& b) {
                return a.nbvi < b.nbvi;
              });
    
    // Select with spacing
    uint8_t temp_band[SELECTED_SUBCARRIERS_COUNT] = {0};
    uint8_t temp_band_size = 0;
    select_with_spacing_(all_metrics, temp_band, &temp_band_size);
    
    if (temp_band_size != SELECTED_SUBCARRIERS_COUNT) {
      continue;
    }
    
    // Validate
    float fp_rate = 0.0f;
    std::vector<float> temp_mv_values;
    validate_subcarriers_(temp_band, temp_band_size, &fp_rate, temp_mv_values);
    
    ESP_LOGV(TAG, "Window %zu: FP rate %.1f%%", idx + 1, fp_rate * 100.0f);
    
    if (fp_rate < best_fp_rate) {
      best_fp_rate = fp_rate;
      std::memcpy(best_band, temp_band, SELECTED_SUBCARRIERS_COUNT);
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
  std::memcpy(selected_band_, best_band, SELECTED_SUBCARRIERS_COUNT);
  selected_band_size_ = SELECTED_SUBCARRIERS_COUNT;
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
  
  if (buffer_count_ < window_size_) {
    return ESP_FAIL;
  }
  
  std::vector<WindowVariance> all_windows;
  
  for (uint16_t start = 0; start + window_size_ <= buffer_count_; start += window_step_) {
    std::vector<uint8_t> window_data = read_window_(start, window_size_);
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
                                                       current_band_.size());
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
  float p_threshold = calculate_percentile_(variances, percentile_);
  
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
  std::vector<uint8_t> window_data = read_window_(baseline_start, window_size_);
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
  
  float threshold = calculate_percentile_(valid_means, noise_gate_percentile_);
  
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
  for (size_t i = 5; i < sorted_metrics.size() && selected.size() < SELECTED_SUBCARRIERS_COUNT; i++) {
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
  
  // Fallback: if not enough subcarriers with spacing, add remaining without spacing constraint
  // This matches Python implementation for robustness
  if (selected.size() < SELECTED_SUBCARRIERS_COUNT) {
    for (size_t i = 0; i < sorted_metrics.size() && selected.size() < SELECTED_SUBCARRIERS_COUNT; i++) {
      if (std::isinf(sorted_metrics[i].nbvi)) {
        continue;
      }
      uint8_t sc = sorted_metrics[i].subcarrier;
      // Check if already selected
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
  
  std::vector<float> turbulence_buffer(MVS_WINDOW_SIZE);
  uint16_t motion_count = 0;
  uint16_t total_packets = 0;
  
  for (uint16_t pkt = 0; pkt < buffer_count_; pkt++) {
    std::vector<uint8_t> packet_data = read_window_(pkt, 1);
    if (packet_data.size() != HT20_NUM_SUBCARRIERS) {
      continue;
    }
    
    float float_mags[HT20_NUM_SUBCARRIERS];
    for (uint8_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
      float_mags[sc] = static_cast<float>(packet_data[sc]);
    }
    
    float turbulence = calculate_spatial_turbulence(float_mags, band, band_size);
    
    // Shift buffer
    for (uint16_t i = 0; i < MVS_WINDOW_SIZE - 1; i++) {
      turbulence_buffer[i] = turbulence_buffer[i + 1];
    }
    turbulence_buffer[MVS_WINDOW_SIZE - 1] = turbulence;
    
    if (pkt < MVS_WINDOW_SIZE) {
      continue;
    }
    
    float variance = calculate_variance_two_pass(turbulence_buffer.data(), MVS_WINDOW_SIZE);
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

float NBVICalibrator::calculate_percentile_(const std::vector<float>& values,
                                            uint8_t percentile) const {
  if (values.empty()) {
    return 0.0f;
  }
  
  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  
  size_t n = sorted.size();
  float k = (n - 1) * percentile / 100.0f;
  size_t f = static_cast<size_t>(k);
  size_t c = f + 1;
  
  if (c >= n) {
    return sorted.back();
  }
  
  float d0 = sorted[f] * (c - k);
  float d1 = sorted[c] * (k - f);
  return d0 + d1;
}

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

// ============================================================================
// FILE I/O METHODS
// ============================================================================

bool NBVICalibrator::ensure_spiffs_mounted_() {
  FILE* test = fopen(buffer_path_, "rb");
  if (test) {
    fclose(test);
    return true;
  }
  
  esp_vfs_spiffs_conf_t conf = {
    .base_path = "/spiffs",
    .partition_label = NULL,
    .max_files = 2,
    .format_if_mount_failed = true
  };
  
  esp_err_t ret = esp_vfs_spiffs_register(&conf);
  if (ret != ESP_OK) {
    ESP_LOGE(TAG, "Failed to initialize SPIFFS (%s)", esp_err_to_name(ret));
    return false;
  }
  
  return true;
}

bool NBVICalibrator::open_buffer_file_for_writing_() {
  if (!ensure_spiffs_mounted_()) {
    return false;
  }
  
  buffer_file_ = fopen(buffer_path_, "wb");
  return buffer_file_ != nullptr;
}

bool NBVICalibrator::open_buffer_file_for_reading_() {
  buffer_file_ = fopen(buffer_path_, "rb");
  return buffer_file_ != nullptr;
}

void NBVICalibrator::close_buffer_file_() {
  if (buffer_file_) {
    fclose(buffer_file_);
    buffer_file_ = nullptr;
  }
}

void NBVICalibrator::remove_buffer_file_() {
  FILE* f = fopen(buffer_path_, "wb");
  if (f) {
    fclose(f);
  }
}

std::vector<uint8_t> NBVICalibrator::read_window_(uint16_t start_idx, uint16_t window_size) {
  std::vector<uint8_t> data;
  
  if (!buffer_file_) {
    return data;
  }
  
  size_t bytes_to_read = window_size * HT20_NUM_SUBCARRIERS;
  data.resize(bytes_to_read);
  
  long offset = static_cast<long>(start_idx) * HT20_NUM_SUBCARRIERS;
  if (fseek(buffer_file_, offset, SEEK_SET) != 0) {
    data.clear();
    return data;
  }
  
  size_t bytes_read = fread(data.data(), 1, bytes_to_read, buffer_file_);
  if (bytes_read != bytes_to_read) {
    data.resize(bytes_read);
  }
  
  return data;
}

}  // namespace espectre
}  // namespace esphome
