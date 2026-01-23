/*
 * ESPectre - P95 Calibrator Implementation
 * 
 * P95-based band selection algorithm for optimal subcarrier selection.
 * Uses file-based storage to avoid RAM limitations.
 * Magnitudes stored as uint8 (max CSI magnitude ~181 fits in 1 byte).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "p95_calibrator.h"
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

static const char *TAG = "P95Calibrator";

// ============================================================================
// PUBLIC API
// ============================================================================

void P95Calibrator::init(CSIManager* csi_manager, const char* buffer_path) {
  csi_manager_ = csi_manager;
  buffer_path_ = buffer_path;
  ESP_LOGD(TAG, "P95 Calibrator initialized (buffer: %s)", buffer_path_);
}

esp_err_t P95Calibrator::start_calibration(const uint8_t* current_band,
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
  
  calibrating_ = true;
  csi_manager_->set_calibration_mode(this);
  
  ESP_LOGI(TAG, "P95 Auto-Calibration Starting (file-based storage)");
  
  return ESP_OK;
}

void P95Calibrator::init_subcarrier_config() {
  // HT20 only: 64 subcarriers with fixed guard bands
  // Exclude noisy edges 0-10 and 53-63, DC at 32
  ESP_LOGI(TAG, "P95: HT20 mode, %d subcarriers, valid range [%d-%d], DC=%d",
           HT20_NUM_SUBCARRIERS, HT20_GUARD_BAND_LOW, HT20_GUARD_BAND_HIGH, HT20_DC_SUBCARRIER);
}

bool P95Calibrator::add_packet(const int8_t* csi_data, size_t csi_len) {
  if (!calibrating_ || buffer_count_ >= buffer_size_ || !buffer_file_) {
    return buffer_count_ >= buffer_size_;
  }
  
  // Validate packet has 64 subcarriers (HT20 only)
  uint16_t packet_sc = csi_len / 2;
  if (packet_sc != HT20_NUM_SUBCARRIERS) {
    // Discard packets with wrong SC count
    return false;
  }
  
  // Calculate magnitudes and write directly to file as uint8
  // (max CSI magnitude ~181 fits in 1 byte, saves RAM)
  // HT20: 64 subcarriers
  uint8_t magnitudes[HT20_NUM_SUBCARRIERS];
  
  for (uint16_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
    // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
    int8_t q_val = csi_data[sc * 2];      // Imaginary first
    int8_t i_val = csi_data[sc * 2 + 1];  // Real second
    float mag = calculate_magnitude(i_val, q_val);
    magnitudes[sc] = static_cast<uint8_t>(std::min(mag, 255.0f));
  }
  
  // Write to file
  size_t written = fwrite(magnitudes, 1, HT20_NUM_SUBCARRIERS, buffer_file_);
  if (written != HT20_NUM_SUBCARRIERS) {
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

void P95Calibrator::on_collection_complete_() {
  ESP_LOGD(TAG, "P95: Collection complete, processing...");
  
  // Notify caller that collection is complete (can pause traffic generator)
  if (collection_complete_callback_) {
    collection_complete_callback_();
  }
  
  // Stop receiving CSI packets during processing
  csi_manager_->set_calibration_mode(nullptr);
  
  // Close write mode - file will be reopened in calibration task
  close_buffer_file_();
  
  // Launch calibration in a separate task to avoid blocking main loop
  BaseType_t result = xTaskCreate(
      calibration_task_,
      "p95_cal",
      8192,  // 8KB stack for P95 calculations
      this,
      1,     // Low priority - doesn't need to be fast
      &calibration_task_handle_
  );
  
  if (result != pdPASS) {
    ESP_LOGE(TAG, "Failed to create calibration task");
    finish_calibration_(false);
  }
}

void P95Calibrator::calibration_task_(void* arg) {
  P95Calibrator* self = static_cast<P95Calibrator*>(arg);
  
  // Open buffer file for reading
  if (!self->open_buffer_file_for_reading_()) {
    ESP_LOGE(TAG, "Failed to open buffer file for reading");
    self->finish_calibration_(false);
    vTaskDelete(NULL);
    return;
  }
  
  // Run P95 band calibration
  esp_err_t err = self->run_calibration_();
  
  bool success = (err == ESP_OK && self->selected_band_size_ == SELECTED_SUBCARRIERS_COUNT);
  
  // Cleanup file
  self->close_buffer_file_();
  self->remove_buffer_file_();
  
  // Notify completion
  self->finish_calibration_(success);
  
  // Self-terminate
  vTaskDelete(NULL);
}

void P95Calibrator::finish_calibration_(bool success) {
  calibrating_ = false;
  calibration_task_handle_ = nullptr;
  
  if (result_callback_) {
    result_callback_(selected_band_, selected_band_size_, mv_values_, success);
  }
}

esp_err_t P95Calibrator::run_calibration_() {
  if (buffer_count_ < MVS_WINDOW_SIZE + 10) {
    ESP_LOGE(TAG, "Not enough packets for calibration (%d < %d)", 
             buffer_count_, MVS_WINDOW_SIZE + 10);
    return ESP_FAIL;
  }
  
  ESP_LOGD(TAG, "Starting P95-based band calibration...");
  ESP_LOGD(TAG, "  Buffer: %d packets, MVS window: %d", buffer_count_, MVS_WINDOW_SIZE);
  
  // Read all packets from file
  std::vector<uint8_t> all_data = read_window_(0, buffer_count_);
  if (all_data.size() != buffer_count_ * HT20_NUM_SUBCARRIERS) {
    ESP_LOGE(TAG, "Failed to read calibration data");
    return ESP_FAIL;
  }
  
  // If skipping subcarrier selection (user specified subcarriers), just calculate baseline
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
    
    // Yield periodically to prevent watchdog timeout
    if (i % 10 == 0) {
      vTaskDelay(1);
    }
  }
  
  // Step 3: Select optimal band using P95 (fixed algorithm)
  // Strategy: Find bands with P95 < (threshold - margin), then pick highest P95 among those
  float p95_limit = MVS_THRESHOLD - SAFE_MARGIN;  // 0.85 for threshold=1.0
  
  size_t best_idx = 0;
  float best_p95 = -1.0f;
  bool found_safe = false;
  
  // Find safe bands (P95 < limit) and select the most "active" one
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
    // No safe bands - pick the one with lowest P95
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
  std::memcpy(selected_band_, candidates[best_idx].data(), SELECTED_SUBCARRIERS_COUNT);
  selected_band_size_ = SELECTED_SUBCARRIERS_COUNT;
  mv_values_ = std::move(results[best_idx].mv_values);
  
  ESP_LOGI(TAG, "P95: Band selection successful");
  ESP_LOGD(TAG, "  Selected: [%d-%d]",
           selected_band_[0], selected_band_[SELECTED_SUBCARRIERS_COUNT - 1]);
  ESP_LOGD(TAG, "  P95 MV: %.4f", best_p95);
  ESP_LOGD(TAG, "  Est. FP rate: %.1f%%", results[best_idx].fp_estimate * 100.0f);
  
  return ESP_OK;
}

void P95Calibrator::get_candidate_bands_(std::vector<std::vector<uint8_t>>& candidates) {
  candidates.clear();
  
  // HT20: Generate valid bands of 12 consecutive subcarriers
  // Zone before DC (subcarriers 11-31)
  for (uint16_t start = HT20_GUARD_BAND_LOW; start + SELECTED_SUBCARRIERS_COUNT <= HT20_DC_SUBCARRIER; start++) {
    std::vector<uint8_t> band;
    bool valid = true;
    
    for (uint8_t i = 0; i < SELECTED_SUBCARRIERS_COUNT; i++) {
      uint8_t sc = start + i;
      // Check if in valid range and not DC
      if (sc < HT20_GUARD_BAND_LOW || sc > HT20_GUARD_BAND_HIGH || sc == HT20_DC_SUBCARRIER) {
        valid = false;
        break;
      }
      band.push_back(sc);
    }
    
    if (valid && band.size() == SELECTED_SUBCARRIERS_COUNT) {
      candidates.push_back(band);
    }
  }
  
  // Zone after DC (subcarriers 33-52)
  for (uint16_t start = HT20_DC_SUBCARRIER + 1; start + SELECTED_SUBCARRIERS_COUNT <= HT20_GUARD_BAND_HIGH + 1; start++) {
    std::vector<uint8_t> band;
    bool valid = true;
    
    for (uint8_t i = 0; i < SELECTED_SUBCARRIERS_COUNT; i++) {
      uint8_t sc = start + i;
      if (sc < HT20_GUARD_BAND_LOW || sc > HT20_GUARD_BAND_HIGH || sc == HT20_DC_SUBCARRIER) {
        valid = false;
        break;
      }
      band.push_back(sc);
    }
    
    if (valid && band.size() == SELECTED_SUBCARRIERS_COUNT) {
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
  float valid_mags[SELECTED_SUBCARRIERS_COUNT];
  
  for (uint16_t pkt = 0; pkt < num_packets; pkt++) {
    const uint8_t* packet_data = &all_data[pkt * HT20_NUM_SUBCARRIERS];
    
    for (uint8_t i = 0; i < band_size && i < SELECTED_SUBCARRIERS_COUNT; i++) {
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
  result.p95 = calculate_percentile_(result.mv_values, BAND_SELECTION_PERCENTILE);
  
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

float P95Calibrator::calculate_percentile_(const std::vector<float>& values, uint8_t percentile) const {
  if (values.empty()) {
    return std::numeric_limits<float>::infinity();
  }
  
  std::vector<float> sorted = values;
  std::sort(sorted.begin(), sorted.end());
  
  size_t n = sorted.size();
  float p = percentile / 100.0f;  // Convert to fraction (e.g., 95 -> 0.95)
  float k = (n - 1) * p;
  size_t idx = static_cast<size_t>(k);
  
  if (idx >= n - 1) {
    return sorted.back();
  }
  
  // Linear interpolation
  float frac = k - idx;
  return sorted[idx] * (1.0f - frac) + sorted[idx + 1] * frac;
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

// ============================================================================
// FILE I/O METHODS
// ============================================================================

bool P95Calibrator::ensure_spiffs_mounted_() {
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

bool P95Calibrator::open_buffer_file_for_writing_() {
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

bool P95Calibrator::open_buffer_file_for_reading_() {
  buffer_file_ = fopen(buffer_path_, "rb");
  if (!buffer_file_) {
    ESP_LOGE(TAG, "Failed to open %s for reading", buffer_path_);
    return false;
  }
  return true;
}

void P95Calibrator::close_buffer_file_() {
  if (buffer_file_) {
    fclose(buffer_file_);
    buffer_file_ = nullptr;
  }
}

void P95Calibrator::remove_buffer_file_() {
  // truncate the file
  FILE* f = fopen(buffer_path_, "wb");
  if (f) {
    fclose(f);
  }
}

std::vector<uint8_t> P95Calibrator::read_window_(uint16_t start_idx, uint16_t window_size) {
  std::vector<uint8_t> data;
  
  if (!buffer_file_) {
    ESP_LOGE(TAG, "Buffer file not open for reading");
    return data;
  }
  
  size_t bytes_to_read = window_size * HT20_NUM_SUBCARRIERS;
  data.resize(bytes_to_read);
  
  // Seek to window start
  long offset = static_cast<long>(start_idx) * HT20_NUM_SUBCARRIERS;
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

}  // namespace espectre
}  // namespace esphome
