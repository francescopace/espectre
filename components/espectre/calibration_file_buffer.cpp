/*
 * ESPectre - Calibration File Buffer Implementation
 * 
 * File-based buffer for calibration data storage on SPIFFS.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "calibration_file_buffer.h"
#include "utils.h"
#include "esphome/core/log.h"
#include "esp_spiffs.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include <cstring>
#include <cmath>
#include <algorithm>

namespace esphome {
namespace espectre {

static const char *TAG = "CalFileBuffer";

// ============================================================================
// PUBLIC API
// ============================================================================

void CalibrationFileBuffer::init(const char* buffer_path, uint16_t buffer_size) {
  buffer_path_ = buffer_path;
  buffer_size_ = buffer_size;
  ESP_LOGD(TAG, "Initialized (path: %s, size: %d)", buffer_path_, buffer_size_);
}

void CalibrationFileBuffer::reset() {
  buffer_count_ = 0;
  last_progress_ = 0;
}

bool CalibrationFileBuffer::open_for_writing() {
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

bool CalibrationFileBuffer::open_for_reading() {
  buffer_file_ = fopen(buffer_path_, "rb");
  if (!buffer_file_) {
    ESP_LOGE(TAG, "Failed to open %s for reading", buffer_path_);
    return false;
  }
  return true;
}

void CalibrationFileBuffer::close() {
  if (buffer_file_) {
    fclose(buffer_file_);
    buffer_file_ = nullptr;
  }
}

void CalibrationFileBuffer::remove_file() {
  if (remove(buffer_path_) == 0) {
    return;  // File removed successfully
  }
  // Fallback: truncate file (SPIFFS may not support remove)
  ESP_LOGD(TAG, "remove() failed, truncating file instead");
  FILE* f = fopen(buffer_path_, "wb");
  if (f) {
    fclose(f);
  }
}

// ============================================================================
// PACKET WRITE
// ============================================================================

bool CalibrationFileBuffer::write_packet(const int8_t* csi_data, size_t csi_len) {
  if (is_full() || !buffer_file_) {
    return is_full();
  }
  
  // Validate HT20 format (64 subcarriers)
  uint16_t packet_sc = csi_len / 2;
  if (packet_sc != HT20_NUM_SUBCARRIERS) {
    return false;
  }
  
  // Calculate magnitudes and write as uint8 (max CSI magnitude ~181 fits in 1 byte)
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
  
  // Flush periodically and yield to prevent WiFi starvation
  // This helps prevent ENOMEM errors on ESP32-S3 where PSRAM, SPIFFS,
  // and WiFi compete for bus access
  if (buffer_count_ % 100 == 0) {
    fflush(buffer_file_);
    vTaskDelay(1);  // Minimal yield
  }
  
  // Log progress every 10%
  uint8_t progress = (buffer_count_ * 100) / buffer_size_;
  if (progress >= last_progress_ + 10 || buffer_count_ == buffer_size_) {
    log_progress_bar(TAG, progress / 100.0f, 20, -1,
                     "%d%% (%d/%d)", progress, buffer_count_, buffer_size_);
    last_progress_ = progress;
  }
  
  return is_full();
}

// ============================================================================
// PACKET READ
// ============================================================================

std::vector<uint8_t> CalibrationFileBuffer::read_window(uint16_t start_idx, uint16_t window_size) {
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

// ============================================================================
// SPIFFS MANAGEMENT
// ============================================================================

bool CalibrationFileBuffer::ensure_spiffs_mounted_() {
  // Check if already mounted by trying to open a file
  FILE* test = fopen(buffer_path_, "rb");
  if (test) {
    fclose(test);
    return true;
  }
  
  // Try to mount SPIFFS
  esp_vfs_spiffs_conf_t conf = {
    .base_path = "/spiffs",
    .partition_label = NULL,
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

}  // namespace espectre
}  // namespace esphome
