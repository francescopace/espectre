/*
 * ESPectre - Base Calibrator Implementation
 * 
 * Common calibration lifecycle for all calibrators.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "base_calibrator.h"
#include "csi_manager.h"
#include "esphome/core/log.h"
#include <cstring>

namespace esphome {
namespace espectre {

// ============================================================================
// PUBLIC API
// ============================================================================

void BaseCalibrator::init(CSIManager* csi_manager, const char* buffer_path) {
  csi_manager_ = csi_manager;
  file_buffer_.init(buffer_path);
}

esp_err_t BaseCalibrator::start_calibration(const uint8_t* current_band,
                                            uint8_t current_band_size,
                                            result_callback_t callback) {
  const char* tag = get_tag_();
  
  if (!csi_manager_) {
    ESP_LOGE(tag, "CSI Manager not initialized");
    return ESP_ERR_INVALID_STATE;
  }
  
  if (calibrating_) {
    ESP_LOGW(tag, "Calibration already in progress");
    return ESP_ERR_INVALID_STATE;
  }
  
  // Store context
  result_callback_ = callback;
  current_band_.assign(current_band, current_band + current_band_size);
  
  // Prepare file buffer
  file_buffer_.remove_file();
  if (!file_buffer_.open_for_writing()) {
    ESP_LOGE(tag, "Failed to open buffer file for writing");
    return ESP_ERR_NO_MEM;
  }
  
  file_buffer_.reset();
  mv_values_.clear();
  
  calibrating_ = true;
  csi_manager_->set_calibration_mode(this);
  
  ESP_LOGI(tag, "Calibration starting");
  
  return ESP_OK;
}

bool BaseCalibrator::add_packet(const int8_t* csi_data, size_t csi_len) {
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

void BaseCalibrator::on_collection_complete_() {
  const char* tag = get_tag_();
  ESP_LOGD(tag, "Collection complete, processing...");
  
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
      get_task_name_(),
      8192,  // 8KB stack for calibration calculations
      this,
      1,     // Low priority
      &calibration_task_handle_
  );
  
  if (result != pdPASS) {
    ESP_LOGE(tag, "Failed to create calibration task");
    finish_calibration_(false);
  }
}

void BaseCalibrator::calibration_task_wrapper_(void* arg) {
  BaseCalibrator* self = static_cast<BaseCalibrator*>(arg);
  const char* tag = self->get_tag_();
  
  // Open buffer file for reading
  if (!self->file_buffer_.open_for_reading()) {
    ESP_LOGE(tag, "Failed to open buffer file for reading");
    self->finish_calibration_(false);
    vTaskDelete(NULL);
    return;
  }
  
  // Run the subclass-specific calibration algorithm
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

void BaseCalibrator::finish_calibration_(bool success) {
  calibrating_ = false;
  calibration_task_handle_ = nullptr;
  
  if (result_callback_) {
    result_callback_(selected_band_, selected_band_size_, mv_values_, success);
  }
}

}  // namespace espectre
}  // namespace esphome
