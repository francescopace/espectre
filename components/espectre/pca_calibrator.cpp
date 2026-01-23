/*
 * ESPectre - PCA Calibrator Implementation
 * 
 * Collects baseline correlation values for PCA-based motion detection.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "pca_calibrator.h"
#include "csi_manager.h"
#include "esphome/core/log.h"
#include <cmath>
#include <cstring>
#include <algorithm>

namespace esphome {
namespace espectre {

static const char *TAG = "PCACalibrator";

// ============================================================================
// Constructor / Destructor
// ============================================================================

PCACalibrator::PCACalibrator() {
  // Allocate CSI buffer
  for (size_t i = 0; i < PCA_WINDOW_SIZE; i++) {
    csi_buffer_[i] = new float[PCA_NUM_SUBCARRIERS]();
  }
  // Allocate PCA buffer
  for (size_t i = 0; i < PCA_BUFFER_SIZE; i++) {
    pca_buffer_[i] = new float[PCA_NUM_SUBCARRIERS]();
  }
  
  // Initialize selected_band_ with PCA subcarriers (every PCA_SUBCARRIER_STEP-th)
  for (uint8_t i = 0; i < PCA_NUM_SUBCARRIERS; i++) {
    selected_band_[i] = i * PCA_SUBCARRIER_STEP;
  }
  selected_band_size_ = PCA_NUM_SUBCARRIERS;
}

PCACalibrator::~PCACalibrator() {
  for (size_t i = 0; i < PCA_WINDOW_SIZE; i++) {
    delete[] csi_buffer_[i];
  }
  for (size_t i = 0; i < PCA_BUFFER_SIZE; i++) {
    delete[] pca_buffer_[i];
  }
}

// ============================================================================
// ICalibrator Interface
// ============================================================================

void PCACalibrator::init(CSIManager* csi_manager) {
  csi_manager_ = csi_manager;
  ESP_LOGD(TAG, "PCA Calibrator initialized");
}

esp_err_t PCACalibrator::start_calibration(const uint8_t* /* current_band */,
                                           uint8_t /* current_band_size */,
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
  
  // Reset state
  packet_count_ = 0;
  csi_buffer_count_ = 0;
  pca_buffer_count_ = 0;
  pca_buffer_idx_ = 0;
  correlation_values_.clear();
  correlation_values_.reserve(buffer_size_);
  
  calibrating_ = true;
  csi_manager_->set_calibration_mode(this);
  
  ESP_LOGI(TAG, "PCA Calibration Starting (collecting %d packets)", buffer_size_);
  
  return ESP_OK;
}

bool PCACalibrator::add_packet(const int8_t* csi_data, size_t csi_len) {
  if (!calibrating_) {
    return false;
  }
  
  // Extract amplitudes (every PCA_SUBCARRIER_STEP-th subcarrier)
  size_t buffer_idx = csi_buffer_count_ % PCA_WINDOW_SIZE;
  extract_amplitudes(csi_data, csi_len, csi_buffer_[buffer_idx]);
  csi_buffer_count_++;
  packet_count_++;
  
  // Need enough data for PCA
  if (csi_buffer_count_ < PCA_WINDOW_SIZE) {
    // Log progress every 100 packets
    if (packet_count_ % 100 == 0) {
      ESP_LOGD(TAG, "Collection progress: %d/%d", packet_count_, buffer_size_);
    }
    return packet_count_ >= buffer_size_;
  }
  
  // Compute PCA on current window
  float pca_current[PCA_NUM_SUBCARRIERS];
  if (!compute_pca(pca_current)) {
    return packet_count_ >= buffer_size_;
  }
  
  // Calculate correlation with past PCA vectors
  if (pca_buffer_count_ > 0) {
    float correlation = compute_correlation(pca_current);
    if (correlation > 0.0f) {
      correlation_values_.push_back(correlation);
    }
  }
  
  // Store PCA vector for future comparisons
  std::memcpy(pca_buffer_[pca_buffer_idx_], pca_current, PCA_NUM_SUBCARRIERS * sizeof(float));
  pca_buffer_idx_ = (pca_buffer_idx_ + 1) % PCA_BUFFER_SIZE;
  pca_buffer_count_++;
  
  // Log progress every 100 packets
  if (packet_count_ % 100 == 0) {
    ESP_LOGD(TAG, "Collection progress: %d/%d (correlations: %zu)", 
             packet_count_, buffer_size_, correlation_values_.size());
  }
  
  // Check if collection is complete
  if (packet_count_ >= buffer_size_) {
    on_collection_complete_();
    return true;
  }
  
  return false;
}

// ============================================================================
// Internal Methods
// ============================================================================

void PCACalibrator::extract_amplitudes(const int8_t* csi_data, size_t len, float* amplitudes) {
  // CSI data is I/Q pairs, so len/2 subcarriers
  size_t num_sc = len / 2;
  
  for (size_t i = 0; i < PCA_NUM_SUBCARRIERS; i++) {
    size_t sc = i * PCA_SUBCARRIER_STEP;
    if (sc < num_sc) {
      int8_t real_part = csi_data[sc * 2];
      int8_t imag_part = csi_data[sc * 2 + 1];
      amplitudes[i] = std::sqrt(static_cast<float>(real_part * real_part + imag_part * imag_part));
    } else {
      amplitudes[i] = 0.0f;
    }
  }
}

bool PCACalibrator::compute_pca(float* pca_output) {
  // Build pointer array for pca_power_method
  float* window_ptrs[PCA_WINDOW_SIZE];
  for (size_t i = 0; i < PCA_WINDOW_SIZE; i++) {
    size_t idx = (csi_buffer_count_ - PCA_WINDOW_SIZE + i) % PCA_WINDOW_SIZE;
    window_ptrs[i] = csi_buffer_[idx];
  }
  
  return pca_power_method(window_ptrs, PCA_WINDOW_SIZE, PCA_NUM_SUBCARRIERS, pca_output);
}

float PCACalibrator::compute_correlation(const float* pca_current) {
  // Calculate correlation with recent PCA vectors
  float max_corr = 0.0f;
  
  size_t num_past = std::min(pca_buffer_count_, (size_t)PCA_BUFFER_SIZE);
  
  for (size_t i = 0; i < std::min(num_past, (size_t)5); i++) {
    // Get past PCA vector
    size_t past_idx = (pca_buffer_idx_ + PCA_BUFFER_SIZE - 1 - i) % PCA_BUFFER_SIZE;
    
    float corr = std::fabs(pearson_correlation(pca_current, pca_buffer_[past_idx], PCA_NUM_SUBCARRIERS));
    if (corr > max_corr) {
      max_corr = corr;
    }
  }
  
  return max_corr;
}

void PCACalibrator::on_collection_complete_() {
  ESP_LOGD(TAG, "PCA: Collection complete, %zu correlation values", correlation_values_.size());
  
  // Notify caller that collection is complete (can pause traffic generator)
  if (collection_complete_callback_) {
    collection_complete_callback_();
  }
  
  // Stop receiving CSI packets
  csi_manager_->set_calibration_mode(nullptr);
  
  // Finish calibration (no separate task needed - minimal processing)
  bool success = !correlation_values_.empty();
  finish_calibration_(success);
}

void PCACalibrator::finish_calibration_(bool success) {
  calibrating_ = false;
  
  if (result_callback_) {
    // Return PCA subcarriers and correlation values
    result_callback_(selected_band_, selected_band_size_, correlation_values_, success);
  }
}

}  // namespace espectre
}  // namespace esphome
