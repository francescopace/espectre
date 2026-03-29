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

namespace {
constexpr float NBVI_ACCEPTABLE_FP_RATE = 0.05f;

inline void packet_u8_to_float_magnitudes(const uint8_t* packet_data, float* out_magnitudes) {
  for (uint8_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
    out_magnitudes[sc] = static_cast<float>(packet_data[sc]);
  }
}
}  // namespace

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

  if (current_band == nullptr || current_band_size == 0) {
    ESP_LOGE(TAG, "Invalid current band input");
    return ESP_ERR_INVALID_ARG;
  }
  
  // Store context
  result_callback_ = callback;
  const uint8_t safe_band_size = std::min<uint8_t>(current_band_size, HT20_SELECTED_BAND_SIZE);
  current_band_.assign(current_band, current_band + safe_band_size);
  
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
    
    // Generate candidates
    std::vector<std::vector<uint8_t>> candidates_to_eval;
    uint8_t temp_band[HT20_SELECTED_BAND_SIZE];
    uint8_t temp_size;

    auto push_if_unique = [&](const uint8_t* band, uint8_t size) {
      if (size != HT20_SELECTED_BAND_SIZE) return;
      std::vector<uint8_t> b(band, band + size);
      if (std::find(candidates_to_eval.begin(), candidates_to_eval.end(), b) == candidates_to_eval.end()) {
        candidates_to_eval.push_back(b);
      }
    };

    // 1. Entropy Spaced
    for (auto& m : all_metrics) m.nbvi = m.nbvi_entropy;
    std::sort(all_metrics.begin(), all_metrics.begin() + filtered_count,
              [](const NBVIMetrics& a, const NBVIMetrics& b) { return a.nbvi < b.nbvi; });
    select_with_spacing_strict_(all_metrics, temp_band, &temp_size);
    push_if_unique(temp_band, temp_size);

    // 2. MAD Clustered
    for (auto& m : all_metrics) m.nbvi = m.nbvi_mad;
    std::sort(all_metrics.begin(), all_metrics.begin() + filtered_count,
              [](const NBVIMetrics& a, const NBVIMetrics& b) { return a.nbvi < b.nbvi; });
    select_with_spacing_(all_metrics, temp_band, &temp_size);
    push_if_unique(temp_band, temp_size);

    // 3. Classic Spaced
    for (auto& m : all_metrics) m.nbvi = m.nbvi_classic;
    std::sort(all_metrics.begin(), all_metrics.begin() + filtered_count,
              [](const NBVIMetrics& a, const NBVIMetrics& b) { return a.nbvi < b.nbvi; });
    select_with_spacing_strict_(all_metrics, temp_band, &temp_size);
    push_if_unique(temp_band, temp_size);

    // 4. Classic Clustered
    select_with_spacing_(all_metrics, temp_band, &temp_size);
    push_if_unique(temp_band, temp_size);
    
    for (const auto& candidate_band : candidates_to_eval) {
      float fp_rate = 0.0f;
      std::vector<float> temp_mv_values;
      if (!validate_subcarriers_(candidate_band.data(), candidate_band.size(), &fp_rate, temp_mv_values)) {
        continue;
      }
      
      bool override = false;
      if (!found_valid) {
        override = true;
      } else if (fp_rate <= NBVI_ACCEPTABLE_FP_RATE) {
        if (best_fp_rate > NBVI_ACCEPTABLE_FP_RATE) {
          override = true;
        }
      } else {
        if (fp_rate < best_fp_rate) {
          override = true;
        }
      }
      
      if (override) {
        best_fp_rate = fp_rate;
        std::memcpy(best_band, candidate_band.data(), HT20_SELECTED_BAND_SIZE);
        best_mv_values = std::move(temp_mv_values);
        found_valid = true;
      }
    }
    
    vTaskDelay(1);  // Yield
  }
  
  if (!found_valid) {
    ESP_LOGW(TAG, "All candidate windows failed - using default subcarriers");

    selected_band_size_ = static_cast<uint8_t>(std::min<size_t>(current_band_.size(), HT20_SELECTED_BAND_SIZE));
    if (selected_band_size_ == 0) {
      ESP_LOGE(TAG, "Fallback band is empty");
      return ESP_FAIL;
    }
    std::memcpy(selected_band_, current_band_.data(),
                selected_band_size_ * sizeof(selected_band_[0]));
    
    // Get MV values for default band
    float fp_rate = 0.0f;
    if (!validate_subcarriers_(selected_band_, selected_band_size_, &fp_rate, mv_values_)) {
      ESP_LOGE(TAG, "Fallback band validation failed");
      return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Fallback to default band");
    return ESP_OK;
  }
  
  // Prefer hinted/current band only when candidate is NOT acceptable (>target FP)
  // AND the hint band has better or comparable FP rate.
  constexpr float FP_COMPARE_EPSILON = 1e-6f;
  bool use_hint_band = false;
  float hint_fp_rate = 1.0f;
  std::vector<float> hint_mv_values;
  if (current_band_.size() == HT20_SELECTED_BAND_SIZE) {
    if (validate_subcarriers_(current_band_.data(),
                              static_cast<uint8_t>(current_band_.size()),
                              &hint_fp_rate,
                              hint_mv_values)) {
      if (best_fp_rate > NBVI_ACCEPTABLE_FP_RATE) {
        const float hint_cmp = hint_fp_rate + FP_COMPARE_EPSILON;
        const float best_cmp = best_fp_rate + hint_fp_tolerance_;
        const bool hint_fp_ok = prefer_hint_on_tie_
                                    ? (hint_cmp <= best_cmp)
                                    : (hint_cmp < best_cmp);
        if (hint_fp_ok) {
          use_hint_band = true;
        }
      } else {
        ESP_LOGD(TAG, "Keeping candidate band with FP %.1f%% (target <%.1f%%)",
                 best_fp_rate * 100.0f, NBVI_ACCEPTABLE_FP_RATE * 100.0f);
      }
    }
  }

  // Store results
  if (use_hint_band) {
    std::memcpy(selected_band_, current_band_.data(), HT20_SELECTED_BAND_SIZE);
    selected_band_size_ = HT20_SELECTED_BAND_SIZE;
    mv_values_ = std::move(hint_mv_values);
    ESP_LOGI(TAG, "Using hinted/current band (FP %.1f%% vs best %.1f%%, tol %.1f%%, tie=%s)",
             hint_fp_rate * 100.0f, best_fp_rate * 100.0f, hint_fp_tolerance_ * 100.0f,
             prefer_hint_on_tie_ ? "prefer" : "strict");
  } else {
    std::memcpy(selected_band_, best_band, HT20_SELECTED_BAND_SIZE);
    selected_band_size_ = HT20_SELECTED_BAND_SIZE;
    mv_values_ = std::move(best_mv_values);
  }
  
  ESP_LOGI(TAG, "NBVI Calibration successful");
  ESP_LOGI(TAG, "  Band: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
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
  all_windows.reserve(((buffer_count - window_size_) / window_step_) + 1);
  const size_t expected_window_bytes = static_cast<size_t>(window_size_) * HT20_NUM_SUBCARRIERS;
  std::vector<float> turbulences(window_size_);
  
  for (uint16_t start = 0; start + window_size_ <= buffer_count; start += window_step_) {
    std::vector<uint8_t> window_data = file_buffer_.read_window(start, window_size_);
    if (window_data.size() != expected_window_bytes) {
      continue;
    }

    // Calculate turbulence for each packet using current band.
    for (uint16_t pkt = 0; pkt < window_size_; pkt++) {
      const uint8_t* packet_magnitudes = &window_data[pkt * HT20_NUM_SUBCARRIERS];
      
      float float_mags[HT20_NUM_SUBCARRIERS];
      packet_u8_to_float_magnitudes(packet_magnitudes, float_mags);
      
      turbulences[pkt] = calculate_spatial_turbulence(float_mags, current_band_.data(),
                                                       current_band_.size(), HT20_NUM_SUBCARRIERS,
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
  variances.reserve(all_windows.size());
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
  
  std::vector<float> magnitudes(window_size_);
  for (uint8_t sc = 0; sc < HT20_NUM_SUBCARRIERS; sc++) {
    for (uint16_t pkt = 0; pkt < window_size_; pkt++) {
      magnitudes[pkt] = static_cast<float>(window_data[pkt * HT20_NUM_SUBCARRIERS + sc]);
    }
    
    metrics[sc].subcarrier = sc;
    calculate_nbvi_weighted_(magnitudes, metrics[sc]);
    
    // Exclude guard bands and DC
    if (sc < HT20_GUARD_BAND_LOW || sc > HT20_GUARD_BAND_HIGH || sc == HT20_DC_SUBCARRIER) {
      metrics[sc].nbvi = metrics[sc].nbvi_classic = metrics[sc].nbvi_entropy =
        metrics[sc].nbvi_mad = std::numeric_limits<float>::infinity();
    } else if (metrics[sc].mean < NULL_SUBCARRIER_THRESHOLD) {
      metrics[sc].nbvi = metrics[sc].nbvi_classic = metrics[sc].nbvi_entropy =
        metrics[sc].nbvi_mad = std::numeric_limits<float>::infinity();
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

void NBVICalibrator::select_with_spacing_strict_(const std::vector<NBVIMetrics>& sorted_metrics,
                                                 uint8_t* output_band, uint8_t* output_size) {
  std::vector<uint8_t> valid_candidates;
  for (const auto& m : sorted_metrics) {
    if (!std::isinf(m.nbvi)) {
      valid_candidates.push_back(m.subcarrier);
    }
  }
  
  std::vector<uint8_t> selected;
  for (int current_spacing = min_spacing_; current_spacing >= 0; --current_spacing) {
    selected.clear();
    for (uint8_t sc : valid_candidates) {
      if (selected.size() >= HT20_SELECTED_BAND_SIZE) break;
      
      bool too_close = false;
      for (uint8_t existing : selected) {
        if (std::abs(sc - existing) < current_spacing) {
          too_close = true;
          break;
        }
      }
      if (!too_close) {
        selected.push_back(sc);
      }
    }
    if (selected.size() >= HT20_SELECTED_BAND_SIZE) {
      break;
    }
  }
  
  if (selected.size() < HT20_SELECTED_BAND_SIZE) {
    selected.clear();
    for (size_t i = 0; i < valid_candidates.size() && selected.size() < HT20_SELECTED_BAND_SIZE; i++) {
      selected.push_back(valid_candidates[i]);
    }
  }
  
  std::sort(selected.begin(), selected.end());
  
  *output_size = selected.size();
  std::memcpy(output_band, selected.data(), selected.size());
}

void NBVICalibrator::select_with_spacing_(const std::vector<NBVIMetrics>& sorted_metrics,
                                          uint8_t* output_band,
                                          uint8_t* output_size) {
  std::vector<uint8_t> selected;
  
  // Always include top 5 (best NBVI) without spacing constraint
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
  if (band == nullptr || band_size == 0 || out_fp_rate == nullptr || mvs_window_size_ == 0) {
    return false;
  }

  uint16_t buffer_count = file_buffer_.get_count();
  if (buffer_count == 0 || buffer_count < mvs_window_size_) {
    *out_fp_rate = 0.0f;
    return true;
  }

  // IMPORTANT (memory safety on constrained targets, e.g. ESP32-C5):
  // Do NOT read the full calibration buffer in one allocation.
  // A previous refactor used read_window(0, buffer_count), which can allocate
  // ~50 KB for 750 packets and trigger std::bad_alloc/abort during NBVI.
  //
  // Keep validation bounded by reading fixed-size chunks from SPIFFS.
  // This is intentionally more conservative than a single bulk read to avoid
  // runtime panics during calibration.
  constexpr uint16_t VALIDATION_CHUNK_PACKETS = 64;

  // Keep calibration validation aligned with runtime detector filter chain.
  lowpass_filter_state_t lowpass_state{};
  hampel_turbulence_state_t hampel_state{};
  if (lowpass_enabled_) {
    lowpass_filter_init(&lowpass_state, lowpass_cutoff_hz_, LOWPASS_SAMPLE_RATE, true);
  }
  if (hampel_enabled_) {
    hampel_turbulence_init(&hampel_state, hampel_window_, hampel_threshold_, true);
  }

  std::vector<float> turbulence_ring(mvs_window_size_, 0.0f);
  out_mv_values.reserve(buffer_count - mvs_window_size_ + 1);

  uint16_t ring_idx = 0;
  uint16_t ring_count = 0;
  float running_sum = 0.0f;
  float running_sum_sq = 0.0f;
  for (uint16_t start_pkt = 0; start_pkt < buffer_count; start_pkt += VALIDATION_CHUNK_PACKETS) {
    const uint16_t chunk_packets = std::min<uint16_t>(VALIDATION_CHUNK_PACKETS, buffer_count - start_pkt);
    std::vector<uint8_t> chunk_data = file_buffer_.read_window(start_pkt, chunk_packets);
    const size_t expected_chunk_bytes = static_cast<size_t>(chunk_packets) * HT20_NUM_SUBCARRIERS;
    if (chunk_data.size() != expected_chunk_bytes) {
      ESP_LOGW(TAG, "Validation read mismatch at pkt %u: got %zu, expected %zu",
               start_pkt, chunk_data.size(), expected_chunk_bytes);
      return false;
    }

    for (uint16_t pkt = 0; pkt < chunk_packets; pkt++) {
      const uint8_t* packet_data = &chunk_data[pkt * HT20_NUM_SUBCARRIERS];
      float float_mags[HT20_NUM_SUBCARRIERS];
      packet_u8_to_float_magnitudes(packet_data, float_mags);
      
      float turbulence = calculate_spatial_turbulence(float_mags, band, band_size,
                                                       HT20_NUM_SUBCARRIERS,
                                                       use_cv_normalization_);
      float filtered_turbulence = turbulence;
      if (hampel_enabled_) {
        filtered_turbulence = hampel_filter_turbulence(&hampel_state, filtered_turbulence);
      }
      if (lowpass_enabled_) {
        filtered_turbulence = lowpass_filter_apply(&lowpass_state, filtered_turbulence);
      }

      if (ring_count < mvs_window_size_) {
        turbulence_ring[ring_idx] = filtered_turbulence;
        running_sum += filtered_turbulence;
        running_sum_sq += filtered_turbulence * filtered_turbulence;
        ring_count++;
        ring_idx = (ring_idx + 1) % mvs_window_size_;
      } else {
        const float old = turbulence_ring[ring_idx];
        running_sum -= old;
        running_sum_sq -= old * old;
        turbulence_ring[ring_idx] = filtered_turbulence;
        running_sum += filtered_turbulence;
        running_sum_sq += filtered_turbulence * filtered_turbulence;
        ring_idx = (ring_idx + 1) % mvs_window_size_;
      }

      if (ring_count < mvs_window_size_) {
        continue;
      }

      float mean = running_sum / mvs_window_size_;
      float variance = (running_sum_sq / mvs_window_size_) - (mean * mean);
      if (variance < 0.0f) {
        variance = 0.0f;
      }
      out_mv_values.push_back(variance);
    }
  }

  if (out_mv_values.empty()) {
    *out_fp_rate = 0.0f;
    return true;
  }

  float adaptive_threshold = 0.0f;
  uint8_t threshold_percentile = 0;
  calculate_adaptive_threshold(out_mv_values, ThresholdMode::AUTO,
                               adaptive_threshold, threshold_percentile);

  uint32_t motion_count = 0;
  for (float variance : out_mv_values) {
    if (variance > adaptive_threshold) {
      motion_count++;
    }
  }

  *out_fp_rate = static_cast<float>(motion_count) /
                 static_cast<float>(out_mv_values.size());
  
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
    out_metrics.nbvi_classic = std::numeric_limits<float>::infinity();
    out_metrics.nbvi_entropy = std::numeric_limits<float>::infinity();
    out_metrics.nbvi_mad = std::numeric_limits<float>::infinity();
    out_metrics.mean = 0.0f;
    out_metrics.std = 0.0f;
    out_metrics.mad = 0.0f;
    out_metrics.entropy = 0.0f;
    return;
  }
  
  float sum = 0.0f;
  for (float mag : magnitudes) {
    sum += mag;
  }
  float mean = sum / count;
  
  if (mean < 1e-6f) {
    out_metrics.nbvi = std::numeric_limits<float>::infinity();
    out_metrics.nbvi_classic = std::numeric_limits<float>::infinity();
    out_metrics.nbvi_entropy = std::numeric_limits<float>::infinity();
    out_metrics.nbvi_mad = std::numeric_limits<float>::infinity();
    out_metrics.mean = mean;
    out_metrics.std = 0.0f;
    out_metrics.mad = 0.0f;
    out_metrics.entropy = 0.0f;
    return;
  }
  
  float variance = calculate_variance_two_pass(magnitudes.data(), count);
  float stddev = std::sqrt(variance);
  
  // Entropy
  float min_v = magnitudes[0];
  float max_v = magnitudes[0];
  for (float mag : magnitudes) {
    if (mag < min_v) min_v = mag;
    if (mag > max_v) max_v = mag;
  }
  float range_v = max_v - min_v;
  float entropy = 0.0f;
  if (range_v > 0) {
    int bins[10] = {0};
    float bin_w = range_v / 10.0f;
    for (float v : magnitudes) {
      int b = static_cast<int>((v - min_v) / bin_w);
      if (b == 10) b = 9;
      bins[b]++;
    }
    for (int b : bins) {
      if (b > 0) {
        float p = static_cast<float>(b) / count;
        entropy -= p * std::log2(p);
      }
    }
  }
  
  // MAD
  std::vector<float> sorted_vals = magnitudes;
  std::sort(sorted_vals.begin(), sorted_vals.end());
  float median = sorted_vals[count / 2];
  std::vector<float> abs_devs(count);
  for (size_t i = 0; i < count; i++) {
    abs_devs[i] = std::abs(magnitudes[i] - median);
  }
  std::sort(abs_devs.begin(), abs_devs.end());
  float mad = abs_devs[count / 2];

  // Classic score
  float cv = stddev / mean;
  float nbvi_energy = stddev / (mean * mean);
  float base_score = alpha_ * nbvi_energy + (1.0f - alpha_) * cv;

  // Entropy score
  float entropy_factor = std::max(0.5f, entropy);
  float entropy_score = base_score / entropy_factor;

  // MAD score
  float robust_std = (mad > 1e-6f) ? (mad * 1.4826f) : stddev;
  float cv_mad = robust_std / mean;
  float energy_mad = robust_std / (mean * mean);
  float mad_score = alpha_ * energy_mad + (1.0f - alpha_) * cv_mad;

  out_metrics.nbvi_classic = base_score;
  out_metrics.nbvi_entropy = entropy_score;
  out_metrics.nbvi_mad = mad_score;

  out_metrics.nbvi = out_metrics.nbvi_classic;  // default
  out_metrics.mean = mean;
  out_metrics.std = stddev;
  out_metrics.mad = mad;
  out_metrics.entropy = entropy;
}

}  // namespace espectre
}  // namespace esphome
