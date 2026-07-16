/*
 * ESPectre - Classic Detector Implementation
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "classic_detector.h"

#include "espectre_log.h"
#include "threshold.h"

#include <algorithm>
#include <cmath>

namespace espectre {

static const char* TAG = "ClassicDetector";

ClassicDetector::ClassicDetector(uint16_t window_size, float threshold)
    : BaseDetector(window_size),
      threshold_(clamp_threshold(threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD)),
      current_probability_(0.0f),
      current_logit_(0.0f),
      current_l1_delta_(0.0f),
      current_turb_autocorr_(0.0f),
      startup_logit_count_(0U),
      adapted_threshold_(CLASSIC_DEFAULT_THRESHOLD),
      adapted_threshold_ready_(false) {
  l1_tracker_.configure(l1_delta_capacity_());
  ESP_LOGI(TAG, "Initialized weighted fusion (window=%u, threshold=%.3f)",
           static_cast<unsigned>(window_size_), threshold_);
}

uint16_t ClassicDetector::l1_delta_capacity_() const {
  return window_size_ > L1_DELTA_LAG
             ? static_cast<uint16_t>(window_size_ - L1_DELTA_LAG)
             : 0U;
}

void ClassicDetector::configure_hampel(bool enabled, uint8_t window_size,
                                       float threshold) {
  BaseDetector::configure_hampel(enabled, window_size, threshold);
  l1_tracker_.configure_hampel(enabled, window_size, threshold);
}

void ClassicDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* selected_subcarriers,
                                     uint8_t num_subcarriers) {
  if (csi_data == nullptr) {
    ESP_LOGE(TAG, "process_packet: NULL CSI data");
    return;
  }

  float amplitudes[HT20_SELECTED_BAND_SIZE];
  const uint8_t amplitude_count = extract_subcarrier_amplitudes(
      csi_data, csi_len, selected_subcarriers, num_subcarriers,
      amplitudes, HT20_SELECTED_BAND_SIZE);
  process_amplitudes(amplitudes, amplitude_count);
  l1_tracker_.process(amplitudes, amplitude_count);
}

bool ClassicDetector::is_ready() const {
  return buffer_count_ >= window_size_ &&
         l1_tracker_.count() >= l1_delta_capacity_();
}

float ClassicDetector::calculate_turb_autocorr_() const {
  if (buffer_count_ < 3U || turbulence_buffer_ == nullptr) {
    return 0.0f;
  }

  float ordered[DETECTOR_MAX_WINDOW_SIZE];
  if (buffer_count_ < window_size_) {
    std::copy(turbulence_buffer_, turbulence_buffer_ + buffer_count_, ordered);
  } else {
    for (uint16_t i = 0U; i < buffer_count_; i++) {
      ordered[i] = turbulence_buffer_[(buffer_index_ + i) % window_size_];
    }
  }

  float mean = 0.0f;
  for (uint16_t i = 0U; i < buffer_count_; i++) {
    mean += ordered[i];
  }
  mean /= buffer_count_;

  float variance = 0.0f;
  for (uint16_t i = 0U; i < buffer_count_; i++) {
    const float diff = ordered[i] - mean;
    variance += diff * diff;
  }
  variance /= buffer_count_;
  return calc_autocorrelation(ordered, buffer_count_, mean, variance, 1U);
}

float ClassicDetector::calculate_logit_(float l1_delta, float turb_autocorr) const {
  const float normalized_l1 = (l1_delta - CLASSIC_L1_CENTER) / CLASSIC_L1_SCALE;
  const float normalized_autocorr =
      (turb_autocorr - CLASSIC_AUTOCORR_CENTER) / CLASSIC_AUTOCORR_SCALE;
  return CLASSIC_INTERCEPT + CLASSIC_L1_WEIGHT * normalized_l1 +
         CLASSIC_AUTOCORR_WEIGHT * normalized_autocorr;
}

float ClassicDetector::sigmoid_(float value) {
  if (value < -20.0f) return 0.0f;
  if (value > 20.0f) return 1.0f;
  return 1.0f / (1.0f + std::exp(-value));
}

void ClassicDetector::update_state() {
  if (!is_ready()) {
    current_probability_ = 0.0f;
    state_ = MotionState::IDLE;
    return;
  }

  current_l1_delta_ = l1_tracker_.mean();
  current_turb_autocorr_ = calculate_turb_autocorr_();
  current_logit_ = calculate_logit_(current_l1_delta_, current_turb_autocorr_);
  current_probability_ = sigmoid_(current_logit_);
  if (!adapted_threshold_ready_ && startup_logit_count_ < CLASSIC_STARTUP_SAMPLE_LIMIT) {
    startup_logits_[startup_logit_count_++] = current_logit_;
  }
  state_ = current_probability_ > threshold_ ? MotionState::MOTION : MotionState::IDLE;
}

float ClassicDetector::startup_quantile_() const {
  if (startup_logit_count_ == 0U) {
    return CLASSIC_TRAIN_IDLE_Q95_LOGIT;
  }
  float ordered[CLASSIC_STARTUP_SAMPLE_LIMIT];
  std::copy(startup_logits_, startup_logits_ + startup_logit_count_, ordered);
  std::sort(ordered, ordered + startup_logit_count_);
  const float position = static_cast<float>(startup_logit_count_ - 1U) *
                         CLASSIC_STARTUP_QUANTILE;
  const uint8_t lower = static_cast<uint8_t>(position);
  const uint8_t upper = std::min<uint8_t>(lower + 1U, startup_logit_count_ - 1U);
  const float fraction = position - static_cast<float>(lower);
  return ordered[lower] * (1.0f - fraction) + ordered[upper] * fraction;
}

void ClassicDetector::on_startup_calibration_begin() {
  startup_logit_count_ = 0U;
  adapted_threshold_ready_ = false;
}

void ClassicDetector::on_startup_calibration_complete() {
  const float base_logit = std::log(CLASSIC_DEFAULT_THRESHOLD /
                                    (1.0f - CLASSIC_DEFAULT_THRESHOLD));
  const float adapted_logit = base_logit + CLASSIC_STARTUP_STRENGTH *
      (startup_quantile_() - CLASSIC_TRAIN_IDLE_Q95_LOGIT);
  adapted_threshold_ = sigmoid_(adapted_logit);
  adapted_threshold_ready_ = true;
  ESP_LOGD(TAG, "Startup threshold prepared: %.6f (%u logit samples)",
           adapted_threshold_, static_cast<unsigned>(startup_logit_count_));
}

bool ClassicDetector::set_adaptive_threshold(float) {
  threshold_ = adapted_threshold_ready_ ? adapted_threshold_ : CLASSIC_DEFAULT_THRESHOLD;
  return true;
}

bool ClassicDetector::set_threshold(float threshold) {
  if (!is_valid_threshold(threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD)) {
    ESP_LOGE(TAG, "Invalid threshold: %.3f (must be 0.0-1.0)", threshold);
    return false;
  }
  threshold_ = threshold;
  return true;
}

void ClassicDetector::reset() {
  BaseDetector::reset();
  current_probability_ = 0.0f;
  current_logit_ = 0.0f;
  current_l1_delta_ = 0.0f;
  current_turb_autocorr_ = 0.0f;
  state_ = MotionState::IDLE;
}

void ClassicDetector::clear_buffer() {
  BaseDetector::clear_buffer();
  l1_tracker_.clear();
  current_probability_ = 0.0f;
  current_logit_ = 0.0f;
  current_l1_delta_ = 0.0f;
  current_turb_autocorr_ = 0.0f;
}

}  // namespace espectre
