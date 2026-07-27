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

ClassicDetector::ClassicDetector(uint16_t window_size, float threshold,
                                 uint16_t lag, uint16_t autocorr_lag)
    : BaseDetector(window_size),
      threshold_(clamp_threshold(threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD)),
      current_probability_(0.0f),
      current_logit_(0.0f),
      current_l1_delta_(0.0f),
      current_turb_autocorr_(0.0f),
      startup_logit_count_(0U),
      adapted_threshold_(CLASSIC_DEFAULT_THRESHOLD),
      adapted_threshold_ready_(false),
      lag_(lag > 0U ? lag : 1U),
      autocorr_lag_(autocorr_lag > 0U ? autocorr_lag : 1U),
      settle_block_max_(0.0f),
      settle_block_evaluations_(0U),
      settle_block_count_(0U),
      settle_block_index_(0U) {
  reset_settled_level_();
  l1_tracker_.configure(l1_delta_capacity_(), lag_);
  ESP_LOGI(TAG, "Initialized weighted fusion (window=%u, threshold=%.3f, lag=%u, ac_lag=%u)",
           static_cast<unsigned>(window_size_), threshold_,
           static_cast<unsigned>(lag_), static_cast<unsigned>(autocorr_lag_));
}

uint16_t ClassicDetector::l1_delta_capacity_() const {
  return window_size_ > lag_ ? static_cast<uint16_t>(window_size_ - lag_) : 0U;
}

void ClassicDetector::configure_hampel(bool enabled, uint8_t window_size,
                                       float threshold) {
  BaseDetector::configure_hampel(enabled, window_size, threshold);
  l1_tracker_.configure_hampel(enabled, window_size, threshold);
}

void ClassicDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* selected_subcarriers,
                                     uint8_t num_subcarriers,
                                     int8_t rssi_dbm) {
  if (csi_data == nullptr) {
    ESP_LOGE(TAG, "process_packet: null CSI data");
    return;
  }
  (void) rssi_dbm;

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
  uint16_t count = 0U;
  const float* ordered = ordered_turbulence(count);
  if (ordered == nullptr || count < 3U) {
    return 0.0f;
  }

  float mean = 0.0f;
  for (uint16_t i = 0U; i < count; i++) {
    mean += ordered[i];
  }
  mean /= count;

  float variance = 0.0f;
  for (uint16_t i = 0U; i < count; i++) {
    const float diff = ordered[i] - mean;
    variance += diff * diff;
  }
  variance /= count;
  return calc_autocorrelation(ordered, count, mean, variance, autocorr_lag_);
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

  current_l1_delta_ = l1_tracker_.delta_lag_ratio();
  current_turb_autocorr_ = calculate_turb_autocorr_();
  current_logit_ = calculate_logit_(current_l1_delta_, current_turb_autocorr_);
  current_probability_ = sigmoid_(current_logit_);
  if (!adapted_threshold_ready_ && startup_logit_count_ < CLASSIC_STARTUP_SAMPLE_LIMIT) {
    startup_logits_[startup_logit_count_] = current_logit_;
    startup_logit_count_++;
  }
  observe_settled_level_();
  state_ = current_probability_ > threshold_ ? MotionState::MOTION : MotionState::IDLE;
}

float ClassicDetector::quantile_(const float* values, uint8_t count, float quantile) {
  if (values == nullptr || count == 0U) {
    return 0.0f;
  }
  float ordered[CLASSIC_STARTUP_SAMPLE_LIMIT];
  std::copy(values, values + count, ordered);
  std::sort(ordered, ordered + count);
  const float position = static_cast<float>(count - 1U) * quantile;
  const uint8_t lower = static_cast<uint8_t>(position);
  const uint8_t upper = std::min<uint8_t>(lower + 1U, count - 1U);
  const float fraction = position - static_cast<float>(lower);
  return ordered[lower] * (1.0f - fraction) + ordered[upper] * fraction;
}

float ClassicDetector::startup_quantile_() const {
  return startup_logit_count_ == 0U
             ? CLASSIC_TRAIN_IDLE_Q95_LOGIT
             : quantile_(startup_logits_, startup_logit_count_, CLASSIC_STARTUP_QUANTILE);
}

void ClassicDetector::reset_settled_level_() {
  settle_block_max_ = -1e9f;
  settle_block_evaluations_ = 0U;
  settle_block_count_ = 0U;
  settle_block_index_ = 0U;
}

/**
 * Lower the threshold once the session proves itself quieter than its startup.
 *
 * Startup calibration reads the opening of a session. When that opening is not
 * representative the threshold stays too high for the whole run, and nothing
 * revisits it: on one ESP32 capture the prefix is 4.1x noisier than the rest,
 * leaving the threshold at 3.8x the highest level the session ever reaches.
 *
 * The rule only ever lowers. It reads the median of per-block maxima, so a
 * single spike cannot move it and a stretch of real motion holds it high, which
 * is what keeps it from chasing the metric downward during activity. Mirrors
 * ClassicDetector._observe_settled_level in classic_detector.py.
 */
void ClassicDetector::observe_settled_level_() {
  if (!adapted_threshold_ready_) {
    return;
  }
  if (current_logit_ > settle_block_max_) {
    settle_block_max_ = current_logit_;
  }
  settle_block_evaluations_++;
  if (settle_block_evaluations_ < CLASSIC_SETTLE_BLOCK_EVALUATIONS) {
    return;
  }

  settle_blocks_[settle_block_index_] = settle_block_max_;
  settle_block_index_ = static_cast<uint8_t>((settle_block_index_ + 1U) % CLASSIC_SETTLE_BLOCKS);
  if (settle_block_count_ < CLASSIC_SETTLE_BLOCKS) {
    settle_block_count_++;
  }
  settle_block_max_ = -1e9f;
  settle_block_evaluations_ = 0U;
  if (settle_block_count_ < CLASSIC_SETTLE_BLOCKS) {
    return;
  }

  float ordered[CLASSIC_SETTLE_BLOCKS];
  std::copy(settle_blocks_, settle_blocks_ + CLASSIC_SETTLE_BLOCKS, ordered);
  std::sort(ordered, ordered + CLASSIC_SETTLE_BLOCKS);
  const float settled = ordered[CLASSIC_SETTLE_BLOCKS / 2];
  const float candidate = sigmoid_(settled + CLASSIC_SETTLE_MARGIN_LOGITS);
  if (candidate < threshold_) {
    threshold_ = clamp_threshold(candidate, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD);
  }
}

void ClassicDetector::on_startup_calibration_begin() {
  reset_settled_level_();
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
  ESP_LOGD(TAG, "Startup threshold prepared: %.6f (%u samples)",
           adapted_threshold_, static_cast<unsigned>(startup_logit_count_));
}

bool ClassicDetector::set_adaptive_threshold(float) {
  reset_settled_level_();
  threshold_ = adapted_threshold_ready_ ? adapted_threshold_ : CLASSIC_DEFAULT_THRESHOLD;
  return true;
}

bool ClassicDetector::set_threshold(float threshold) {
  if (!is_valid_threshold(threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD)) {
    ESP_LOGE(TAG, "Invalid threshold: %.6f (must be %.1f-%.1f)",
             threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD);
    return false;
  }
  threshold_ = threshold;
  ESP_LOGI(TAG, "Threshold updated: %.6f", threshold);
  return true;
}

void ClassicDetector::reset() {
  BaseDetector::reset();
  reset_settled_level_();
  current_probability_ = 0.0f;
  current_logit_ = 0.0f;
  current_l1_delta_ = 0.0f;
  current_turb_autocorr_ = 0.0f;
  state_ = MotionState::IDLE;
}

void ClassicDetector::clear_buffer() {
  BaseDetector::clear_buffer();
  reset_settled_level_();
  l1_tracker_.clear();
  current_probability_ = 0.0f;
  current_logit_ = 0.0f;
  current_l1_delta_ = 0.0f;
  current_turb_autocorr_ = 0.0f;
}

}  // namespace espectre
