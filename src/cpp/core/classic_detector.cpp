/*
 * ESPectre - Classic Detector Implementation
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "classic_detector.h"

#include "espectre_log.h"
#include "threshold.h"

#include <algorithm>
#include <cmath>

namespace espectre {

static const char* TAG = "ClassicDetector";

ClassicDetector::ClassicDetector(uint16_t window_size, float threshold,
                                 uint16_t autocorr_lag)
    : BaseDetector(window_size),
      threshold_(clamp_threshold(threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD)),
      current_logit_(0.0f),
      current_turb_autocorr_(0.0f),
      current_turb_iqr_over_mean_aggr_(0.0f),
      startup_logit_count_(0U),
      adapted_threshold_(CLASSIC_DEFAULT_THRESHOLD),
      adapted_threshold_ready_(false),
      autocorr_lag_(autocorr_lag > 0U ? autocorr_lag : 1U),
      settle_block_max_(0.0f),
      settle_block_evaluations_(0U),
      settle_block_count_(0U),
      settle_block_index_(0U),
      aggregated_turbulence_buffer_(window_size_, 0.0f),
      aggregated_turbulence_index_(0U),
      aggregated_turbulence_count_(0U) {
  reset_settled_level_();
  hampel_turbulence_init(&aggregated_hampel_state_,
                         HAMPEL_TURBULENCE_WINDOW_DEFAULT,
                         HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, false);
  lowpass_filter_init(&aggregated_lowpass_state_, LOWPASS_CUTOFF_DEFAULT,
                      LOWPASS_SAMPLE_RATE, false);
  ESP_LOGI(TAG, "Initialized weighted fusion (window=%u, threshold=%.3f, ac_lag=%u)",
           static_cast<unsigned>(window_size_), threshold_,
           static_cast<unsigned>(autocorr_lag_));
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

  const uint8_t* resolved_subcarriers = selected_subcarriers;
  uint8_t resolved_count = num_subcarriers;
  if (resolved_subcarriers == nullptr || resolved_count == 0U) {
    resolved_subcarriers = DEFAULT_SUBCARRIERS;
    resolved_count = HT20_SELECTED_BAND_SIZE;
  }
  float packet_amplitudes[HT20_NUM_SUBCARRIERS]{};
  const uint8_t packet_count = extract_packet_subcarrier_amplitudes(
      csi_data, csi_len, packet_amplitudes, HT20_NUM_SUBCARRIERS);
  float amplitudes[HT20_SELECTED_BAND_SIZE]{};
  const uint8_t amplitude_count = select_subcarrier_amplitudes(
      packet_amplitudes, packet_count, resolved_subcarriers, resolved_count,
      amplitudes, HT20_SELECTED_BAND_SIZE);
  process_amplitudes(amplitudes, amplitude_count);
  float aggregated_amplitudes[HT20_SELECTED_BAND_SIZE]{};
  const uint8_t aggregated_count =
      select_adjacent_aggregated_subcarrier_amplitudes(
          packet_amplitudes, packet_count, resolved_subcarriers, resolved_count,
          TURB_IQR_AGGREGATION_WIDTH, aggregated_amplitudes,
          HT20_SELECTED_BAND_SIZE);
  add_aggregated_turbulence_(calculate_spatial_turbulence_from_amplitudes(
      aggregated_amplitudes, aggregated_count));
}

bool ClassicDetector::is_ready() const {
  return buffer_count_ >= window_size_ &&
         aggregated_turbulence_count_ >= window_size_;
}

float ClassicDetector::calculate_turb_autocorr_() const {
  uint16_t count = 0U;
  const float* ordered = ordered_turbulence(count);
  if (ordered == nullptr || count < 3U) {
    return 0.0f;
  }

  const MeanVariance stats = calculate_mean_variance_two_pass(ordered, count);
  return calc_autocorrelation(ordered, count, stats.mean, stats.variance, autocorr_lag_);
}

float ClassicDetector::calculate_turb_iqr_over_mean_aggr_() const {
  if (aggregated_turbulence_count_ < window_size_ ||
      ordered_turbulence_ == nullptr) return 0.0f;
  const uint16_t tail = static_cast<uint16_t>(
      window_size_ - aggregated_turbulence_index_);
  std::copy(aggregated_turbulence_buffer_.begin() + aggregated_turbulence_index_,
            aggregated_turbulence_buffer_.end(), ordered_turbulence_);
  std::copy(aggregated_turbulence_buffer_.begin(),
            aggregated_turbulence_buffer_.begin() + aggregated_turbulence_index_,
            ordered_turbulence_ + tail);
  const MeanVariance stats = calculate_mean_variance_two_pass(
      ordered_turbulence_, window_size_);
  std::sort(ordered_turbulence_, ordered_turbulence_ + window_size_);
  const float iqr = percentile_from_sorted(
      ordered_turbulence_, window_size_, 0.75f) - percentile_from_sorted(
      ordered_turbulence_, window_size_, 0.25f);
  return iqr / std::max(std::fabs(stats.mean), 1e-6f);
}

float ClassicDetector::calculate_logit_(float turb_autocorr,
                                        float turb_iqr_over_mean_aggr) const {
  const float normalized_autocorr =
      (turb_autocorr - CLASSIC_AUTOCORR_CENTER) / CLASSIC_AUTOCORR_SCALE;
  const float normalized_iqr =
      (turb_iqr_over_mean_aggr - CLASSIC_TURB_IQR_OVER_MEAN_AGGR_CENTER) /
      CLASSIC_TURB_IQR_OVER_MEAN_AGGR_SCALE;
  return CLASSIC_INTERCEPT + CLASSIC_AUTOCORR_WEIGHT * normalized_autocorr +
         CLASSIC_TURB_IQR_OVER_MEAN_AGGR_WEIGHT * normalized_iqr;
}

float ClassicDetector::sigmoid_(float value) {
  if (value < -20.0f) return 0.0f;
  if (value > 20.0f) return 1.0f;
  return 1.0f / (1.0f + std::exp(-value));
}

void ClassicDetector::update_state() {
  if (!is_ready()) {
    clear_evaluation_state_();
    return;
  }

  current_turb_autocorr_ = calculate_turb_autocorr_();
  current_turb_iqr_over_mean_aggr_ = calculate_turb_iqr_over_mean_aggr_();
  current_logit_ = calculate_logit_(current_turb_autocorr_,
                                    current_turb_iqr_over_mean_aggr_);
  current_metric_ = sigmoid_(current_logit_);
  if (!adapted_threshold_ready_ && startup_logit_count_ < CLASSIC_STARTUP_SAMPLE_LIMIT) {
    startup_logits_[startup_logit_count_] = current_logit_;
    startup_logit_count_++;
  }
  observe_settled_level_();
  state_ = current_metric_ > threshold_ ? MotionState::MOTION : MotionState::IDLE;
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

// Both overrides clear only the Classic-specific derived values; the shared
// metric and motion state are cleared by the base.
void ClassicDetector::reset() {
  BaseDetector::reset();
  reset_settled_level_();
  clear_fusion_inputs_();
}

void ClassicDetector::clear_buffer() {
  BaseDetector::clear_buffer();
  reset_settled_level_();
  std::fill(aggregated_turbulence_buffer_.begin(),
            aggregated_turbulence_buffer_.end(), 0.0f);
  aggregated_turbulence_index_ = 0U;
  aggregated_turbulence_count_ = 0U;
  hampel_turbulence_init(&aggregated_hampel_state_,
                         aggregated_hampel_state_.window_size,
                         aggregated_hampel_state_.threshold,
                         aggregated_hampel_state_.enabled);
  lowpass_filter_reset(&aggregated_lowpass_state_);
  clear_fusion_inputs_();
}

void ClassicDetector::configure_hampel(bool enabled, uint8_t window_size,
                                       float threshold) {
  BaseDetector::configure_hampel(enabled, window_size, threshold);
  hampel_turbulence_init(&aggregated_hampel_state_, window_size, threshold,
                         enabled);
}

void ClassicDetector::configure_lowpass(bool enabled, float cutoff_hz) {
  BaseDetector::configure_lowpass(enabled, cutoff_hz);
  lowpass_filter_init(&aggregated_lowpass_state_, cutoff_hz,
                      LOWPASS_SAMPLE_RATE, enabled);
}

void ClassicDetector::add_aggregated_turbulence_(float turbulence) {
  const float hampel = hampel_filter_turbulence(
      &aggregated_hampel_state_, turbulence);
  const float filtered = lowpass_filter_apply(&aggregated_lowpass_state_, hampel);
  aggregated_turbulence_buffer_[aggregated_turbulence_index_] = filtered;
  aggregated_turbulence_index_++;
  if (aggregated_turbulence_index_ >= window_size_) aggregated_turbulence_index_ = 0U;
  if (aggregated_turbulence_count_ < window_size_) aggregated_turbulence_count_++;
}

void ClassicDetector::clear_fusion_inputs_() {
  current_logit_ = 0.0f;
  current_turb_autocorr_ = 0.0f;
  current_turb_iqr_over_mean_aggr_ = 0.0f;
}

}  // namespace espectre
