/*
 * ESPectre - Lightweight Detector Implementation
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "lightweight_detector.h"

#include "espectre_log.h"
#include "threshold.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace espectre {

static const char* TAG = "LightweightDetector";

LightweightDetector::LightweightDetector(uint16_t window_size, float threshold,
                                         uint16_t autocorr_lag)
    : BaseDetector(window_size),
      threshold_(clamp_threshold(threshold, LIGHTWEIGHT_MIN_THRESHOLD, LIGHTWEIGHT_MAX_THRESHOLD)),
      current_logit_(0.0f),
      current_turb_autocorr_(0.0f),
      current_turb_iqr_over_mean_aggr_(0.0f),
      startup_logit_count_(0U),
      adapted_threshold_(LIGHTWEIGHT_DEFAULT_THRESHOLD),
      adapted_threshold_ready_(false),
      manual_threshold_override_(false),
      autocorr_lag_(autocorr_lag > 0U ? autocorr_lag : 1U),
      settle_block_max_(0.0f),
      settle_block_evaluations_(0U),
      settle_block_count_(0U),
      settle_block_index_(0U),
      aggregated_turbulence_buffer_(alloc_zeroed_floats(window_size_)) {
  reset_settled_level_();
  aggregated_turbulence_.bind(aggregated_turbulence_buffer_.get(), window_size_);
  if (aggregated_turbulence_buffer_ == nullptr) {
    ESPECTRE_LOGE(TAG, "Failed to allocate aggregated turbulence buffer");
  }
  ESPECTRE_LOGI(TAG, "Initialized weighted fusion (window=%u, threshold=%.3f, ac_lag=%u)",
           static_cast<unsigned>(window_size_), threshold_,
           static_cast<unsigned>(autocorr_lag_));
}

void LightweightDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* selected_subcarriers,
                                     uint8_t num_subcarriers,
                                     int8_t rssi_dbm) {
  if (csi_data == nullptr) {
    ESPECTRE_LOGE(TAG, "process_packet: null CSI data");
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
  const uint8_t packet_count = fill_packet_subcarrier_energies(
      csi_data, csi_len, packet_amplitudes, HT20_NUM_SUBCARRIERS);
  detail::required_energies_to_amplitudes<TURB_IQR_AGGREGATION_WIDTH>(
      packet_amplitudes, packet_count, resolved_subcarriers, resolved_count, true);
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

bool LightweightDetector::is_ready() const {
  return BaseDetector::is_ready() &&
         aggregated_turbulence_.count() >= window_size_ &&
         aggregated_turbulence_.valid_count() >= minimum_valid_samples_;
}

float LightweightDetector::calculate_turb_autocorr_() const {
  uint16_t count = 0U;
  const float* ordered = ordered_turbulence(count);
  if (ordered == nullptr || count < 3U) {
    return 0.0f;
  }

  float sum = 0.0f;
  uint16_t valid_count = 0U;
  for (uint16_t i = 0U; i < count; ++i) {
    if (!std::isfinite(ordered[i])) continue;
    sum += ordered[i];
    ++valid_count;
  }
  if (valid_count < 2U) return 0.0f;
  const float mean = sum / valid_count;
  float variance_sum = 0.0f;
  for (uint16_t i = 0U; i < count; ++i) {
    if (!std::isfinite(ordered[i])) continue;
    const float diff = ordered[i] - mean;
    variance_sum += diff * diff;
  }
  return calc_autocorrelation(
      ordered, count, mean, variance_sum / valid_count, autocorr_lag_);
}

float LightweightDetector::calculate_turb_iqr_over_mean_aggr_() const {
  if (aggregated_turbulence_.count() < window_size_ || ordered_turbulence_ == nullptr) {
    return 0.0f;
  }
  uint16_t ordered_count = 0U;
  if (aggregated_turbulence_.ordered_view(ordered_turbulence_, window_size_, ordered_count) == nullptr ||
      ordered_count != window_size_) {
    return 0.0f;
  }
  uint16_t valid_count = 0U;
  float sum = 0.0f;
  for (uint16_t i = 0U; i < window_size_; ++i) {
    const float value = ordered_turbulence_[i];
    if (!std::isfinite(value)) continue;
    ordered_turbulence_[valid_count++] = value;
    sum += value;
  }
  if (valid_count < 2U) return 0.0f;
  const float mean = sum / valid_count;
  const float iqr = percentile_in_place(
      ordered_turbulence_, valid_count, 0.75f) - percentile_in_place(
      ordered_turbulence_, valid_count, 0.25f);
  return iqr / std::max(std::fabs(mean), 1e-6f);
}

float LightweightDetector::calculate_logit_(float turb_autocorr,
                                        float turb_iqr_over_mean_aggr) const {
  const float normalized_autocorr =
      (turb_autocorr - LIGHTWEIGHT_AUTOCORR_CENTER) / LIGHTWEIGHT_AUTOCORR_SCALE;
  const float normalized_iqr =
      (turb_iqr_over_mean_aggr - LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_CENTER) /
      LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_SCALE;
  return LIGHTWEIGHT_INTERCEPT + LIGHTWEIGHT_AUTOCORR_WEIGHT * normalized_autocorr +
         LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_WEIGHT * normalized_iqr;
}

float LightweightDetector::sigmoid_(float value) {
  if (value < -20.0f) return 0.0f;
  if (value > 20.0f) return 1.0f;
  return 1.0f / (1.0f + std::exp(-value));
}

void LightweightDetector::update_state() {
  if (!is_ready()) {
    clear_evaluation_state_();
    return;
  }

  current_turb_autocorr_ = calculate_turb_autocorr_();
  current_turb_iqr_over_mean_aggr_ = calculate_turb_iqr_over_mean_aggr_();
  current_logit_ = calculate_logit_(current_turb_autocorr_,
                                    current_turb_iqr_over_mean_aggr_);
  current_metric_ = sigmoid_(current_logit_);
  if (!adapted_threshold_ready_ && startup_logit_count_ < LIGHTWEIGHT_STARTUP_SAMPLE_LIMIT) {
    startup_logits_[startup_logit_count_] = current_logit_;
    startup_logit_count_++;
  }
  observe_settled_level_();
  state_ = current_metric_ > threshold_ ? MotionState::MOTION : MotionState::IDLE;
}

float LightweightDetector::quantile_(const float* values, uint8_t count, float quantile) {
  if (values == nullptr || count == 0U) {
    return 0.0f;
  }
  float ordered[LIGHTWEIGHT_STARTUP_SAMPLE_LIMIT];
  std::copy(values, values + count, ordered);
  std::sort(ordered, ordered + count);
  const float position = static_cast<float>(count - 1U) * quantile;
  const uint8_t lower = static_cast<uint8_t>(position);
  const uint8_t upper = std::min<uint8_t>(lower + 1U, count - 1U);
  const float fraction = position - static_cast<float>(lower);
  return ordered[lower] * (1.0f - fraction) + ordered[upper] * fraction;
}

float LightweightDetector::startup_quantile_() const {
  return startup_logit_count_ == 0U
             ? LIGHTWEIGHT_TRAIN_IDLE_Q95_LOGIT
             : quantile_(startup_logits_, startup_logit_count_, LIGHTWEIGHT_STARTUP_QUANTILE);
}

void LightweightDetector::reset_settled_level_() {
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
 * LightweightDetector._observe_settled_level in lightweight_detector.py.
 */
void LightweightDetector::observe_settled_level_() {
  if (!adapted_threshold_ready_ || manual_threshold_override_) {
    return;
  }
  if (current_logit_ > settle_block_max_) {
    settle_block_max_ = current_logit_;
  }
  settle_block_evaluations_++;
  if (settle_block_evaluations_ < LIGHTWEIGHT_SETTLE_BLOCK_EVALUATIONS) {
    return;
  }

  settle_blocks_[settle_block_index_] = settle_block_max_;
  settle_block_index_ = static_cast<uint8_t>((settle_block_index_ + 1U) % LIGHTWEIGHT_SETTLE_BLOCKS);
  if (settle_block_count_ < LIGHTWEIGHT_SETTLE_BLOCKS) {
    settle_block_count_++;
  }
  settle_block_max_ = -1e9f;
  settle_block_evaluations_ = 0U;
  if (settle_block_count_ < LIGHTWEIGHT_SETTLE_BLOCKS) {
    return;
  }

  float ordered[LIGHTWEIGHT_SETTLE_BLOCKS];
  std::copy(settle_blocks_, settle_blocks_ + LIGHTWEIGHT_SETTLE_BLOCKS, ordered);
  std::sort(ordered, ordered + LIGHTWEIGHT_SETTLE_BLOCKS);
  const float settled = ordered[LIGHTWEIGHT_SETTLE_BLOCKS / 2];
  const float candidate = sigmoid_(settled + LIGHTWEIGHT_SETTLE_MARGIN_LOGITS);
  if (candidate < threshold_) {
    threshold_ = clamp_threshold(candidate, LIGHTWEIGHT_MIN_THRESHOLD, LIGHTWEIGHT_MAX_THRESHOLD);
  }
}

void LightweightDetector::on_startup_calibration_begin() {
  reset_settled_level_();
  manual_threshold_override_ = false;
  startup_logit_count_ = 0U;
  adapted_threshold_ready_ = false;
}

void LightweightDetector::on_startup_calibration_complete() {
  const float base_logit = std::log(LIGHTWEIGHT_DEFAULT_THRESHOLD /
                                    (1.0f - LIGHTWEIGHT_DEFAULT_THRESHOLD));
  const float adapted_logit = base_logit + LIGHTWEIGHT_STARTUP_STRENGTH *
      (startup_quantile_() - LIGHTWEIGHT_TRAIN_IDLE_Q95_LOGIT);
  adapted_threshold_ = sigmoid_(adapted_logit);
  adapted_threshold_ready_ = true;
  ESPECTRE_LOGD(TAG, "Startup threshold prepared: %.6f (%u samples)",
           adapted_threshold_, static_cast<unsigned>(startup_logit_count_));
}

bool LightweightDetector::set_adaptive_threshold(float) {
  reset_settled_level_();
  manual_threshold_override_ = false;
  threshold_ = adapted_threshold_ready_ ? adapted_threshold_ : LIGHTWEIGHT_DEFAULT_THRESHOLD;
  return true;
}

bool LightweightDetector::set_threshold(float threshold) {
  if (!is_valid_threshold(threshold, LIGHTWEIGHT_MIN_THRESHOLD, LIGHTWEIGHT_MAX_THRESHOLD)) {
    ESPECTRE_LOGE(TAG, "Invalid threshold: %.6f (must be %.1f-%.1f)",
             threshold, LIGHTWEIGHT_MIN_THRESHOLD, LIGHTWEIGHT_MAX_THRESHOLD);
    return false;
  }
  threshold_ = threshold;
  manual_threshold_override_ = true;
  ESPECTRE_LOGI(TAG, "Threshold updated: %.6f", threshold);
  return true;
}

// Both overrides clear only the Lightweight-specific derived values; the shared
// metric and motion state are cleared by the base.
void LightweightDetector::reset() {
  BaseDetector::reset();
  reset_settled_level_();
  clear_fusion_inputs_();
}

void LightweightDetector::clear_buffer() {
  BaseDetector::clear_buffer();
  reset_settled_level_();
  aggregated_turbulence_.clear();
  clear_fusion_inputs_();
}

void LightweightDetector::configure_hampel(bool enabled, uint8_t window_size,
                                       float threshold) {
  BaseDetector::configure_hampel(enabled, window_size, threshold);
  aggregated_turbulence_.configure_hampel(enabled, window_size, threshold);
}

void LightweightDetector::configure_lowpass(bool enabled, float cutoff_hz) {
  BaseDetector::configure_lowpass(enabled, cutoff_hz);
  aggregated_turbulence_.configure_lowpass(enabled, cutoff_hz);
}

void LightweightDetector::add_aggregated_turbulence_(float turbulence) {
  aggregated_turbulence_.add(turbulence);
}

void LightweightDetector::advance_missing_slots(uint32_t count) {
  BaseDetector::advance_missing_slots(count);
  aggregated_turbulence_.advance_missing_slots(count);
}

void LightweightDetector::clear_fusion_inputs_() {
  current_logit_ = 0.0f;
  current_turb_autocorr_ = 0.0f;
  current_turb_iqr_over_mean_aggr_ = 0.0f;
}

}  // namespace espectre
