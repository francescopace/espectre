/*
 * ESPectre - Lightweight Detector
 *
 * Vote-free weighted fusion of turbulence autocorrelation and aggregated
 * turbulence IQR. Mirrors
 * src/python/micro_espectre/lightweight_detector.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "base_detector.h"
#include "csi_format.h"
#include "csi_features.h"
#include "filtered_turbulence_ring.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace espectre {

constexpr float LIGHTWEIGHT_AUTOCORR_CENTER = 0.3919344866784947f;
constexpr float LIGHTWEIGHT_AUTOCORR_SCALE = 0.3798648330757351f;
constexpr float LIGHTWEIGHT_AUTOCORR_WEIGHT = 5.083034533668216f;
constexpr float LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_CENTER = 0.24612139211074338f;
constexpr float LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_SCALE = 0.20056599613462603f;
constexpr float LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_WEIGHT = 4.997501915217463f;
constexpr float LIGHTWEIGHT_INTERCEPT = 1.0776769868761f;

constexpr float LIGHTWEIGHT_TRAIN_IDLE_Q95_LOGIT = -2.253902812716911f;
constexpr float LIGHTWEIGHT_STARTUP_QUANTILE = 0.95f;
constexpr float LIGHTWEIGHT_STARTUP_STRENGTH = 0.5f;
constexpr uint8_t LIGHTWEIGHT_STARTUP_SAMPLE_LIMIT = 64U;

// Settled-level rule: how long the stream has to stay quiet before the startup
// threshold is allowed to come down, and by how much margin above the level it
// settled at. 12 blocks of 20 evaluations is 60 s at the nominal cadence. The
// margin is in logit units; 2.7 is the conservative temporal-window operating
// point that clears the weak-link recall floor without changing the measured
// normal-link or quiet-room FP tails.
constexpr uint8_t LIGHTWEIGHT_SETTLE_BLOCKS = 12U;
constexpr uint8_t LIGHTWEIGHT_SETTLE_BLOCK_EVALUATIONS = 20U;
constexpr float LIGHTWEIGHT_SETTLE_MARGIN_LOGITS = 2.7f;

/**
 * The default detector: self-calibrating, no training data required.
 *
 * Fuses turbulence autocorrelation with robust spread from a five-bin
 * aggregated turbulence stream, and adapts its threshold to the room
 * during startup calibration. After that, a long quiet stretch can still
 * lower the live threshold when the opening was noisier than the rest of
 * the session. The full runtime emits `IRuntimeListener::on_threshold_changed()`
 * for that drop; a core-only integration must re-read `get_threshold()` after
 * `update_state()`. Prefer it unless you have a reason to run
 * `HighAccuracyDetector`.
 *
 * Most integrations never construct one: `RuntimeConfig::detection_algorithm`
 * selects it and the runtime owns the lifecycle. Drive it directly only on the
 * core-only path, where your firmware already captures CSI:
 *
 * @code
 * espectre::LightweightDetector detector;
 * // per packet, from your capture callback:
 * detector.process_packet(csi, csi_len, espectre::DEFAULT_SUBCARRIERS,
 *                         espectre::HT20_SELECTED_BAND_SIZE, rssi_dbm);
 * // on your evaluation cadence:
 * detector.update_state();
 * if (detector.is_ready() && detector.get_state() == espectre::MotionState::MOTION) { ... }
 * @endcode
 *
 * `is_ready()` is false until the window fills; results before that are not
 * meaningful. See `runtime/esp_idf/csi_pipeline.cpp` for the reference
 * normalization, cadence, and hit filtering around these calls, and
 * `docs/ALGORITHMS.md` for the algorithm itself.
 *
 * @par Threading
 * Not thread-safe. `process_packet()` and `update_state()` must not run
 * concurrently.
 */
class LightweightDetector : public BaseDetector {
 public:
  /**
   * @param window_size Detector window in packets
   * @param threshold Motion probability threshold
   * @param autocorr_lag Turbulence autocorrelation distance in packets
   *
   * Production uses the nominal-rate default. Alternate lags are exposed for
   * replay experiments only: changing the feature offset requires validating
   * the fitted coefficients before deployment. See detector_timing.h.
   */
  LightweightDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE,
                      float threshold = LIGHTWEIGHT_DEFAULT_THRESHOLD,
                      uint16_t autocorr_lag = 1U);

  ~LightweightDetector() override = default;
  LightweightDetector(LightweightDetector&& other) noexcept = default;
  LightweightDetector& operator=(LightweightDetector&& other) noexcept = default;
  LightweightDetector(const LightweightDetector&) = delete;
  LightweightDetector& operator=(const LightweightDetector&) = delete;

  void process_packet(const int8_t* csi_data, size_t csi_len,
                      const uint8_t* selected_subcarriers = nullptr,
                      uint8_t num_subcarriers = 0,
                      int8_t rssi_dbm = INT8_MIN) override;
  void advance_missing_slots(uint32_t count) override;
  void update_state() override;
  void reset() override;
  void clear_buffer() override;
  void configure_hampel(bool enabled,
                        uint8_t window_size = HAMPEL_TURBULENCE_WINDOW_DEFAULT,
                        float threshold = HAMPEL_TURBULENCE_THRESHOLD_DEFAULT) override;
  void configure_lowpass(bool enabled,
                         float cutoff_hz = LOWPASS_CUTOFF_DEFAULT) override;
  bool is_ready() const override;
  bool set_threshold(float threshold) override;
  bool set_adaptive_threshold(float threshold) override;
  float get_threshold() const override { return threshold_; }
  const char* get_name() const override { return "Lightweight"; }
  float get_startup_threshold_factor() const override {
    return LIGHTWEIGHT_STARTUP_THRESHOLD_FACTOR;
  }
  bool startup_gate_enabled() const override { return true; }
  void on_startup_calibration_begin() override;
  void on_startup_calibration_complete() override;

  float get_turb_autocorr() const { return current_turb_autocorr_; }
  float get_turb_iqr_over_mean_aggr() const { return current_turb_iqr_over_mean_aggr_; }
  float get_logit() const { return current_logit_; }

 private:
  float calculate_turb_autocorr_() const;
  float calculate_turb_iqr_over_mean_aggr_() const;
  float calculate_logit_(float turb_autocorr, float turb_iqr_over_mean_aggr) const;
  void add_aggregated_turbulence_(float turbulence);
  static float sigmoid_(float value);
  static float quantile_(const float* values, uint8_t count, float quantile);
  float startup_quantile_() const;
  void observe_settled_level_();
  void reset_settled_level_();
  void clear_fusion_inputs_();

  float threshold_;
  float current_logit_;
  float current_turb_autocorr_;
  float current_turb_iqr_over_mean_aggr_;
  float startup_logits_[LIGHTWEIGHT_STARTUP_SAMPLE_LIMIT]{};
  uint8_t startup_logit_count_;
  float adapted_threshold_;
  bool adapted_threshold_ready_;
  uint16_t autocorr_lag_;
  float settle_blocks_[LIGHTWEIGHT_SETTLE_BLOCKS]{};
  float settle_block_max_;
  uint8_t settle_block_evaluations_;
  uint8_t settle_block_count_;
  uint8_t settle_block_index_;
  std::vector<float> aggregated_turbulence_buffer_;
  FilteredTurbulenceRing aggregated_turbulence_;
};

}  // namespace espectre
