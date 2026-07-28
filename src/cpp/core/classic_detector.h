/*
 * ESPectre - Classic Detector
 *
 * Vote-free weighted fusion of L1 profile displacement and turbulence
 * autocorrelation. Mirrors src/python/micro_espectre/classic_detector.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "base_detector.h"
#include "csi_format.h"
#include "csi_features.h"
#include "l1_delta_tracker.h"

#include <cstddef>
#include <cstdint>

namespace espectre {

constexpr float CLASSIC_DEFAULT_THRESHOLD = 0.8090618336447031f;
constexpr float CLASSIC_MIN_THRESHOLD = 0.0f;
constexpr float CLASSIC_MAX_THRESHOLD = 1.0f;
constexpr float CLASSIC_STARTUP_THRESHOLD_FACTOR = 1.0f;

constexpr float CLASSIC_L1_CENTER = 1.4372828727159759f;
constexpr float CLASSIC_L1_SCALE = 0.5846221043293537f;
constexpr float CLASSIC_L1_WEIGHT = 2.807005032259383f;
constexpr float CLASSIC_AUTOCORR_CENTER = 0.3899157842282158f;
constexpr float CLASSIC_AUTOCORR_SCALE = 0.3789361406116048f;
constexpr float CLASSIC_AUTOCORR_WEIGHT = 4.0307753529344765f;
constexpr float CLASSIC_INTERCEPT = 0.7924447436944712f;

constexpr float CLASSIC_TRAIN_IDLE_Q95_LOGIT = -0.6116129330770868f;
constexpr float CLASSIC_STARTUP_QUANTILE = 0.95f;
constexpr float CLASSIC_STARTUP_STRENGTH = 0.75f;
constexpr uint8_t CLASSIC_STARTUP_SAMPLE_LIMIT = 64U;

// Settled-level rule: how long the stream has to stay quiet before the startup
// threshold is allowed to come down, and by how much margin above the level it
// settled at. 12 blocks of 20 evaluations is 60 s at the nominal cadence. The
// margin is in logit units; 3.0 is the largest value that still recovers the
// ESP32 capture, and below 2.0 the empty-room recordings start to alarm.
constexpr uint8_t CLASSIC_SETTLE_BLOCKS = 12U;
constexpr uint8_t CLASSIC_SETTLE_BLOCK_EVALUATIONS = 20U;
constexpr float CLASSIC_SETTLE_MARGIN_LOGITS = 3.0f;

class ClassicDetector : public BaseDetector {
 public:
  /**
   * @param window_size Detector window in packets
   * @param threshold Motion probability threshold
   * @param lag Profile-displacement distance in packets
   * @param autocorr_lag Turbulence autocorrelation distance in packets
   *
   * Production uses the nominal-rate defaults. Alternate lags are exposed for
   * replay experiments only: changing either feature offset requires validating
   * the fitted coefficients before deployment. See detector_timing.h.
   */
  ClassicDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE,
                  float threshold = CLASSIC_DEFAULT_THRESHOLD,
                  uint16_t lag = L1_DELTA_LAG,
                  uint16_t autocorr_lag = 1U);

  ~ClassicDetector() override = default;
  ClassicDetector(ClassicDetector&& other) noexcept = default;
  ClassicDetector& operator=(ClassicDetector&& other) noexcept = default;
  ClassicDetector(const ClassicDetector&) = delete;
  ClassicDetector& operator=(const ClassicDetector&) = delete;

  void process_packet(const int8_t* csi_data, size_t csi_len,
                      const uint8_t* selected_subcarriers = nullptr,
                      uint8_t num_subcarriers = 0,
                      int8_t rssi_dbm = INT8_MIN) override;
  void update_state() override;
  void reset() override;
  void clear_buffer() override;
  bool is_ready() const override;
  bool set_threshold(float threshold) override;
  bool set_adaptive_threshold(float threshold) override;
  float get_threshold() const override { return threshold_; }
  const char* get_name() const override { return "Classic"; }
  float get_startup_threshold_factor() const override {
    return CLASSIC_STARTUP_THRESHOLD_FACTOR;
  }
  bool startup_gate_enabled() const override { return true; }
  void on_startup_calibration_begin() override;
  void on_startup_calibration_complete() override;
  void configure_hampel(bool enabled,
                        uint8_t window_size = HAMPEL_TURBULENCE_WINDOW_DEFAULT,
                        float threshold = HAMPEL_TURBULENCE_THRESHOLD_DEFAULT) override;

  // Named for what it holds: the L1 lag ratio, not the L1 mean it replaced
  // on 2026-07-26. The mean carried the link's noise floor.
  float get_lag_ratio() const { return current_lag_ratio_; }
  float get_turb_autocorr() const { return current_turb_autocorr_; }
  float get_logit() const { return current_logit_; }

 private:
  uint16_t l1_delta_capacity_() const;
  float calculate_turb_autocorr_() const;
  float calculate_logit_(float lag_ratio, float turb_autocorr) const;
  static float sigmoid_(float value);
  static float quantile_(const float* values, uint8_t count, float quantile);
  float startup_quantile_() const;
  void observe_settled_level_();
  void reset_settled_level_();
  void clear_fusion_inputs_();

  float threshold_;
  float current_logit_;
  float current_lag_ratio_;
  float current_turb_autocorr_;
  float startup_logits_[CLASSIC_STARTUP_SAMPLE_LIMIT]{};
  uint8_t startup_logit_count_;
  float adapted_threshold_;
  bool adapted_threshold_ready_;
  uint16_t lag_;
  uint16_t autocorr_lag_;
  float settle_blocks_[CLASSIC_SETTLE_BLOCKS]{};
  float settle_block_max_;
  uint8_t settle_block_evaluations_;
  uint8_t settle_block_count_;
  uint8_t settle_block_index_;
  L1DeltaTracker l1_tracker_;
};

}  // namespace espectre
