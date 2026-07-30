/*
 * ESPectre - Classic Detector
 *
 * Vote-free weighted fusion of turbulence autocorrelation and channel
 * frequency-coherence curve spread. Mirrors
 * src/python/micro_espectre/classic_detector.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "base_detector.h"
#include "csi_format.h"
#include "csi_features.h"
#include "l1_delta_tracker.h"
#include "ml_feature_trackers.h"

#include <cstddef>
#include <cstdint>

namespace espectre {

constexpr float CLASSIC_DEFAULT_THRESHOLD = 0.6959857092777915f;
constexpr float CLASSIC_MIN_THRESHOLD = 0.0f;
constexpr float CLASSIC_MAX_THRESHOLD = 1.0f;
constexpr float CLASSIC_STARTUP_THRESHOLD_FACTOR = 1.0f;

constexpr float CLASSIC_AUTOCORR_CENTER = 0.4054217624112162f;
constexpr float CLASSIC_AUTOCORR_SCALE = 0.36758465285308456f;
constexpr float CLASSIC_AUTOCORR_WEIGHT = 5.501903876354938f;
constexpr float CLASSIC_FREQ_COH_CURVE_STD_CENTER = 0.014752343728085844f;
constexpr float CLASSIC_FREQ_COH_CURVE_STD_SCALE = 0.02602884858084268f;
constexpr float CLASSIC_FREQ_COH_CURVE_STD_WEIGHT = 4.040278978639349f;
constexpr float CLASSIC_INTERCEPT = 0.5020797967446212f;

constexpr float CLASSIC_TRAIN_IDLE_Q95_LOGIT = -0.6930943805793314f;
constexpr float CLASSIC_STARTUP_QUANTILE = 0.95f;
constexpr float CLASSIC_STARTUP_STRENGTH = 0.5f;
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
   * @param lag Temporal offset in packets for the curve-spread tracker
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

  float get_turb_autocorr() const { return current_turb_autocorr_; }
  float get_chan_freq_coh_curve_std() const {
    return current_chan_freq_coh_curve_std_;
  }
  float get_logit() const { return current_logit_; }

 private:
  uint16_t l1_delta_capacity_() const;
  float calculate_turb_autocorr_() const;
  float calculate_logit_(float turb_autocorr,
                         float chan_freq_coh_curve_std) const;
  static float sigmoid_(float value);
  static float quantile_(const float* values, uint8_t count, float quantile);
  float startup_quantile_() const;
  void observe_settled_level_();
  void reset_settled_level_();
  void clear_fusion_inputs_();

  float threshold_;
  float current_logit_;
  float current_turb_autocorr_;
  float current_chan_freq_coh_curve_std_;
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
  ChannelShapeTracker shape_tracker_;
};

}  // namespace espectre
