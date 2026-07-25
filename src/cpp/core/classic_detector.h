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

constexpr float CLASSIC_DEFAULT_THRESHOLD = 0.7892048221516996f;
constexpr float CLASSIC_MIN_THRESHOLD = 0.0f;
constexpr float CLASSIC_MAX_THRESHOLD = 1.0f;
constexpr float CLASSIC_STARTUP_THRESHOLD_FACTOR = 1.0f;

constexpr float CLASSIC_L1_CENTER = 0.05739352783479646f;
constexpr float CLASSIC_L1_SCALE = 0.04216339546436966f;
constexpr float CLASSIC_L1_WEIGHT = 1.241242648072718f;
constexpr float CLASSIC_AUTOCORR_CENTER = 0.3919767063392909f;
constexpr float CLASSIC_AUTOCORR_SCALE = 0.37575055837461374f;
constexpr float CLASSIC_AUTOCORR_WEIGHT = 5.032184078396507f;
constexpr float CLASSIC_INTERCEPT = -0.26428814254089134f;

constexpr float CLASSIC_TRAIN_IDLE_Q95_LOGIT = -0.06185062000916678f;
constexpr float CLASSIC_STARTUP_QUANTILE = 0.95f;
constexpr float CLASSIC_STARTUP_STRENGTH = 0.75f;
constexpr uint8_t CLASSIC_STARTUP_SAMPLE_LIMIT = 64U;
constexpr float CLASSIC_L1_NOISE_BLEND_START = CLASSIC_L1_CENTER + CLASSIC_L1_SCALE;
constexpr float CLASSIC_L1_NOISE_BLEND_END =
    CLASSIC_L1_CENTER + 2.5f * CLASSIC_L1_SCALE;
constexpr float CLASSIC_L1_EXCURSION_GAIN = 1.5f;

class ClassicDetector : public BaseDetector {
 public:
  /**
   * @param window_size Detector window in packets
   * @param threshold Motion probability threshold
   * @param lag Profile-displacement distance in packets
   * @param autocorr_lag Turbulence autocorrelation distance in packets
   *
   * Both lags default to the nominal-rate constants. Callers that know the
   * measured cadence pass the counts spanning L1_DELTA_LAG_US and
   * TURB_AUTOCORR_LAG_US instead, because both quantities are functions of the
   * elapsed interval rather than of how many packets fall inside it. See
   * detector_timing.h.
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
  float get_motion_metric() const override { return current_probability_; }
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

  float get_l1_delta() const { return current_l1_delta_; }
  float get_turb_autocorr() const { return current_turb_autocorr_; }
  float get_logit() const { return current_logit_; }

 private:
  uint16_t l1_delta_capacity_() const;
  float calculate_turb_autocorr_() const;
  float calculate_logit_(float l1_delta, float turb_autocorr) const;
  static float sigmoid_(float value);
  static float quantile_(const float* values, uint8_t count, float quantile);
  float startup_quantile_() const;
  float startup_l1_median_() const;

  float threshold_;
  float current_probability_;
  float current_logit_;
  float current_l1_delta_;
  float current_turb_autocorr_;
  float startup_logits_[CLASSIC_STARTUP_SAMPLE_LIMIT]{};
  float startup_l1_deltas_[CLASSIC_STARTUP_SAMPLE_LIMIT]{};
  uint8_t startup_logit_count_;
  float startup_l1_floor_;
  float l1_noise_blend_;
  float adapted_threshold_;
  bool adapted_threshold_ready_;
  uint16_t lag_;
  uint16_t autocorr_lag_;
  L1DeltaTracker l1_tracker_;
};

}  // namespace espectre
