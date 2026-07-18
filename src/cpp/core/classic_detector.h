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

constexpr float CLASSIC_DEFAULT_THRESHOLD = 0.6066111851930618f;
constexpr float CLASSIC_MIN_THRESHOLD = 0.0f;
constexpr float CLASSIC_MAX_THRESHOLD = 1.0f;
constexpr float CLASSIC_STARTUP_THRESHOLD_FACTOR = 1.0f;

constexpr float CLASSIC_L1_CENTER = 0.03669842332601547f;
constexpr float CLASSIC_L1_SCALE = 0.026984458789229393f;
constexpr float CLASSIC_L1_WEIGHT = 5.572897434234619f;
constexpr float CLASSIC_AUTOCORR_CENTER = 0.27886947989463806f;
constexpr float CLASSIC_AUTOCORR_SCALE = 0.33479437232017517f;
constexpr float CLASSIC_AUTOCORR_WEIGHT = 3.1952695846557617f;
constexpr float CLASSIC_INTERCEPT = -2.1254162788391113f;

constexpr float CLASSIC_TRAIN_IDLE_Q95_LOGIT = -0.6372601389884949f;
constexpr float CLASSIC_STARTUP_QUANTILE = 0.95f;
constexpr float CLASSIC_STARTUP_STRENGTH = 0.3f;
constexpr uint8_t CLASSIC_STARTUP_SAMPLE_LIMIT = 64U;

class ClassicDetector : public BaseDetector {
 public:
  ClassicDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE,
                  float threshold = CLASSIC_DEFAULT_THRESHOLD);

  ~ClassicDetector() override = default;
  ClassicDetector(ClassicDetector&& other) noexcept = default;
  ClassicDetector& operator=(ClassicDetector&& other) noexcept = default;
  ClassicDetector(const ClassicDetector&) = delete;
  ClassicDetector& operator=(const ClassicDetector&) = delete;

  void process_packet(const int8_t* csi_data, size_t csi_len,
                      const uint8_t* selected_subcarriers = nullptr,
                      uint8_t num_subcarriers = 0) override;
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
  float startup_quantile_() const;

  float threshold_;
  float current_probability_;
  float current_logit_;
  float current_l1_delta_;
  float current_turb_autocorr_;
  float startup_logits_[CLASSIC_STARTUP_SAMPLE_LIMIT]{};
  uint8_t startup_logit_count_;
  float adapted_threshold_;
  bool adapted_threshold_ready_;
  L1DeltaTracker l1_tracker_;
};

}  // namespace espectre
