/*
 * ESPectre - ClassicDetector Unit Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#define private public
#define protected public
#include "classic_detector.h"
#undef protected
#undef private

#include <cmath>

#include "detector_state_contract.h"

using namespace espectre;

void setUp(void) {}
void tearDown(void) {}

void test_classic_detector_uses_probability_scale(void) {
  ClassicDetector detector;
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, CLASSIC_DEFAULT_THRESHOLD, detector.get_threshold());
  TEST_ASSERT_TRUE(detector.set_threshold(0.75f));
  TEST_ASSERT_FALSE(detector.set_threshold(1.01f));
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.75f, detector.get_threshold());
}

void test_classic_detector_logit_matches_exported_linear_fusion(void) {
  ClassicDetector detector;
  const float logit = detector.calculate_logit_(CLASSIC_AUTOCORR_CENTER,
                                                 CLASSIC_FREQ_COH_CURVE_STD_CENTER);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, CLASSIC_INTERCEPT, logit);
  TEST_ASSERT_FLOAT_WITHIN(
      1e-6f,
      CLASSIC_INTERCEPT + CLASSIC_AUTOCORR_WEIGHT +
          CLASSIC_FREQ_COH_CURVE_STD_WEIGHT,
      detector.calculate_logit_(CLASSIC_AUTOCORR_CENTER + CLASSIC_AUTOCORR_SCALE,
                                CLASSIC_FREQ_COH_CURVE_STD_CENTER +
                                    CLASSIC_FREQ_COH_CURVE_STD_SCALE));
}

void test_classic_detector_hampel_master_switch_controls_turbulence(void) {
  ClassicDetector detector;
  detector.configure_hampel(true, 5U, 3.0f);
  TEST_ASSERT_TRUE(detector.hampel_state_.enabled);

  detector.configure_hampel(false, 5U, 3.0f);
  TEST_ASSERT_FALSE(detector.hampel_state_.enabled);
}

void test_classic_detector_tracks_only_frequency_curve_shape_state(void) {
  ClassicDetector detector;
  TEST_ASSERT_TRUE(detector.shape_tracker_.tracks_frequency());
  TEST_ASSERT_FALSE(detector.shape_tracker_.tracks_shape());
  TEST_ASSERT_TRUE(detector.shape_tracker_.ring_.empty());
  TEST_ASSERT_TRUE(detector.shape_tracker_.motion_energy_ring_.empty());
}

void test_classic_detector_startup_q95_adapts_threshold(void) {
  ClassicDetector detector;
  detector.startup_logit_count_ = 4U;
  detector.startup_logits_[0] = -1.0f;
  detector.startup_logits_[1] = -0.8f;
  detector.startup_logits_[2] = -0.6f;
  detector.startup_logits_[3] = -0.4f;

  detector.on_startup_calibration_complete();
  TEST_ASSERT_TRUE(detector.adapted_threshold_ready_);
  TEST_ASSERT_TRUE(detector.set_adaptive_threshold(0.1f));
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, detector.adapted_threshold_, detector.get_threshold());
  TEST_ASSERT_TRUE(detector.get_threshold() > 0.0f);
  TEST_ASSERT_TRUE(detector.get_threshold() < 1.0f);
}

void test_classic_detector_noisy_startup_still_uses_shifted_logit_threshold(void) {
  ClassicDetector detector;
  detector.startup_logit_count_ = 4U;
  for (uint8_t i = 0U; i < detector.startup_logit_count_; i++) {
    detector.startup_logits_[i] = 10.0f;
  }

  detector.on_startup_calibration_complete();
  TEST_ASSERT_TRUE(detector.adapted_threshold_ > CLASSIC_DEFAULT_THRESHOLD);
  TEST_ASSERT_TRUE(detector.adapted_threshold_ < 1.0f);
}

void test_classic_detector_clear_buffer_resets_feature_state(void) {
  ClassicDetector detector(10U);
  detector.current_metric_ = 0.9f;
  detector.current_turb_autocorr_ = 0.2f;
  detector.current_chan_freq_coh_curve_std_ = 0.5f;
  detector.startup_logit_count_ = 3U;

  detector.clear_buffer();

  TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
  TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_turb_autocorr());
  TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_chan_freq_coh_curve_std());
  TEST_ASSERT_EQUAL(3, detector.startup_logit_count_);
  TEST_ASSERT_TRUE(detector.get_state() == MotionState::IDLE);

  detector.on_startup_calibration_begin();
  TEST_ASSERT_EQUAL(0, detector.startup_logit_count_);
  TEST_ASSERT_FALSE(detector.adapted_threshold_ready_);
}

// Shared across every detector; see detector_state_contract.h for why the
// metric is set directly rather than driven through synthetic traffic.
void test_classic_detector_honours_shared_state_contract(void) {
  ClassicDetector clear_target;
  test_support::assert_clear_buffer_drops_evaluation_state(clear_target);

  ClassicDetector reset_target;
  test_support::assert_reset_drops_evaluation_state(reset_target);

  ClassicDetector idle_target;
  test_support::assert_not_ready_evaluation_stays_idle(idle_target);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_classic_detector_uses_probability_scale);
  RUN_TEST(test_classic_detector_logit_matches_exported_linear_fusion);
  RUN_TEST(test_classic_detector_hampel_master_switch_controls_turbulence);
  RUN_TEST(test_classic_detector_tracks_only_frequency_curve_shape_state);
  RUN_TEST(test_classic_detector_startup_q95_adapts_threshold);
  RUN_TEST(test_classic_detector_noisy_startup_still_uses_shifted_logit_threshold);
  RUN_TEST(test_classic_detector_clear_buffer_resets_feature_state);
  RUN_TEST(test_classic_detector_honours_shared_state_contract);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char** argv) { return process(); }
#endif
