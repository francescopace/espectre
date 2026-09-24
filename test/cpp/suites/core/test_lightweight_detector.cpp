/*
 * ESPectre - LightweightDetector Unit Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#define private public
#define protected public
#include "lightweight_detector.h"
#undef protected
#undef private

#include <cmath>

#include "detector_state_contract.h"

using namespace espectre;

void setUp(void) {}
void tearDown(void) {}

void test_lightweight_detector_uses_probability_scale(void) {
  LightweightDetector detector;
  TEST_ASSERT_EQUAL_STRING("Lightweight", detector.get_name());
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, LIGHTWEIGHT_DEFAULT_THRESHOLD, detector.get_threshold());
  TEST_ASSERT_TRUE(detector.set_threshold(0.75f));
  TEST_ASSERT_FALSE(detector.set_threshold(1.01f));
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.75f, detector.get_threshold());
}

void test_lightweight_detector_logit_matches_exported_linear_fusion(void) {
  LightweightDetector detector;
  const float logit = detector.calculate_logit_(LIGHTWEIGHT_AUTOCORR_CENTER,
      LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_CENTER);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, LIGHTWEIGHT_INTERCEPT, logit);
  TEST_ASSERT_FLOAT_WITHIN(
      1e-6f,
      LIGHTWEIGHT_INTERCEPT + LIGHTWEIGHT_AUTOCORR_WEIGHT +
          LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_WEIGHT,
      detector.calculate_logit_(LIGHTWEIGHT_AUTOCORR_CENTER + LIGHTWEIGHT_AUTOCORR_SCALE,
          LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_CENTER +
              LIGHTWEIGHT_TURB_IQR_OVER_MEAN_AGGR_SCALE));
}

void test_lightweight_detector_hampel_master_switch_controls_turbulence(void) {
  LightweightDetector detector;
  detector.configure_hampel(true, 5U, 3.0f);
  TEST_ASSERT_TRUE(detector.hampel_state_.enabled);
  TEST_ASSERT_TRUE(detector.aggregated_turbulence_.hampel_enabled());

  detector.configure_hampel(false, 5U, 3.0f);
  TEST_ASSERT_FALSE(detector.hampel_state_.enabled);
  TEST_ASSERT_FALSE(detector.aggregated_turbulence_.hampel_enabled());
}

void test_lightweight_detector_owns_aggregated_turbulence_ring(void) {
  LightweightDetector detector;
  TEST_ASSERT_TRUE(detector.is_valid());
  TEST_ASSERT_NOT_NULL(detector.aggregated_turbulence_buffer_.get());
  TEST_ASSERT_EQUAL(detector.get_window_size(), detector.aggregated_turbulence_.capacity());
  TEST_ASSERT_EQUAL(0, detector.aggregated_turbulence_.count());
}

void test_filtered_turbulence_ring_skips_large_missing_runs(void) {
  float storage[4];
  float scratch[4];
  FilteredTurbulenceRing ring;
  ring.bind(storage, 4U);
  ring.add(1.0f);
  ring.add(2.0f);

  ring.advance_missing_slots(9U);
  TEST_ASSERT_EQUAL(4, ring.count());
  TEST_ASSERT_EQUAL(0, ring.valid_count());

  ring.add(3.0f);
  uint16_t count = 0U;
  const float *ordered = ring.ordered_view(scratch, 4U, count);
  TEST_ASSERT_NOT_NULL(ordered);
  TEST_ASSERT_EQUAL(4, count);
  TEST_ASSERT_TRUE(std::isnan(ordered[0]));
  TEST_ASSERT_TRUE(std::isnan(ordered[1]));
  TEST_ASSERT_TRUE(std::isnan(ordered[2]));
  TEST_ASSERT_EQUAL_FLOAT(3.0f, ordered[3]);
}

namespace {

float logit(float probability) { return std::log(probability / (1.0f - probability)); }

float adapted_from_constant_evidence(uint8_t samples, float value) {
  LightweightDetector detector;
  detector.startup_logit_count_ = samples;
  for (uint8_t i = 0U; i < samples; i++) {
    detector.startup_logits_[i] = value;
  }
  detector.on_startup_calibration_complete();
  return detector.adapted_threshold_;
}

}  // namespace

void test_lightweight_detector_startup_q95_adapts_threshold(void) {
  LightweightDetector detector;
  detector.startup_logit_count_ = LIGHTWEIGHT_STARTUP_MIN_SAMPLES;
  for (uint8_t i = 0U; i < detector.startup_logit_count_; i++) {
    detector.startup_logits_[i] = -1.0f + 0.02f * static_cast<float>(i);
  }
  const float q95 = detector.startup_quantile_();

  detector.on_startup_calibration_complete();
  TEST_ASSERT_TRUE(detector.adapted_threshold_ready_);
  TEST_ASSERT_FLOAT_WITHIN(
      1e-5f,
      logit(LIGHTWEIGHT_DEFAULT_THRESHOLD) +
          LIGHTWEIGHT_STARTUP_STRENGTH * (q95 - LIGHTWEIGHT_TRAIN_IDLE_Q95_LOGIT),
      logit(detector.adapted_threshold_));
  TEST_ASSERT_TRUE(detector.set_adaptive_threshold(0.1f));
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, detector.adapted_threshold_, detector.get_threshold());
}

void test_lightweight_detector_startup_evidence_is_bounded(void) {
  // Motion evidence counts at the calibration motion logit, so the adapted
  // threshold stays below the ceiling the calibration guard restarts on.
  const float motion = adapted_from_constant_evidence(40U, 10.0f);
  TEST_ASSERT_FLOAT_WITHIN(
      1e-5f,
      logit(LIGHTWEIGHT_DEFAULT_THRESHOLD) +
          LIGHTWEIGHT_STARTUP_STRENGTH *
              (LIGHTWEIGHT_CALIBRATION_MOTION_LOGIT - LIGHTWEIGHT_TRAIN_IDLE_Q95_LOGIT),
      logit(motion));
  TEST_ASSERT_TRUE(motion < LightweightDetector().calibration_motion_ceiling());
  // Too few samples keep the default rather than a q95 of one or two values.
  TEST_ASSERT_FLOAT_WITHIN(
      1e-6f, LIGHTWEIGHT_DEFAULT_THRESHOLD,
      adapted_from_constant_evidence(LIGHTWEIGHT_STARTUP_MIN_SAMPLES - 1U, 10.0f));
  // A very quiet session never adapts below the training idle q95.
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, LIGHTWEIGHT_TRAIN_IDLE_Q95_LOGIT,
                           logit(adapted_from_constant_evidence(40U, -20.0f)));
}

namespace {

// A quiet window of `count` evaluations with an optional run of `burst`
// evaluations at `burst_logit`, starting at `burst_start`.
void load_startup_evidence(LightweightDetector& detector, uint8_t count, uint8_t burst_start,
                           uint8_t burst, float burst_logit) {
  detector.startup_logit_count_ = count;
  for (uint8_t i = 0U; i < count; i++) {
    const bool in_burst = i >= burst_start && i < burst_start + burst;
    detector.startup_logits_[i] = in_burst ? burst_logit : -6.0f + 0.01f * static_cast<float>(i % 7U);
  }
}

}  // namespace

void test_lightweight_detector_steps_calibration_past_a_burst(void) {
  LightweightDetector detector;
  // A clean base budget concludes at once.
  load_startup_evidence(detector, LIGHTWEIGHT_STARTUP_BASE_SAMPLES, 0U, 0U, 0.0f);
  TEST_ASSERT_TRUE(detector.startup_calibration_conclusive());
  // A 3 s burst inside it asks for more evidence...
  load_startup_evidence(detector, LIGHTWEIGHT_STARTUP_BASE_SAMPLES, 15U, 12U, 3.0f);
  TEST_ASSERT_FALSE(detector.startup_calibration_conclusive());
  // ...until the rest spans a clean base budget, and the burst never sets
  // the threshold.
  load_startup_evidence(detector, LIGHTWEIGHT_STARTUP_BASE_SAMPLES + 20U, 15U, 12U, 3.0f);
  TEST_ASSERT_TRUE(detector.startup_calibration_conclusive());
  detector.on_startup_calibration_complete();
  const float with_burst = detector.adapted_threshold_;
  load_startup_evidence(detector, LIGHTWEIGHT_STARTUP_BASE_SAMPLES + 20U, 0U, 0U, 0.0f);
  detector.on_startup_calibration_complete();
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, detector.adapted_threshold_, with_burst);
}

void test_lightweight_detector_keeps_recurring_noise_in_its_threshold(void) {
  // A noisy link repeats its episodes, so removing one run leaves the others
  // and the threshold stays high rather than trusting the quiet stretches.
  LightweightDetector detector;
  detector.startup_logit_count_ = LIGHTWEIGHT_STARTUP_SAMPLE_LIMIT;
  for (uint8_t i = 0U; i < detector.startup_logit_count_; i++) {
    detector.startup_logits_[i] = (i % 20U) < 3U ? 3.0f : -6.0f;
  }
  TEST_ASSERT_TRUE(detector.startup_calibration_conclusive());
  detector.on_startup_calibration_complete();
  TEST_ASSERT_TRUE(detector.adapted_threshold_ > LIGHTWEIGHT_NOISY_LINK_THRESHOLD);
}

void test_lightweight_detector_exposes_its_calibration_motion_ceiling(void) {
  TEST_ASSERT_FLOAT_WITHIN(1e-3f, LIGHTWEIGHT_CALIBRATION_MOTION_LOGIT,
                           logit(LightweightDetector().calibration_motion_ceiling()));
}

void test_manual_threshold_suspends_settling_until_recalibration(void) {
  LightweightDetector detector;
  detector.on_startup_calibration_complete();
  detector.set_adaptive_threshold(0.5f);
  TEST_ASSERT_TRUE(detector.set_threshold(0.9f));
  detector.reset();
  detector.clear_buffer();
  int8_t packet[HT20_CSI_LEN];
  std::fill(packet, packet + HT20_CSI_LEN, 20);
  const unsigned packets = detector.get_window_size() +
      LIGHTWEIGHT_SETTLE_BLOCKS * LIGHTWEIGHT_SETTLE_BLOCK_EVALUATIONS;
  for (unsigned i = 0U; i < packets; ++i) {
    detector.process_packet(packet, sizeof(packet), nullptr, 0U);
    detector.update_state();
  }
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.9f, detector.get_threshold());

  detector.on_startup_calibration_begin();
  detector.on_startup_calibration_complete();
  detector.set_adaptive_threshold(0.5f);
  const float calibrated = detector.get_threshold();
  TEST_ASSERT_FALSE(detector.set_threshold(1.01f));
  for (unsigned i = 0U; i < packets; ++i) {
    detector.process_packet(packet, sizeof(packet), nullptr, 0U);
    detector.update_state();
  }
  TEST_ASSERT_TRUE(detector.get_threshold() < calibrated);
}

void test_abandoned_calibration_keeps_the_threshold_and_its_adaptation(void) {
  int8_t packet[HT20_CSI_LEN];
  std::fill(packet, packet + HT20_CSI_LEN, 20);
  LightweightDetector detector;
  const unsigned packets = detector.get_window_size() +
      LIGHTWEIGHT_SETTLE_BLOCKS * LIGHTWEIGHT_SETTLE_BLOCK_EVALUATIONS;
  auto run_quiet = [&]() {
    for (unsigned i = 0U; i < packets; ++i) {
      detector.process_packet(packet, sizeof(packet), nullptr, 0U);
      detector.update_state();
    }
  };
  detector.on_startup_calibration_complete();
  detector.set_adaptive_threshold(0.5f);
  const float calibrated = detector.get_threshold();

  // Settling waits while a calibration collects evidence...
  detector.on_startup_calibration_begin();
  run_quiet();
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, calibrated, detector.get_threshold());
  // ...and resumes once that calibration ends without a result.
  detector.on_startup_calibration_abandoned();
  TEST_ASSERT_EQUAL(0, detector.startup_logit_count_);
  run_quiet();
  TEST_ASSERT_TRUE(detector.get_threshold() < calibrated);

  // A manual threshold survives an abandoned calibration.
  TEST_ASSERT_TRUE(detector.set_threshold(0.9f));
  detector.on_startup_calibration_begin();
  detector.on_startup_calibration_abandoned();
  run_quiet();
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.9f, detector.get_threshold());
}

void test_lightweight_detector_uncalibrated_abandon_lets_the_default_settle(void) {
  int8_t packet[HT20_CSI_LEN];
  std::fill(packet, packet + HT20_CSI_LEN, 20);
  LightweightDetector detector;
  const unsigned packets = detector.get_window_size() +
      LIGHTWEIGHT_SETTLE_BLOCKS * LIGHTWEIGHT_SETTLE_BLOCK_EVALUATIONS;

  // A startup calibration rejected for motion keeps the threshold in force...
  detector.on_startup_calibration_begin();
  detector.on_startup_calibration_abandoned();
  TEST_ASSERT_TRUE(detector.adapted_threshold_ready_);
  TEST_ASSERT_FLOAT_WITHIN(1e-6f, LIGHTWEIGHT_DEFAULT_THRESHOLD, detector.get_threshold());
  // ...and a quiet room still lowers it, never under the training idle q95.
  for (unsigned i = 0U; i < packets; ++i) {
    detector.process_packet(packet, sizeof(packet), nullptr, 0U);
    detector.update_state();
  }
  TEST_ASSERT_TRUE(detector.get_threshold() < LIGHTWEIGHT_DEFAULT_THRESHOLD);
  TEST_ASSERT_FLOAT_WITHIN(1e-5f, LIGHTWEIGHT_TRAIN_IDLE_Q95_LOGIT,
                           logit(detector.get_threshold()));
}

void test_lightweight_detector_clear_buffer_resets_feature_state(void) {
  LightweightDetector detector(10U);
  detector.current_metric_ = 0.9f;
  detector.current_turb_autocorr_ = 0.2f;
  detector.current_turb_iqr_over_mean_aggr_ = 0.5f;
  detector.startup_logit_count_ = 3U;

  detector.clear_buffer();

  TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
  TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_turb_autocorr());
  TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_turb_iqr_over_mean_aggr());
  TEST_ASSERT_EQUAL(3, detector.startup_logit_count_);
  TEST_ASSERT_TRUE(detector.get_state() == MotionState::IDLE);

  detector.on_startup_calibration_begin();
  TEST_ASSERT_EQUAL(0, detector.startup_logit_count_);
  TEST_ASSERT_FALSE(detector.adapted_threshold_ready_);
}

// Shared across every detector; see detector_state_contract.h for why the
// metric is set directly rather than driven through synthetic traffic.
void test_lightweight_detector_honours_shared_state_contract(void) {
  LightweightDetector clear_target;
  test_support::assert_clear_buffer_drops_evaluation_state(clear_target);

  LightweightDetector reset_target;
  test_support::assert_reset_drops_evaluation_state(reset_target);

  LightweightDetector idle_target;
  test_support::assert_not_ready_evaluation_stays_idle(idle_target);
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_lightweight_detector_uses_probability_scale);
  RUN_TEST(test_lightweight_detector_logit_matches_exported_linear_fusion);
  RUN_TEST(test_lightweight_detector_hampel_master_switch_controls_turbulence);
  RUN_TEST(test_lightweight_detector_owns_aggregated_turbulence_ring);
  RUN_TEST(test_filtered_turbulence_ring_skips_large_missing_runs);
  RUN_TEST(test_lightweight_detector_startup_q95_adapts_threshold);
  RUN_TEST(test_lightweight_detector_startup_evidence_is_bounded);
  RUN_TEST(test_lightweight_detector_exposes_its_calibration_motion_ceiling);
  RUN_TEST(test_lightweight_detector_steps_calibration_past_a_burst);
  RUN_TEST(test_lightweight_detector_keeps_recurring_noise_in_its_threshold);
  RUN_TEST(test_manual_threshold_suspends_settling_until_recalibration);
  RUN_TEST(test_abandoned_calibration_keeps_the_threshold_and_its_adaptation);
  RUN_TEST(test_lightweight_detector_uncalibrated_abandon_lets_the_default_settle);
  RUN_TEST(test_lightweight_detector_clear_buffer_resets_feature_state);
  RUN_TEST(test_lightweight_detector_honours_shared_state_contract);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char** argv) { return process(); }
#endif
