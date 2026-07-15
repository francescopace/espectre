/*
 * ESPectre - ClassicDetector Unit Tests
 *
 * Validates the classic fusion-specific state transitions that do not exist in
 * the standalone legacy detectors.
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

using namespace espectre;

void setUp(void) {}
void tearDown(void) {}

void test_classic_detector_clear_buffer_preserves_frozen_floor(void) {
    ClassicDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, CLASSIC_DEFAULT_THRESHOLD);

    detector.apply_startup_floor(1.0f, true, 400);
    detector.on_startup_calibration_complete();
    TEST_ASSERT_TRUE(detector.floor_frozen_);
    TEST_ASSERT_TRUE(detector.recovery_vote_enabled_);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, detector.variance_floor_);

    detector.clear_buffer();

    TEST_ASSERT_TRUE(detector.floor_frozen_);
    TEST_ASSERT_TRUE(detector.recovery_vote_enabled_);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 1.0f, detector.variance_floor_);
    TEST_ASSERT_TRUE(detector.get_state() == MotionState::IDLE);
}

void test_classic_detector_uses_recovery_vote_in_ambiguous_band(void) {
    ClassicDetector detector(10, 1.0f);

    detector.l1_tracker_.delta_count_ = detector.window_size_;
    detector.l1_tracker_.delta_sum_ = 0.0f;
    for (uint16_t i = 0; i < detector.window_size_; i++) {
        detector.l1_tracker_.delta_ring_[i] = 0.7f;  // ambiguous band: 0.6 < l1 <= 1.0
        detector.l1_tracker_.delta_sum_ += 0.7f;
        detector.turbulence_buffer_[i] = (i % 2 == 0) ? 0.0f : 3.0f;
    }
    detector.buffer_count_ = detector.window_size_;
    detector.recovery_vote_enabled_ = true;
    detector.floor_frozen_ = true;
    detector.floor_count_ = CLASSIC_VARIANCE_FLOOR_MIN;
    detector.variance_floor_ = 0.5f;

    detector.update_state();

    TEST_ASSERT_TRUE(detector.get_state() == MotionState::MOTION);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.7f, detector.get_motion_metric());
    TEST_ASSERT_TRUE(detector.get_last_moving_variance() > CLASSIC_RECOVERY_VOTE_RATIO * detector.get_variance_floor());
}

void test_classic_detector_can_disable_recovery_vote(void) {
    ClassicDetector detector(10, 1.0f, false);
    int8_t csi[128] = {};
    const uint8_t subcarrier = 1;

    detector.process_packet(csi, sizeof(csi), &subcarrier, 1);
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
    TEST_ASSERT_EQUAL(0, detector.get_buffer_count());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_startup_floor_metric());

    detector.apply_startup_floor(0.5f, true, CLASSIC_VARIANCE_FLOOR_MIN);
    TEST_ASSERT_EQUAL(0, detector.floor_count_);
    TEST_ASSERT_FALSE(detector.recovery_vote_enabled());

    detector.l1_tracker_.delta_count_ = detector.window_size_;
    detector.l1_tracker_.delta_sum_ = 0.0f;
    for (uint16_t i = 0; i < detector.window_size_; i++) {
        detector.l1_tracker_.delta_ring_[i] = 0.7f;
        detector.l1_tracker_.delta_sum_ += 0.7f;
        detector.turbulence_buffer_[i] = (i % 2 == 0) ? 0.0f : 3.0f;
    }
    detector.buffer_count_ = detector.window_size_;
    detector.floor_count_ = CLASSIC_VARIANCE_FLOOR_MIN;
    detector.variance_floor_ = 0.5f;
    detector.recovery_vote_enabled_ = true;

    detector.update_state();

    TEST_ASSERT_FALSE(detector.recovery_vote_configured());
    TEST_ASSERT_TRUE(detector.get_state() == MotionState::IDLE);
}

void test_classic_detector_collects_recovery_samples_only_in_ambiguous_band(void) {
    ClassicDetector detector(10, 1.0f);
    detector.floor_frozen_ = true;
    detector.recovery_vote_enabled_ = true;
    detector.floor_count_ = CLASSIC_VARIANCE_FLOOR_MIN;

    detector.current_l1_metric_ = 0.5f;
    TEST_ASSERT_FALSE(detector.should_collect_recovery_sample_());

    detector.current_l1_metric_ = 0.8f;
    TEST_ASSERT_TRUE(detector.should_collect_recovery_sample_());

    detector.current_l1_metric_ = 1.1f;
    TEST_ASSERT_FALSE(detector.should_collect_recovery_sample_());

    detector.floor_frozen_ = false;
    TEST_ASSERT_TRUE(detector.should_collect_recovery_sample_());
}

void test_classic_detector_skips_turbulence_path_outside_recovery_band(void) {
    ClassicDetector detector(10, 1.0f);
    const int8_t csi[128] = {};
    const uint8_t subcarrier = 1;
    detector.floor_frozen_ = true;
    detector.recovery_vote_enabled_ = true;
    detector.floor_count_ = CLASSIC_VARIANCE_FLOOR_MIN;

    detector.current_l1_metric_ = 0.5f;
    detector.process_packet(csi, sizeof(csi), &subcarrier, 1);
    TEST_ASSERT_EQUAL(0, detector.buffer_count_);

    detector.current_l1_metric_ = 0.8f;
    detector.process_packet(csi, sizeof(csi), &subcarrier, 1);
    TEST_ASSERT_EQUAL(1, detector.buffer_count_);
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_classic_detector_clear_buffer_preserves_frozen_floor);
    RUN_TEST(test_classic_detector_uses_recovery_vote_in_ambiguous_band);
    RUN_TEST(test_classic_detector_can_disable_recovery_vote);
    RUN_TEST(test_classic_detector_collects_recovery_samples_only_in_ambiguous_band);
    RUN_TEST(test_classic_detector_skips_turbulence_path_outside_recovery_band);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
