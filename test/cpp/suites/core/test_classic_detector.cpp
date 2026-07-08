/*
 * ESPectre - ClassicDetector Unit Tests
 *
 * Validates the classic fusion-specific state transitions that do not exist in
 * the standalone legacy detectors.
 */

#include "test_harness.h"

#define private public
#define protected public
#include "classic_detector.h"
#undef protected
#undef private

using namespace esphome::espectre;

void setUp(void) {}
void tearDown(void) {}

void test_classic_detector_clear_buffer_preserves_frozen_floor(void) {
    ClassicDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, CLASSIC_DEFAULT_THRESHOLD);

    for (uint16_t i = 0; i < 400; i++) {
        detector.variance_floor_ring_[i] = 1.0f;
    }
    detector.floor_count_ = 400;
    detector.floor_idx_ = 400;

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

    detector.delta_count_ = detector.window_size_;
    for (uint16_t i = 0; i < detector.window_size_; i++) {
        detector.delta_ring_[i] = 0.7f;  // ambiguous band: 0.6 < l1 <= 1.0
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

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_classic_detector_clear_buffer_preserves_frozen_floor);
    RUN_TEST(test_classic_detector_uses_recovery_vote_in_ambiguous_band);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
