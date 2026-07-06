/*
 * ESPectre - L1DeltaDetector Unit Tests
 *
 * Tests the L1DeltaDetector class
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_harness.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include "l1_delta_detector.h"
#include "utils.h"
#include "esphome/core/log.h"

// Include CSI data loader
#include "csi_test_data.h"

#define static_presence_packets csi_test_data::static_presence_packets()
#define motion_packets csi_test_data::motion_packets()
#define num_static_presence csi_test_data::num_static_presence()
#define num_motion csi_test_data::num_motion()

using namespace esphome::espectre;

static const char *TAG = "test_l1_delta_detector";

static const uint8_t* const TEST_SUBCARRIERS = DEFAULT_SUBCARRIERS;

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// TEST HELPERS
// ============================================================================

// Build a CSI buffer whose selected-band amplitudes follow a synthetic
// frequency-selective profile. `variant` changes the profile shape (multipath
// change), `scale` applies a per-packet scalar gain (AGC).
static void build_csi_profile(int8_t* csi_buf, int variant, int scale = 1) {
    std::memset(csi_buf, 0, 128);
    for (int i = 0; i < 12; i++) {
        int sc = TEST_SUBCARRIERS[i];
        int amplitude = (20 + 3 * ((i + 2 * variant) % 5)) * scale;
        csi_buf[sc * 2] = 0;                        // imaginary
        csi_buf[sc * 2 + 1] = (int8_t)amplitude;    // real
    }
}

// Feed `count` packets cycling through 3 distinct profiles so every lagged
// comparison crosses a profile change (L1_DELTA_LAG % 3 != 0).
static void feed_changing_profiles(L1DeltaDetector& detector, int count, int scale = 1) {
    int8_t csi_buf[128];
    for (int i = 0; i < count; i++) {
        build_csi_profile(csi_buf, i % 3, scale);
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
        detector.update_state();
    }
}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_l1_delta_detector_default_constructor(void) {
    L1DeltaDetector detector;

    TEST_ASSERT_EQUAL(DETECTOR_DEFAULT_WINDOW_SIZE, detector.get_window_size());
    TEST_ASSERT_EQUAL_FLOAT(L1_DELTA_DEFAULT_THRESHOLD, detector.get_threshold());
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
    TEST_ASSERT_EQUAL(0, detector.get_total_packets());
}

void test_l1_delta_detector_custom_constructor(void) {
    L1DeltaDetector detector(100, 0.05f);

    TEST_ASSERT_EQUAL(100, detector.get_window_size());
    TEST_ASSERT_EQUAL_FLOAT(0.05f, detector.get_threshold());
}

void test_l1_delta_detector_get_name(void) {
    L1DeltaDetector detector;

    TEST_ASSERT_EQUAL_STRING("L1D", detector.get_name());
}

void test_l1_delta_detector_startup_threshold_factor(void) {
    L1DeltaDetector detector;

    // Benchmark-tuned factor; must stay aligned with the Python runtime.
    TEST_ASSERT_EQUAL_FLOAT(1.1f, detector.get_startup_threshold_factor());
    TEST_ASSERT_EQUAL_FLOAT(L1_DELTA_STARTUP_THRESHOLD_FACTOR,
                            detector.get_startup_threshold_factor());
}

// ============================================================================
// THRESHOLD TESTS
// ============================================================================

void test_l1_delta_detector_set_threshold_valid(void) {
    L1DeltaDetector detector;

    TEST_ASSERT_TRUE(detector.set_threshold(0.08f));
    TEST_ASSERT_EQUAL_FLOAT(0.08f, detector.get_threshold());
}

void test_l1_delta_detector_set_threshold_invalid(void) {
    L1DeltaDetector detector;
    float original = detector.get_threshold();

    TEST_ASSERT_FALSE(detector.set_threshold(-0.1f));
    TEST_ASSERT_FALSE(detector.set_threshold(10.1f));
    TEST_ASSERT_EQUAL_FLOAT(original, detector.get_threshold());
}

// ============================================================================
// WARMUP / READINESS TESTS
// ============================================================================

void test_l1_delta_detector_not_ready_initially(void) {
    L1DeltaDetector detector(20, 0.05f);

    TEST_ASSERT_FALSE(detector.is_ready());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
}

void test_l1_delta_detector_ready_after_lag_plus_window(void) {
    L1DeltaDetector detector(20, 0.05f);

    // The first L1_DELTA_LAG packets have no lagged reference, then the
    // metric window must fill: ready after lag + window packets.
    feed_changing_profiles(detector, L1_DELTA_LAG + 20 - 1);
    TEST_ASSERT_FALSE(detector.is_ready());

    feed_changing_profiles(detector, 1);
    TEST_ASSERT_TRUE(detector.is_ready());
}

void test_l1_delta_detector_all_zero_csi_never_ready(void) {
    L1DeltaDetector detector(10, 0.05f);

    int8_t csi_buf[128] = {0};
    for (int i = 0; i < 100; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
        detector.update_state();
    }

    // Zero amplitudes produce no valid profile: no metric, no readiness.
    TEST_ASSERT_FALSE(detector.is_ready());
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
}

// ============================================================================
// METRIC BEHAVIOR TESTS (SYNTHETIC)
// ============================================================================

void test_l1_delta_detector_static_profile_zero_metric(void) {
    L1DeltaDetector detector(20, 0.05f);

    int8_t csi_buf[128];
    build_csi_profile(csi_buf, 0);
    for (int i = 0; i < 60; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
        detector.update_state();
    }

    TEST_ASSERT_TRUE(detector.is_ready());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
}

void test_l1_delta_detector_changing_profile_triggers_motion(void) {
    L1DeltaDetector detector(20, 0.05f);

    feed_changing_profiles(detector, 80);

    TEST_ASSERT_TRUE(detector.is_ready());
    TEST_ASSERT_TRUE(detector.get_motion_metric() > 0.05f);
    TEST_ASSERT_EQUAL(MotionState::MOTION, detector.get_state());
}

void test_l1_delta_detector_metric_gain_invariant(void) {
    L1DeltaDetector detector_1x(20, 1.0f);
    L1DeltaDetector detector_2x(20, 1.0f);

    feed_changing_profiles(detector_1x, 80, 1);
    feed_changing_profiles(detector_2x, 80, 2);

    // Per-packet scalar gain must cancel exactly in the normalized profile.
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, detector_1x.get_motion_metric(),
                             detector_2x.get_motion_metric());
    TEST_ASSERT_TRUE(detector_1x.get_motion_metric() > 0.0f);
}

// ============================================================================
// RESET / CLEAR TESTS
// ============================================================================

void test_l1_delta_detector_reset_keeps_warm_buffers(void) {
    L1DeltaDetector detector(20, 0.05f);

    feed_changing_profiles(detector, 80);
    TEST_ASSERT_TRUE(detector.is_ready());

    detector.reset();

    // Warm restart: state machine back to IDLE, metric window preserved.
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
    TEST_ASSERT_TRUE(detector.is_ready());
}

void test_l1_delta_detector_clear_buffer_cold_restart(void) {
    L1DeltaDetector detector(20, 0.05f);

    feed_changing_profiles(detector, 80);
    TEST_ASSERT_TRUE(detector.is_ready());

    detector.clear_buffer();
    detector.update_state();

    TEST_ASSERT_FALSE(detector.is_ready());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
}

// ============================================================================
// REAL DATA TESTS
// ============================================================================

void test_l1_delta_detector_real_data_separation(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }

    const int warmup = L1_DELTA_LAG + 50;
    const int sample = 200;
    if (num_static_presence < warmup + sample || num_motion < warmup + sample) {
        TEST_IGNORE_MESSAGE("Not enough packets for separation test");
        return;
    }

    L1DeltaDetector static_detector(50, 1.0f);
    float static_max = 0.0f;
    for (int i = 0; i < warmup + sample; i++) {
        static_detector.process_packet(static_presence_packets[i], 128, TEST_SUBCARRIERS, 12);
        static_detector.update_state();
        if (i >= warmup) {
            static_max = std::max(static_max, static_detector.get_motion_metric());
        }
    }

    L1DeltaDetector motion_detector(50, 1.0f);
    float motion_peak = 0.0f;
    for (int i = 0; i < warmup + sample; i++) {
        motion_detector.process_packet(motion_packets[i], 128, TEST_SUBCARRIERS, 12);
        motion_detector.update_state();
        if (i >= warmup) {
            motion_peak = std::max(motion_peak, motion_detector.get_motion_metric());
        }
    }

    ESP_LOGI(TAG, "L1-delta static max: %.4f, motion peak: %.4f", static_max, motion_peak);
    TEST_ASSERT_TRUE(static_max > 0.0f);
    TEST_ASSERT_TRUE(motion_peak > static_max);
}

void test_l1_delta_detector_motion_detects_motion(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }

    // Calibrate the threshold from static presence (max x 1.1, production rule).
    L1DeltaDetector detector(50, 1.0f);
    float static_max = 0.0f;
    int calibration_packets = std::min(num_static_presence, 400);
    for (int i = 0; i < calibration_packets; i++) {
        detector.process_packet(static_presence_packets[i], 128, TEST_SUBCARRIERS, 12);
        detector.update_state();
        if (detector.is_ready()) {
            static_max = std::max(static_max, detector.get_motion_metric());
        }
    }
    if (static_max <= 0.0f) {
        TEST_IGNORE_MESSAGE("No calibration metric available");
        return;
    }
    detector.set_threshold(static_max * detector.get_startup_threshold_factor());

    // Evaluate ~10 s of the motion capture: the first seconds of a capture can
    // be nearly still, and the static->motion transition keeps the warm metric
    // window below threshold for up to window + lag packets by construction.
    int motion_count = 0;
    int evaluated = 0;
    for (int i = 0; i < 1000 && i < num_motion; i++) {
        detector.process_packet(motion_packets[i], 128, TEST_SUBCARRIERS, 12);
        detector.update_state();
        evaluated++;
        if (detector.get_state() == MotionState::MOTION) {
            motion_count++;
        }
    }

    float detection_rate = (float)motion_count / evaluated;
    ESP_LOGI(TAG, "L1-delta motion detection rate: %.1f%%", detection_rate * 100);
    TEST_ASSERT_TRUE(detection_rate > 0.5f);  // At least 50% detection
}

// ============================================================================
// TEST RUNNER
// ============================================================================

int process(void) {
    UNITY_BEGIN();

    // Initialization tests
    RUN_TEST(test_l1_delta_detector_default_constructor);
    RUN_TEST(test_l1_delta_detector_custom_constructor);
    RUN_TEST(test_l1_delta_detector_get_name);
    RUN_TEST(test_l1_delta_detector_startup_threshold_factor);

    // Threshold tests
    RUN_TEST(test_l1_delta_detector_set_threshold_valid);
    RUN_TEST(test_l1_delta_detector_set_threshold_invalid);

    // Warmup / readiness tests
    RUN_TEST(test_l1_delta_detector_not_ready_initially);
    RUN_TEST(test_l1_delta_detector_ready_after_lag_plus_window);
    RUN_TEST(test_l1_delta_detector_all_zero_csi_never_ready);

    // Metric behavior tests
    RUN_TEST(test_l1_delta_detector_static_profile_zero_metric);
    RUN_TEST(test_l1_delta_detector_changing_profile_triggers_motion);
    RUN_TEST(test_l1_delta_detector_metric_gain_invariant);

    // Reset / clear tests
    RUN_TEST(test_l1_delta_detector_reset_keeps_warm_buffers);
    RUN_TEST(test_l1_delta_detector_clear_buffer_cold_restart);

    // Real data tests
    RUN_TEST(test_l1_delta_detector_real_data_separation);
    RUN_TEST(test_l1_delta_detector_motion_detects_motion);

    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
