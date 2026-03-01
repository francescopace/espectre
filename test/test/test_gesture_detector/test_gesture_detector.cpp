/*
 * ESPectre - GestureDetector Unit Tests
 *
 * Tests the gesture pipeline:
 *   - gesture_features.h: event morphology feature extraction
 *   - gesture_detector.h/.cpp: ring buffer, event lifecycle, inference routing
 *
 * These tests run on the native (PC) platform via PlatformIO.
 * No gesture_weights.h is required — inference is tested as "not available"
 * until a model is trained. Feature extraction is fully tested with synthetic data.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <vector>
#include "gesture_features.h"
#include "gesture_detector.h"
#include "esphome/core/log.h"

using namespace esphome::espectre;

static const char *TAG = "test_gesture_detector";

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// HELPER: synthetic CSI data (zeros → no turbulence)
// ============================================================================

static constexpr size_t CSI_LEN = 128;  // 64 subcarriers * 2 bytes (I+Q)

static void make_csi(int8_t* buf, float amplitude = 10.0f, uint8_t phase_shift = 0) {
    memset(buf, 0, CSI_LEN);
    for (int i = 0; i < 64; i++) {
        // Simple synthetic signal: I = amplitude * cos(phase_shift), Q = amplitude * sin(phase_shift)
        buf[i * 2]     = static_cast<int8_t>(amplitude);
        buf[i * 2 + 1] = static_cast<int8_t>(phase_shift % 32);
    }
}

// ============================================================================
// GESTURE FEATURE TESTS
// ============================================================================

void test_gesture_features_empty_buffer(void) {
    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(nullptr, nullptr, 0, 12, features);

    for (int i = 0; i < GESTURE_NUM_FEATURES; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, features[i]);
    }
}

void test_gesture_features_constant_signal(void) {
    // Constant turbulence → no peaks, no rise/fall variation
    constexpr uint16_t N = 100;
    float turb[N];
    float phases[N * 12];
    for (int i = 0; i < N; i++) {
        turb[i] = 5.0f;
        for (int j = 0; j < 12; j++) {
            phases[i * 12 + j] = 0.0f;
        }
    }

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // event_duration is log-compressed and quantized (N=100 -> 0.9)
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.9f, features[0]);

    // peak_to_mean_ratio = 1.0 for constant signal
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, features[2]);

    // n_local_peaks = 0 for constant signal
    TEST_ASSERT_EQUAL_FLOAT(0.0f, features[5]);

    // FWHM = 0 or 1 for constant signal (degenerate case)
    TEST_ASSERT_TRUE(features[6] >= 0.0f);
    TEST_ASSERT_TRUE(features[6] <= 1.0f);
}

void test_gesture_features_single_peak(void) {
    // Single sharp triangular spike at midpoint
    // (Gaussian has too smooth a transition and peak prominence < 10% threshold)
    constexpr uint16_t N = 100;
    float turb[N];
    float phases[N * 12];
    memset(phases, 0, sizeof(phases));

    int mid = N / 2;
    for (int i = 0; i < N; i++) {
        turb[i] = 1.0f;  // Baseline
    }
    // Create a sharp triangular spike: 3 points only
    turb[mid - 1] = 3.0f;
    turb[mid]     = 10.0f;  // Peak (prominence = 10 - 3 = 7, threshold = 0.1 * 9 = 0.9)
    turb[mid + 1] = 3.0f;

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // Peak should be near center (0.4..0.6)
    TEST_ASSERT_TRUE(features[1] >= 0.3f);  // peak_position
    TEST_ASSERT_TRUE(features[1] <= 0.7f);

    // Peak-to-mean ratio should be > 1 for a spike
    TEST_ASSERT_TRUE(features[2] > 1.0f);

    // Exactly 1 local peak → n_local_peaks = 1/10 = 0.1
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.1f, features[5]);

    // FWHM should be reasonable (0 < FWHM < 1)
    TEST_ASSERT_TRUE(features[6] > 0.0f);
    TEST_ASSERT_TRUE(features[6] < 1.0f);
}

void test_gesture_features_pre_post_energy_symmetric(void) {
    // Symmetric signal: pre/post ratio should be ~1.0
    constexpr uint16_t N = 100;
    float turb[N];
    float phases[N * 12];
    memset(phases, 0, sizeof(phases));

    for (int i = 0; i < N; i++) {
        // Simple triangle symmetric around center
        float x = std::abs(i - N / 2.0f);
        turb[i] = 1.0f + (N / 2.0f - x) / (N / 2.0f);
    }

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // pre_post_energy_ratio should be close to 1.0 for symmetric signal
    TEST_ASSERT_FLOAT_WITHIN(0.2f, 1.0f, features[4]);
}

void test_gesture_features_output_range(void) {
    // All features should be within known bounds
    constexpr uint16_t N = 200;
    float turb[N];
    float phases[N * 12];

    for (int i = 0; i < N; i++) {
        turb[i] = 1.0f + (i % 10) * 0.5f;
        for (int j = 0; j < 12; j++) {
            phases[i * 12 + j] = (j - 6) * 0.5f;
        }
    }

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    TEST_ASSERT_TRUE(features[0] >= 0.0f);    // event_duration >= 0
    TEST_ASSERT_TRUE(features[1] >= 0.0f && features[1] <= 1.0f);  // peak_position [0,1]
    TEST_ASSERT_TRUE(features[2] >= 0.0f && features[2] <= 10.0f); // peak_to_mean [0,10]
    TEST_ASSERT_TRUE(features[3] >= -1.0f && features[3] <= 1.0f);  // rise_fall_asymmetry [-1,1]
    TEST_ASSERT_TRUE(features[4] >= 0.1f && features[4] <= 10.0f);  // pre_post ratio [0.1,10]
    TEST_ASSERT_TRUE(features[5] >= 0.0f);                           // n_local_peaks >= 0
    TEST_ASSERT_TRUE(features[6] >= 0.0f && features[6] <= 1.0f);   // FWHM [0,1]
    TEST_ASSERT_TRUE(features[7] >= 0.0f);                           // turb_mad >= 0
    TEST_ASSERT_TRUE(features[8] >= 0.0f);                           // turb_iqr >= 0
    TEST_ASSERT_TRUE(features[9] >= 0.0f);                           // phase_diff_var >= 0
    TEST_ASSERT_TRUE(features[10] >= 0.0f);                          // phase_entropy >= 0
    TEST_ASSERT_TRUE(features[11] >= 0.0f && features[11] <= 1.0f); // circular variance [0,1]
}

// ============================================================================
// GESTURE DETECTOR RING BUFFER TESTS
// ============================================================================

void test_gesture_detector_initial_state(void) {
    GestureDetector gd;

    TEST_ASSERT_FALSE(gd.is_active());
    TEST_ASSERT_EQUAL(0, gd.event_len());
    TEST_ASSERT_NULL(gd.last_gesture());
}

void test_gesture_detector_ring_fill(void) {
    GestureDetector gd;

    // Feed 50 packets — ring should not be full yet
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    for (int i = 0; i < 50; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    // Not active, event empty
    TEST_ASSERT_FALSE(gd.is_active());
    TEST_ASSERT_EQUAL(0, gd.event_len());

    // start_detection after 50 packets: pre-roll should be 50 packets
    gd.start_detection();
    TEST_ASSERT_TRUE(gd.is_active());
    TEST_ASSERT_EQUAL(50, gd.event_len());
}

void test_gesture_detector_ring_full_preroll(void) {
    GestureDetector gd;

    // Fill ring buffer beyond capacity
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    for (int i = 0; i < GESTURE_RING_SIZE + 50; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    // Pre-roll should be capped to fixed pre-roll length (ring can be full)
    gd.start_detection();
    TEST_ASSERT_EQUAL(GESTURE_PREROLL_LEN, gd.event_len());
}

void test_gesture_detector_event_accumulation(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    // Fill ring partially
    for (int i = 0; i < 30; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    gd.start_detection();
    uint16_t preroll = gd.event_len();

    // Feed 20 more packets while active
    for (int i = 0; i < 20; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    TEST_ASSERT_EQUAL(preroll + 20, gd.event_len());
}

void test_gesture_detector_finalize_detection_short_event(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    // Don't pre-fill ring — event will be too short for classification
    gd.start_detection();  // pre-roll = 0 packets
    TEST_ASSERT_EQUAL(0, gd.event_len());

    // Feed only 5 packets (< fixed 200-packet window)
    for (int i = 0; i < 5; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    // finalize_detection returns nullptr because event is too short
    const char* gesture = gd.finalize_detection();
    TEST_ASSERT_NULL(gesture);
    TEST_ASSERT_FALSE(gd.is_active());
}

void test_gesture_detector_finalize_detection_no_weights(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    // Fill ring partially
    for (int i = 0; i < 50; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    gd.start_detection();

    // Feed 50 more packets while active (window remains incomplete: 100/200)
    for (int i = 0; i < 50; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    TEST_ASSERT_TRUE(gd.event_len() < GESTURE_WINDOW_LEN);

    const char* gesture = gd.finalize_detection();

    // Fixed-window policy: incomplete events are never classified.
    TEST_ASSERT_NULL(gesture);

    TEST_ASSERT_FALSE(gd.is_active());
    TEST_ASSERT_EQUAL(0, gd.event_len());
}

void test_gesture_detector_clear_ring(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    // Fill ring
    for (int i = 0; i < 100; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    // Start event, then clear ring (simulate channel change)
    gd.start_detection();
    TEST_ASSERT_TRUE(gd.is_active());

    gd.clear_ring();

    // Should be reset
    TEST_ASSERT_FALSE(gd.is_active());
    TEST_ASSERT_EQUAL(0, gd.event_len());

    // After clear, start_detection should yield empty pre-roll
    gd.start_detection();
    TEST_ASSERT_EQUAL(0, gd.event_len());
}

void test_gesture_detector_reset(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    for (int i = 0; i < 50; i++) {
        gd.process_packet(csi, CSI_LEN);
    }
    gd.start_detection();
    for (int i = 0; i < 20; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    gd.reset();

    TEST_ASSERT_FALSE(gd.is_active());
    TEST_ASSERT_EQUAL(0, gd.event_len());
    TEST_ASSERT_NULL(gd.last_gesture());
}

void test_gesture_detector_max_event_len(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    gd.start_detection();

    // Feed more than max event length
    for (int i = 0; i < GESTURE_MAX_EVENT_LEN + 100; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    // Event should be capped at GESTURE_MAX_EVENT_LEN
    TEST_ASSERT_EQUAL(GESTURE_MAX_EVENT_LEN, gd.event_len());
}

// ============================================================================
// CROSS-PLATFORM FEATURE VALIDATION TESTS
// These tests use synthetic data that can be replicated in Python tests
// to ensure C++ and Python implementations produce identical results.
// ============================================================================

void test_cross_platform_triangular_peak(void) {
    constexpr uint16_t N = 100;
    float turb[N];
    float phases[N * 12];
    memset(phases, 0, sizeof(phases));

    // Create constant baseline with triangular peak at center
    for (int i = 0; i < N; i++) {
        turb[i] = 1.0f;
    }
    int mid = N / 2;
    turb[mid - 1] = 3.0f;
    turb[mid] = 10.0f;
    turb[mid + 1] = 3.0f;

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // event_duration: log-compressed + quantized (N=100 -> 0.9)
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.9f, features[0]);

    // peak_position = 50/99 ≈ 0.505
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 50.0f / 99.0f, features[1]);

    // peak_to_mean_ratio: peak=10, mean=(97*1 + 3 + 10 + 3)/100 = 1.13 → ratio ≈ 8.85
    float expected_mean = (97 * 1.0f + 3.0f + 10.0f + 3.0f) / 100.0f;
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 10.0f / expected_mean, features[2]);

    // symmetric peak -> rise_fall_asymmetry ~= 0
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.0f, features[3]);

    // n_local_peaks = 1 → 0.1
    TEST_ASSERT_FLOAT_WITHIN(0.05f, 0.1f, features[5]);
}

void test_cross_platform_constant_signal(void) {
    constexpr uint16_t N = 200;
    float turb[N];
    float phases[N * 12];
    memset(phases, 0, sizeof(phases));

    for (int i = 0; i < N; i++) {
        turb[i] = 5.0f;
    }

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // event_duration = 200/200 = 1.0
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, features[0]);

    // peak_to_mean_ratio = 1.0 for constant signal
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, features[2]);

    // pre_post_energy_ratio = 1.0 for constant signal
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, features[4]);

    // n_local_peaks = 0 for constant signal
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, features[5]);
}

void test_cross_platform_ramp_signal(void) {
    constexpr uint16_t N = 100;
    float turb[N];
    float phases[N * 12];
    memset(phases, 0, sizeof(phases));

    // Linear ramp from 0 to 99
    for (int i = 0; i < N; i++) {
        turb[i] = static_cast<float>(i);
    }

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // peak_position = 99/99 = 1.0 (peak at end)
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, features[1]);

    // peak at end -> rise_fall_asymmetry near +1
    TEST_ASSERT_TRUE(features[3] > 0.95f);

    // pre_post_energy_ratio < 1 (more energy in second half)
    TEST_ASSERT_TRUE(features[4] < 1.0f);
}

void test_cross_platform_phases_constant(void) {
    constexpr uint16_t N = 50;
    constexpr uint8_t N_PH = 4;
    float turb[N];
    float phases[N * N_PH];

    // All turbulence = 1.0, all phases = 0.5
    for (int i = 0; i < N; i++) {
        turb[i] = 1.0f;
        for (int j = 0; j < N_PH; j++) {
            phases[i * N_PH + j] = 0.5f;
        }
    }

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, N_PH, features);

    // phase_diff_var = 0 (all phases identical)
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, features[9]);

    // phase_entropy = 0 (flat phase per packet)
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, features[10]);

    // circular variance = 0 (all phases identical)
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, features[11]);
}

void test_cross_platform_phases_linear_ramp(void) {
    constexpr uint16_t N = 20;
    constexpr uint8_t N_PH = 4;
    float turb[N];
    float phases[N * N_PH];

    // Each packet has phases [0, 1, 2, 3]
    for (int i = 0; i < N; i++) {
        turb[i] = 1.0f;
        phases[i * N_PH + 0] = 0.0f;
        phases[i * N_PH + 1] = 1.0f;
        phases[i * N_PH + 2] = 2.0f;
        phases[i * N_PH + 3] = 3.0f;
    }

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, N_PH, features);

    // Circular variance for phases [0,1,2,3] is ~0.526
    TEST_ASSERT_FLOAT_WITHIN(0.02f, 0.5258f, features[11]);

    // phase_diff_var: diffs = [1, 1, 1], mean_diff = 1, var = 0
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, 0.0f, features[9]);
}

// ============================================================================
// ADDITIONAL EDGE CASE TESTS
// ============================================================================

void test_gesture_features_minimum_length(void) {
    // Event with exactly 2 packets (minimum for non-zero features)
    constexpr uint16_t N = 2;
    float turb[N] = {1.0f, 5.0f};
    float phases[N * 12];
    memset(phases, 0, sizeof(phases));

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // Should produce valid (non-zero) output
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.2f, features[0]);  // duration (log-compressed, quantized)
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 1.0f, features[1]);  // peak at end → position = 1.0
    TEST_ASSERT_TRUE(features[2] > 1.0f);  // peak_to_mean > 1
}

void test_gesture_features_single_packet(void) {
    // Single packet should return defaults (mostly 0.5 or 0)
    constexpr uint16_t N = 1;
    float turb[N] = {5.0f};
    float phases[N * 12];
    memset(phases, 0, sizeof(phases));

    float features[GESTURE_NUM_FEATURES];
    extract_gesture_features(turb, phases, N, 12, features);

    // n < 2 triggers default values
    TEST_ASSERT_FLOAT_WITHIN(1e-4f, 0.0f, features[0]);  // extract returns early
}

void test_gesture_detector_multiple_events(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    // Simulate 3 consecutive events
    for (int evt = 0; evt < 3; evt++) {
        // Fill ring buffer
        for (int i = 0; i < 50; i++) {
            gd.process_packet(csi, CSI_LEN);
        }

        gd.start_detection();
        TEST_ASSERT_TRUE(gd.is_active());

        // Accumulate event
        for (int i = 0; i < 30; i++) {
            gd.process_packet(csi, CSI_LEN);
        }

        const char* gesture = gd.finalize_detection();
        TEST_ASSERT_FALSE(gd.is_active());
        TEST_ASSERT_EQUAL(0, gd.event_len());

        // Result depends on weights availability
        // (gesture may be nullptr or a valid name)
    }
}

void test_gesture_detector_double_motion_start(void) {
    GestureDetector gd;
    int8_t csi[CSI_LEN];
    make_csi(csi, 10.0f, 0);

    // Fill ring
    for (int i = 0; i < 50; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    gd.start_detection();
    uint16_t first_len = gd.event_len();
    TEST_ASSERT_TRUE(gd.is_active());

    // Feed some packets
    for (int i = 0; i < 20; i++) {
        gd.process_packet(csi, CSI_LEN);
    }

    // Second start_detection (re-trigger): should reset event buffer
    gd.start_detection();
    TEST_ASSERT_TRUE(gd.is_active());
    // Event should contain fresh pre-roll from ring
    TEST_ASSERT_TRUE(gd.event_len() <= GESTURE_RING_SIZE);
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    UNITY_BEGIN();

    // Feature extraction tests
    RUN_TEST(test_gesture_features_empty_buffer);
    RUN_TEST(test_gesture_features_constant_signal);
    RUN_TEST(test_gesture_features_single_peak);
    RUN_TEST(test_gesture_features_pre_post_energy_symmetric);
    RUN_TEST(test_gesture_features_output_range);
    RUN_TEST(test_gesture_features_minimum_length);
    RUN_TEST(test_gesture_features_single_packet);

    // Cross-platform validation tests
    RUN_TEST(test_cross_platform_triangular_peak);
    RUN_TEST(test_cross_platform_constant_signal);
    RUN_TEST(test_cross_platform_ramp_signal);
    RUN_TEST(test_cross_platform_phases_constant);
    RUN_TEST(test_cross_platform_phases_linear_ramp);

    // Gesture detector lifecycle tests
    RUN_TEST(test_gesture_detector_initial_state);
    RUN_TEST(test_gesture_detector_ring_fill);
    RUN_TEST(test_gesture_detector_ring_full_preroll);
    RUN_TEST(test_gesture_detector_event_accumulation);
    RUN_TEST(test_gesture_detector_finalize_detection_short_event);
    RUN_TEST(test_gesture_detector_finalize_detection_no_weights);
    RUN_TEST(test_gesture_detector_clear_ring);
    RUN_TEST(test_gesture_detector_reset);
    RUN_TEST(test_gesture_detector_max_event_len);
    RUN_TEST(test_gesture_detector_multiple_events);
    RUN_TEST(test_gesture_detector_double_motion_start);

    return UNITY_END();
}
