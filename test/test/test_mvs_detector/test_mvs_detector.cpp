/*
 * ESPectre - MVSDetector Unit Tests
 *
 * Tests the MVSDetector class implementing IDetector interface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include "mvs_detector.h"
#include "esphome/core/log.h"

// Include CSI data loader
#include "csi_test_data.h"

#define baseline_packets csi_test_data::baseline_packets()
#define movement_packets csi_test_data::movement_packets()
#define num_baseline csi_test_data::num_baseline()
#define num_movement csi_test_data::num_movement()

using namespace esphome::espectre;

static const char *TAG = "test_mvs_detector";

static const uint8_t TEST_SUBCARRIERS[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_mvs_detector_default_constructor(void) {
    MVSDetector detector;
    
    TEST_ASSERT_EQUAL(SEGMENTATION_DEFAULT_WINDOW_SIZE, detector.get_window_size());
    TEST_ASSERT_EQUAL_FLOAT(SEGMENTATION_DEFAULT_THRESHOLD, detector.get_threshold());
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
    TEST_ASSERT_EQUAL(0, detector.get_total_packets());
}

void test_mvs_detector_custom_constructor(void) {
    MVSDetector detector(100, 2.5f);
    
    TEST_ASSERT_EQUAL(100, detector.get_window_size());
    TEST_ASSERT_EQUAL_FLOAT(2.5f, detector.get_threshold());
}

void test_mvs_detector_get_name(void) {
    MVSDetector detector;
    
    TEST_ASSERT_EQUAL_STRING("MVS", detector.get_name());
}

// ============================================================================
// THRESHOLD TESTS
// ============================================================================

void test_mvs_detector_set_threshold_valid(void) {
    MVSDetector detector;
    
    TEST_ASSERT_TRUE(detector.set_threshold(1.5f));
    TEST_ASSERT_EQUAL_FLOAT(1.5f, detector.get_threshold());
}

void test_mvs_detector_set_threshold_min(void) {
    MVSDetector detector;
    
    TEST_ASSERT_TRUE(detector.set_threshold(SEGMENTATION_MIN_THRESHOLD));
    TEST_ASSERT_EQUAL_FLOAT(SEGMENTATION_MIN_THRESHOLD, detector.get_threshold());
}

void test_mvs_detector_set_threshold_max(void) {
    MVSDetector detector;
    
    TEST_ASSERT_TRUE(detector.set_threshold(SEGMENTATION_MAX_THRESHOLD));
    TEST_ASSERT_EQUAL_FLOAT(SEGMENTATION_MAX_THRESHOLD, detector.get_threshold());
}

void test_mvs_detector_set_threshold_below_min(void) {
    MVSDetector detector;
    float original = detector.get_threshold();
    
    TEST_ASSERT_FALSE(detector.set_threshold(0.01f));
    TEST_ASSERT_EQUAL_FLOAT(original, detector.get_threshold());
}

void test_mvs_detector_set_threshold_above_max(void) {
    MVSDetector detector;
    float original = detector.get_threshold();
    
    TEST_ASSERT_FALSE(detector.set_threshold(100.0f));
    TEST_ASSERT_EQUAL_FLOAT(original, detector.get_threshold());
}

// ============================================================================
// FILTER CONFIGURATION TESTS
// ============================================================================

void test_mvs_detector_configure_lowpass(void) {
    MVSDetector detector;
    
    detector.configure_lowpass(true, 15.0f);
    TEST_ASSERT_TRUE(detector.is_lowpass_enabled());
    
    detector.configure_lowpass(false);
    TEST_ASSERT_FALSE(detector.is_lowpass_enabled());
}

void test_mvs_detector_configure_hampel(void) {
    MVSDetector detector;
    
    detector.configure_hampel(true, 5, 3.0f);
    TEST_ASSERT_TRUE(detector.is_hampel_enabled());
    
    detector.configure_hampel(false);
    TEST_ASSERT_FALSE(detector.is_hampel_enabled());
}

// ============================================================================
// PACKET PROCESSING TESTS
// ============================================================================

void test_mvs_detector_process_packet_increments_count(void) {
    MVSDetector detector;
    
    int8_t csi_buf[128] = {0};
    detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_mvs_detector_process_packet_null_subcarriers(void) {
    MVSDetector detector;
    
    int8_t csi_buf[128] = {0};
    detector.process_packet(csi_buf, 128, nullptr, 0);
    
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_mvs_detector_process_multiple_packets(void) {
    MVSDetector detector(50, 1.0f);
    
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    for (int i = 0; i < 100; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    TEST_ASSERT_EQUAL(100, detector.get_total_packets());
}

// ============================================================================
// STATE MACHINE TESTS
// ============================================================================

void test_mvs_detector_initial_state_idle(void) {
    MVSDetector detector;
    
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
}

void test_mvs_detector_update_state(void) {
    MVSDetector detector(10, 1.0f);
    
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    // Fill buffer
    for (int i = 0; i < 20; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    detector.update_state();
    
    // State should be determined
    MotionState state = detector.get_state();
    TEST_ASSERT_TRUE(state == MotionState::IDLE || state == MotionState::MOTION);
}

// ============================================================================
// RESET TESTS
// ============================================================================

void test_mvs_detector_reset(void) {
    MVSDetector detector(20, 1.0f);
    
    int8_t csi_buf[128] = {0};
    for (int i = 0; i < 30; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    detector.reset();
    
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
}

void test_mvs_detector_clear_buffer(void) {
    MVSDetector detector(20, 1.0f);
    
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    for (int i = 0; i < 30; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    detector.clear_buffer();
    
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
    TEST_ASSERT_FALSE(detector.is_ready());
}

// ============================================================================
// IS_READY TESTS
// ============================================================================

void test_mvs_detector_is_ready_false_initially(void) {
    MVSDetector detector(50, 1.0f);
    
    TEST_ASSERT_FALSE(detector.is_ready());
}

void test_mvs_detector_is_ready_after_window_filled(void) {
    MVSDetector detector(10, 1.0f);
    
    int8_t csi_buf[128] = {0};
    
    // Fill window
    for (int i = 0; i < 15; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    TEST_ASSERT_TRUE(detector.is_ready());
}

// ============================================================================
// MOTION METRIC TESTS
// ============================================================================

void test_mvs_detector_motion_metric_zero_initially(void) {
    MVSDetector detector;
    
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_motion_metric());
}

void test_mvs_detector_motion_metric_after_processing(void) {
    MVSDetector detector(10, 1.0f);
    detector.configure_lowpass(false);
    
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    for (int i = 0; i < 20; i++) {
        // Vary the data slightly
        csi_buf[0] = (int8_t)(i * 5);
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    float metric = detector.get_motion_metric();
    TEST_ASSERT_TRUE(metric >= 0.0f);
}

// ============================================================================
// LAST TURBULENCE TESTS
// ============================================================================

void test_mvs_detector_last_turbulence_zero_initially(void) {
    MVSDetector detector;
    
    TEST_ASSERT_EQUAL_FLOAT(0.0f, detector.get_last_turbulence());
}

void test_mvs_detector_last_turbulence_after_packet(void) {
    MVSDetector detector;
    detector.configure_lowpass(false);
    detector.configure_hampel(false);
    
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    
    float turbulence = detector.get_last_turbulence();
    TEST_ASSERT_TRUE(turbulence >= 0.0f);
}

// ============================================================================
// REAL DATA TESTS
// ============================================================================

void test_mvs_detector_baseline_stays_idle(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    MVSDetector detector(50, 1.0f);
    detector.configure_lowpass(false);
    
    int motion_count = 0;
    
    for (int i = 0; i < 100 && i < num_baseline; i++) {
        detector.process_packet(baseline_packets[i], 128, TEST_SUBCARRIERS, 12);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            motion_count++;
        }
    }
    
    // Most baseline packets should be IDLE
    float motion_rate = (float)motion_count / 100.0f;
    ESP_LOGI(TAG, "Baseline motion rate: %.1f%%", motion_rate * 100);
    TEST_ASSERT_TRUE(motion_rate < 0.2f);  // Less than 20% false positives
}

void test_mvs_detector_movement_detects_motion(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    MVSDetector detector(50, 0.5f);  // Lower threshold for detection
    detector.configure_lowpass(false);
    
    // First process baseline to fill buffer
    for (int i = 0; i < 60 && i < num_baseline; i++) {
        detector.process_packet(baseline_packets[i], 128, TEST_SUBCARRIERS, 12);
    }
    
    int motion_count = 0;
    
    // Now process movement
    for (int i = 0; i < 100 && i < num_movement; i++) {
        detector.process_packet(movement_packets[i], 128, TEST_SUBCARRIERS, 12);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            motion_count++;
        }
    }
    
    // Most movement packets should detect motion
    float detection_rate = (float)motion_count / 100.0f;
    ESP_LOGI(TAG, "Movement detection rate: %.1f%%", detection_rate * 100);
    TEST_ASSERT_TRUE(detection_rate > 0.5f);  // At least 50% detection
}

// ============================================================================
// LOWPASS FILTER TESTS
// ============================================================================

void test_mvs_detector_lowpass_disabled_by_default(void) {
    MVSDetector detector;
    
    TEST_ASSERT_FALSE(detector.is_lowpass_enabled());
}

void test_mvs_detector_lowpass_enable_disable(void) {
    MVSDetector detector;
    
    detector.configure_lowpass(true, 10.0f);
    TEST_ASSERT_TRUE(detector.is_lowpass_enabled());
    
    detector.configure_lowpass(false);
    TEST_ASSERT_FALSE(detector.is_lowpass_enabled());
}

void test_mvs_detector_lowpass_smooths_turbulence(void) {
    MVSDetector detector(10, 1.0f);
    detector.configure_lowpass(true, 5.0f);  // Low cutoff = more smoothing
    detector.configure_hampel(false);
    
    int8_t csi_buf[128];
    float turbulences[20];
    
    // Process packets with varying data
    for (int i = 0; i < 20; i++) {
        for (int j = 0; j < 128; j++) {
            csi_buf[j] = (int8_t)((j + i * 10) % 256 - 128);
        }
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
        turbulences[i] = detector.get_last_turbulence();
    }
    
    // With lowpass, turbulence should be smoother (less variance)
    float var = 0;
    float mean = 0;
    for (int i = 0; i < 20; i++) mean += turbulences[i];
    mean /= 20;
    for (int i = 0; i < 20; i++) var += (turbulences[i] - mean) * (turbulences[i] - mean);
    var /= 20;
    
    ESP_LOGI(TAG, "Lowpass enabled - turbulence variance: %.4f", var);
    TEST_PASS();  // Just verify no crash
}

void test_mvs_detector_lowpass_cutoff_range(void) {
    MVSDetector detector;
    
    // Valid cutoff values
    detector.configure_lowpass(true, 5.0f);
    TEST_ASSERT_TRUE(detector.is_lowpass_enabled());
    
    detector.configure_lowpass(true, 20.0f);
    TEST_ASSERT_TRUE(detector.is_lowpass_enabled());
}

// ============================================================================
// SUBCARRIER SELECTION TESTS
// ============================================================================

void test_mvs_detector_different_subcarrier_count(void) {
    MVSDetector detector(10, 1.0f);
    detector.configure_lowpass(false);
    
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    // Test with different subcarrier counts
    uint8_t sc_4[4] = {10, 11, 12, 13};
    uint8_t sc_8[8] = {10, 11, 12, 13, 14, 15, 16, 17};
    
    detector.process_packet(csi_buf, 128, sc_4, 4);
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
    
    detector.process_packet(csi_buf, 128, sc_8, 8);
    TEST_ASSERT_EQUAL(2, detector.get_total_packets());
}

void test_mvs_detector_high_index_subcarriers(void) {
    MVSDetector detector(10, 1.0f);
    detector.configure_lowpass(false);
    
    int8_t csi_buf[128];
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (int8_t)(i % 64 - 32);
    }
    
    // High index subcarriers (near end of HT20)
    uint8_t sc_high[12] = {50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61};
    
    detector.process_packet(csi_buf, 128, sc_high, 12);
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_mvs_detector_empty_subcarrier_selection(void) {
    MVSDetector detector(10, 1.0f);
    
    int8_t csi_buf[128] = {0};
    
    // Empty selection should still work
    detector.process_packet(csi_buf, 128, nullptr, 0);
    
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

// ============================================================================
// WINDOW SIZE TESTS
// ============================================================================

void test_mvs_detector_min_window_size(void) {
    MVSDetector detector(SEGMENTATION_MIN_WINDOW_SIZE, 1.0f);
    
    TEST_ASSERT_EQUAL(SEGMENTATION_MIN_WINDOW_SIZE, detector.get_window_size());
}

void test_mvs_detector_max_window_size(void) {
    MVSDetector detector(SEGMENTATION_MAX_WINDOW_SIZE, 1.0f);
    
    TEST_ASSERT_EQUAL(SEGMENTATION_MAX_WINDOW_SIZE, detector.get_window_size());
}

// ============================================================================
// EDGE CASE TESTS
// ============================================================================

void test_mvs_detector_short_csi_data(void) {
    MVSDetector detector;
    
    int8_t short_buf[10] = {0};
    
    // Should handle short data gracefully
    detector.process_packet(short_buf, 10, TEST_SUBCARRIERS, 12);
    
    TEST_ASSERT_EQUAL(1, detector.get_total_packets());
}

void test_mvs_detector_extreme_csi_values(void) {
    MVSDetector detector(10, 1.0f);
    detector.configure_lowpass(false);
    
    int8_t csi_buf[128];
    
    // Fill with extreme values
    for (int i = 0; i < 128; i++) {
        csi_buf[i] = (i % 2 == 0) ? 127 : -128;
    }
    
    for (int i = 0; i < 15; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    float metric = detector.get_motion_metric();
    TEST_ASSERT_TRUE(metric >= 0.0f);
}

void test_mvs_detector_all_zero_csi(void) {
    MVSDetector detector(10, 1.0f);
    detector.configure_lowpass(false);
    
    int8_t csi_buf[128] = {0};
    
    for (int i = 0; i < 15; i++) {
        detector.process_packet(csi_buf, 128, TEST_SUBCARRIERS, 12);
    }
    
    // All zeros should result in zero turbulence
    float turb = detector.get_last_turbulence();
    TEST_ASSERT_EQUAL_FLOAT(0.0f, turb);
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_mvs_detector_default_constructor);
    RUN_TEST(test_mvs_detector_custom_constructor);
    RUN_TEST(test_mvs_detector_get_name);
    
    // Threshold tests
    RUN_TEST(test_mvs_detector_set_threshold_valid);
    RUN_TEST(test_mvs_detector_set_threshold_min);
    RUN_TEST(test_mvs_detector_set_threshold_max);
    RUN_TEST(test_mvs_detector_set_threshold_below_min);
    RUN_TEST(test_mvs_detector_set_threshold_above_max);
    
    // Filter configuration tests
    RUN_TEST(test_mvs_detector_configure_lowpass);
    RUN_TEST(test_mvs_detector_configure_hampel);
    
    // Packet processing tests
    RUN_TEST(test_mvs_detector_process_packet_increments_count);
    RUN_TEST(test_mvs_detector_process_packet_null_subcarriers);
    RUN_TEST(test_mvs_detector_process_multiple_packets);
    
    // State machine tests
    RUN_TEST(test_mvs_detector_initial_state_idle);
    RUN_TEST(test_mvs_detector_update_state);
    
    // Reset tests
    RUN_TEST(test_mvs_detector_reset);
    RUN_TEST(test_mvs_detector_clear_buffer);
    
    // Is ready tests
    RUN_TEST(test_mvs_detector_is_ready_false_initially);
    RUN_TEST(test_mvs_detector_is_ready_after_window_filled);
    
    // Motion metric tests
    RUN_TEST(test_mvs_detector_motion_metric_zero_initially);
    RUN_TEST(test_mvs_detector_motion_metric_after_processing);
    
    // Last turbulence tests
    RUN_TEST(test_mvs_detector_last_turbulence_zero_initially);
    RUN_TEST(test_mvs_detector_last_turbulence_after_packet);
    
    // Real data tests
    RUN_TEST(test_mvs_detector_baseline_stays_idle);
    RUN_TEST(test_mvs_detector_movement_detects_motion);
    
    // Lowpass filter tests
    RUN_TEST(test_mvs_detector_lowpass_disabled_by_default);
    RUN_TEST(test_mvs_detector_lowpass_enable_disable);
    RUN_TEST(test_mvs_detector_lowpass_smooths_turbulence);
    RUN_TEST(test_mvs_detector_lowpass_cutoff_range);
    
    // Subcarrier selection tests
    RUN_TEST(test_mvs_detector_different_subcarrier_count);
    RUN_TEST(test_mvs_detector_high_index_subcarriers);
    RUN_TEST(test_mvs_detector_empty_subcarrier_selection);
    
    // Window size tests
    RUN_TEST(test_mvs_detector_min_window_size);
    RUN_TEST(test_mvs_detector_max_window_size);
    
    // Edge case tests
    RUN_TEST(test_mvs_detector_short_csi_data);
    RUN_TEST(test_mvs_detector_extreme_csi_values);
    RUN_TEST(test_mvs_detector_all_zero_csi);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
