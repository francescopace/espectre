/*
 * ESPectre - NBVICalibrator Tests
 *
 * Tests for the NBVICalibrator class with real CSI data.
 * Uses file-based storage with temporary files.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include "nbvi_calibrator.h"
#include "csi_manager.h"
#include "csi_processor.h"
#include "threshold.h"
#include "utils.h"
#include "esphome/core/log.h"

// Include CSI data loader (loads from NPZ files)
#include "csi_test_data.h"

// Compatibility macros for existing test code
#define baseline_packets csi_test_data::baseline_packets()
#define movement_packets csi_test_data::movement_packets()
#define num_baseline csi_test_data::num_baseline()
#define num_movement csi_test_data::num_movement()
#define packet_size csi_test_data::packet_size()

using namespace esphome::espectre;

static const char *TAG = "test_nbvi_calibrator";

// Test buffer file path
static const char* TEST_BUFFER_PATH = "/tmp/test_nbvi_buffer.bin";

// Global CSI processor for tests
static csi_processor_context_t g_processor;

// Default subcarrier band for testing
static const uint8_t DEFAULT_BAND[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
static const uint8_t DEFAULT_BAND_SIZE = 12;

void setUp(void) {
    // Load test data if not already loaded
    if (!csi_test_data::is_loaded()) {
        csi_test_data::load();
    }
    csi_processor_init(&g_processor, 50, 1.0f);
    remove(TEST_BUFFER_PATH);
}

void tearDown(void) {
    csi_processor_cleanup(&g_processor);
    remove(TEST_BUFFER_PATH);
}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_init_without_csi_manager(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    // Should not crash, but calibration should fail without CSI manager
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

void test_init_with_custom_buffer_path(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    // Just verify it doesn't crash
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

void test_start_calibration_fails_without_csi_manager(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    esp_err_t err = calibrator.start_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE, nullptr);
    
    // Should fail because CSI manager is not set
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, err);
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

// ============================================================================
// CONFIGURATION TESTS
// ============================================================================

void test_set_buffer_size(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    calibrator.set_buffer_size(500);
    TEST_ASSERT_EQUAL(500, calibrator.get_buffer_size());
    
    calibrator.set_buffer_size(100);
    TEST_ASSERT_EQUAL(100, calibrator.get_buffer_size());
}

void test_set_buffer_size_default(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    // Default buffer size should be 700
    TEST_ASSERT_EQUAL(700, calibrator.get_buffer_size());
}

void test_set_window_size(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    // Just verify it doesn't crash
    calibrator.set_window_size(100);
    calibrator.set_window_step(25);
}

void test_set_alpha(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    // Just verify it doesn't crash
    calibrator.set_alpha(0.3f);
    calibrator.set_alpha(0.7f);
}

void test_set_min_spacing(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    calibrator.set_min_spacing(2);
    calibrator.set_min_spacing(3);
}

void test_set_percentile(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    calibrator.set_percentile(5);
    calibrator.set_percentile(15);
}

void test_set_noise_gate_percentile(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    calibrator.set_noise_gate_percentile(20);
    calibrator.set_noise_gate_percentile(30);
}

// ============================================================================
// IS_CALIBRATING TESTS
// ============================================================================

void test_is_calibrating_false_initially(void) {
    NBVICalibrator calibrator;
    calibrator.init(nullptr, TEST_BUFFER_PATH);
    
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

// ============================================================================
// THRESHOLD CALCULATION TESTS (using threshold.h)
// ============================================================================

void test_threshold_calculation_with_mv_values(void) {
    // Test threshold calculation with synthetic MV values
    std::vector<float> mv_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f, 0.9f, 1.0f};
    
    float threshold = calculate_adaptive_threshold(mv_values, 95, 1.4f);
    
    // P95 of [0.1-1.0] should be around 0.95, multiplied by 1.4
    TEST_ASSERT_TRUE(threshold > 0.0f);
    TEST_ASSERT_TRUE(threshold < 5.0f);
    
    ESP_LOGI(TAG, "Threshold from synthetic MV: %.4f", threshold);
}

void test_threshold_calculation_empty_values(void) {
    std::vector<float> mv_values;
    
    float threshold = calculate_adaptive_threshold(mv_values, 95, 1.4f);
    
    // Empty values should return default (1.0 * 1.4 = 1.4)
    TEST_ASSERT_EQUAL_FLOAT(1.4f, threshold);
}

void test_threshold_mode_auto(void) {
    std::vector<float> mv_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    
    float threshold_auto;
    uint8_t percentile;
    float factor;
    float pxx;
    
    calculate_adaptive_threshold(mv_values, ThresholdMode::AUTO, 
                                  threshold_auto, percentile, factor, pxx);
    
    TEST_ASSERT_EQUAL(95, percentile);
    TEST_ASSERT_EQUAL_FLOAT(1.4f, factor);
    TEST_ASSERT_TRUE(threshold_auto > 0.0f);
}

void test_threshold_mode_min(void) {
    std::vector<float> mv_values = {0.1f, 0.2f, 0.3f, 0.4f, 0.5f};
    
    float threshold_min;
    uint8_t percentile;
    float factor;
    float pxx;
    
    calculate_adaptive_threshold(mv_values, ThresholdMode::MIN, 
                                  threshold_min, percentile, factor, pxx);
    
    TEST_ASSERT_EQUAL(100, percentile);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, factor);
    TEST_ASSERT_TRUE(threshold_min > 0.0f);
}

// ============================================================================
// CALIBRATION TESTS WITH REAL DATA
// ============================================================================

void test_nbvi_full_calibration_with_real_data(void) {
    // Create CSI Manager and NBVI Calibrator
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f);
    
    NBVICalibrator calibrator;
    calibrator.init(&csi_manager, TEST_BUFFER_PATH);
    calibrator.set_buffer_size(200);
    
    // Variables to capture callback results
    uint8_t result_band[12] = {0};
    uint8_t result_size = 0;
    std::vector<float> result_mv_values;
    bool calibration_success = false;
    
    // Start calibration
    esp_err_t err = calibrator.start_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& mv_values, bool success) {
            if (success && size > 0) {
                memcpy(result_band, band, size);
                result_size = size;
                result_mv_values = mv_values;
            }
            calibration_success = success;
        });
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(calibrator.is_calibrating());
    
    // Feed baseline packets
    size_t packets_to_feed = std::min((size_t)200, (size_t)num_baseline);
    for (size_t i = 0; i < packets_to_feed; i++) {
        bool buffer_full = calibrator.add_packet(baseline_packets[i], packet_size);
        if (buffer_full) {
            break;
        }
    }
    
    // Calibration should have completed
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
    TEST_ASSERT_TRUE(calibration_success);
    TEST_ASSERT_EQUAL(12, result_size);
    
    // Verify all subcarriers are in valid range
    ESP_LOGI(TAG, "NBVI selected subcarriers:");
    for (int i = 0; i < result_size; i++) {
        ESP_LOGI(TAG, "  %d: SC %d", i+1, result_band[i]);
        TEST_ASSERT_TRUE(result_band[i] >= 11);
        TEST_ASSERT_TRUE(result_band[i] <= 52);
    }
    
    // Verify MV values are returned
    TEST_ASSERT_TRUE(result_mv_values.size() > 0);
}

void test_nbvi_returns_mv_values_for_threshold(void) {
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f);
    
    NBVICalibrator calibrator;
    calibrator.init(&csi_manager, TEST_BUFFER_PATH);
    calibrator.set_buffer_size(200);
    
    std::vector<float> result_mv_values;
    bool calibration_success = false;
    
    calibrator.start_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& mv_values, bool success) {
            (void)band;
            (void)size;
            result_mv_values = mv_values;
            calibration_success = success;
        });
    
    // Feed baseline packets
    size_t packets_to_feed = std::min((size_t)200, (size_t)num_baseline);
    for (size_t i = 0; i < packets_to_feed; i++) {
        calibrator.add_packet(baseline_packets[i], packet_size);
    }
    
    TEST_ASSERT_TRUE(calibration_success);
    TEST_ASSERT_TRUE(result_mv_values.size() > 0);
    
    // Calculate adaptive threshold from mv_values
    float adaptive_threshold = calculate_adaptive_threshold(result_mv_values, 95, 1.4f);
    TEST_ASSERT_TRUE(adaptive_threshold > 0.0f);
    TEST_ASSERT_TRUE(adaptive_threshold < 10.0f);
    
    ESP_LOGI(TAG, "NBVI adaptive threshold: %.4f (from %zu mv_values)", 
             adaptive_threshold, result_mv_values.size());
}

void test_nbvi_add_packet_rejects_short_data(void) {
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f);
    
    NBVICalibrator calibrator;
    calibrator.init(&csi_manager, TEST_BUFFER_PATH);
    
    calibrator.start_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {});
    
    // Short packet should be rejected
    int8_t short_data[64] = {0};  // Too short (should be 128 for HT20)
    bool result = calibrator.add_packet(short_data, 64);
    TEST_ASSERT_FALSE(result);
}

void test_nbvi_is_calibrating_true_after_start(void) {
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f);
    
    NBVICalibrator calibrator;
    calibrator.init(&csi_manager, TEST_BUFFER_PATH);
    
    calibrator.start_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {});
    
    TEST_ASSERT_TRUE(calibrator.is_calibrating());
}

void test_nbvi_is_calibrating_false_after_completion(void) {
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, true, 11.0f, false, 7, 3.0f);
    
    NBVICalibrator calibrator;
    calibrator.init(&csi_manager, TEST_BUFFER_PATH);
    calibrator.set_buffer_size(50);  // Small buffer for quick test
    
    calibrator.start_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [](const uint8_t*, uint8_t, const std::vector<float>&, bool) {});
    
    // Feed packets until complete
    for (int i = 0; i < 50 && calibrator.is_calibrating(); i++) {
        calibrator.add_packet(baseline_packets[i % num_baseline], packet_size);
    }
    
    TEST_ASSERT_FALSE(calibrator.is_calibrating());
}

// ============================================================================
// TEST RUNNER
// ============================================================================

extern "C" int main(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_init_without_csi_manager);
    RUN_TEST(test_init_with_custom_buffer_path);
    RUN_TEST(test_start_calibration_fails_without_csi_manager);
    
    // Configuration tests
    RUN_TEST(test_set_buffer_size);
    RUN_TEST(test_set_buffer_size_default);
    RUN_TEST(test_set_window_size);
    RUN_TEST(test_set_alpha);
    RUN_TEST(test_set_min_spacing);
    RUN_TEST(test_set_percentile);
    RUN_TEST(test_set_noise_gate_percentile);
    
    // Is calibrating tests
    RUN_TEST(test_is_calibrating_false_initially);
    
    // Threshold calculation tests
    RUN_TEST(test_threshold_calculation_with_mv_values);
    RUN_TEST(test_threshold_calculation_empty_values);
    RUN_TEST(test_threshold_mode_auto);
    RUN_TEST(test_threshold_mode_min);
    
    // Calibration tests with real data
    RUN_TEST(test_nbvi_full_calibration_with_real_data);
    RUN_TEST(test_nbvi_returns_mv_values_for_threshold);
    RUN_TEST(test_nbvi_add_packet_rejects_short_data);
    RUN_TEST(test_nbvi_is_calibrating_true_after_start);
    RUN_TEST(test_nbvi_is_calibrating_false_after_completion);
    
    return UNITY_END();
}
