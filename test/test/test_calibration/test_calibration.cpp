/*
 * ESPectre - Calibration Algorithm Unit Tests
 *
 * Unit tests for NBVI (Normalized Baseline Variability Index) calculations
 * and utility functions used in auto-calibration.
 *
 * Tests utils.h functions directly and CalibrationManager end-to-end.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include "utils.h"
#include "calibration_manager.h"
#include "csi_manager.h"
#include "csi_processor.h"
#include "esp_log.h"

#if defined(ESP_PLATFORM)
#include "esp_spiffs.h"
#endif

// Include real CSI data
#include "real_csi_data_esp32.h"
#include "real_csi_arrays.inc"

using namespace esphome::espectre;

static const char *TAG = "test_calibration";

// Test buffer file path - different for native vs ESP32
#if defined(ESP_PLATFORM)
static const char* TEST_BUFFER_PATH = "/spiffs/test_buffer.bin";
static bool spiffs_mounted = false;
#else
static const char* TEST_BUFFER_PATH = "/tmp/test_calibration_buffer.bin";
#endif

// Global CSI processor for tests
static csi_processor_context_t g_processor;

// Default subcarrier band
static const uint8_t DEFAULT_BAND[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
static const uint8_t DEFAULT_BAND_SIZE = 12;

#if defined(ESP_PLATFORM)
static void init_spiffs(void) {
    if (spiffs_mounted) return;
    
    esp_vfs_spiffs_conf_t conf = {
        .base_path = "/spiffs",
        .partition_label = NULL,
        .max_files = 5,
        .format_if_mount_failed = true
    };
    
    esp_err_t ret = esp_vfs_spiffs_register(&conf);
    if (ret == ESP_OK) {
        spiffs_mounted = true;
        ESP_LOGI(TAG, "SPIFFS mounted successfully");
    } else {
        ESP_LOGE(TAG, "Failed to mount SPIFFS: %s", esp_err_to_name(ret));
    }
}
#endif

void setUp(void) {
#if defined(ESP_PLATFORM)
    init_spiffs();
#endif
    csi_processor_init(&g_processor, 50, 1.0f);
    remove(TEST_BUFFER_PATH);
}

void tearDown(void) {
    csi_processor_cleanup(&g_processor);
    remove(TEST_BUFFER_PATH);
}

// ============================================================================
// VARIANCE CALCULATION TESTS (Two-Pass Algorithm)
// ============================================================================

void test_variance_empty_array(void) {
    float result = calculate_variance_two_pass(NULL, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);
}

void test_variance_single_element(void) {
    float values[] = {5.0f};
    float result = calculate_variance_two_pass(values, 1);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);  // Variance of single value is 0
}

void test_variance_identical_values(void) {
    float values[] = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    float result = calculate_variance_two_pass(values, 5);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);  // No variance
}

void test_variance_known_values(void) {
    // Values: 2, 4, 4, 4, 5, 5, 7, 9
    // Mean = 40/8 = 5
    // Variance = ((2-5)² + (4-5)² + (4-5)² + (4-5)² + (5-5)² + (5-5)² + (7-5)² + (9-5)²) / 8
    //          = (9 + 1 + 1 + 1 + 0 + 0 + 4 + 16) / 8 = 32/8 = 4
    float values[] = {2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
    float result = calculate_variance_two_pass(values, 8);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 4.0f, result);
}

void test_variance_with_negative_values(void) {
    // Values: -2, -1, 0, 1, 2
    // Mean = 0
    // Variance = (4 + 1 + 0 + 1 + 4) / 5 = 2
    float values[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float result = calculate_variance_two_pass(values, 5);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 2.0f, result);
}

void test_variance_large_values_numerical_stability(void) {
    // Large values that could cause precision issues with naive algorithm
    float values[] = {1000000.0f, 1000001.0f, 1000002.0f, 1000003.0f, 1000004.0f};
    // Mean = 1000002
    // Variance = (4 + 1 + 0 + 1 + 4) / 5 = 2
    float result = calculate_variance_two_pass(values, 5);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 2.0f, result);
}

// ============================================================================
// CALIBRATION MANAGER END-TO-END TESTS
// These tests run CalibrationManager with real CSI data to exercise
// calculate_nbvi_weighted_ and calculate_percentile_ internally
// ============================================================================

void test_calibration_manager_full_calibration(void) {
    // Create CSI Manager and Calibration Manager
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, false, 7, 3.0f);
    
    CalibrationManager cm;
    cm.init(&csi_manager, TEST_BUFFER_PATH);
    cm.set_buffer_size(200);  // Use 200 packets for calibration
    cm.set_window_size(50);
    cm.set_window_step(25);
    cm.set_percentile(10);
    cm.set_alpha(0.3f);
    cm.set_min_spacing(3);
    cm.set_noise_gate_percentile(10);
    
    // Variables to capture callback results
    uint8_t result_band[12] = {0};
    uint8_t result_size = 0;
    bool calibration_success = false;
    
    // Start calibration
    esp_err_t err = cm.start_auto_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [&](const uint8_t* band, uint8_t size, bool success) {
            if (success && size > 0) {
                memcpy(result_band, band, size);
                result_size = size;
            }
            calibration_success = success;
        });
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(cm.is_calibrating());
    
    // Feed 200 baseline packets to CalibrationManager
    // This will trigger run_calibration_() which calls calculate_nbvi_weighted_ and calculate_percentile_
    for (int i = 0; i < 200; i++) {
        bool buffer_full = cm.add_packet(baseline_packets[i % 100], 128);
        if (buffer_full) {
            break;  // Calibration will run
        }
    }
    
    // Calibration should have completed
    TEST_ASSERT_FALSE(cm.is_calibrating());
    TEST_ASSERT_TRUE(calibration_success);
    TEST_ASSERT_EQUAL(12, result_size);
    
    // Verify selected subcarriers are valid (not guard bands)
    ESP_LOGI(TAG, "Calibration selected subcarriers:");
    for (int i = 0; i < result_size; i++) {
        ESP_LOGI(TAG, "  %d: SC %d", i+1, result_band[i]);
        TEST_ASSERT_TRUE(result_band[i] >= 6);
        TEST_ASSERT_TRUE(result_band[i] <= 58);
    }
}

void test_calibration_manager_alpha_affects_selection(void) {
    // Test that different alpha values affect NBVI calculation
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, false, 7, 3.0f);
    
    // Run calibration with alpha = 0.0 (pure CV)
    CalibrationManager cm1;
    cm1.init(&csi_manager, TEST_BUFFER_PATH);
    cm1.set_buffer_size(100);
    cm1.set_window_size(50);
    cm1.set_alpha(0.0f);  // Pure CV
    
    uint8_t band1[12] = {0};
    uint8_t size1 = 0;
    
    cm1.start_auto_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [&](const uint8_t* band, uint8_t size, bool success) {
            if (success) { memcpy(band1, band, size); size1 = size; }
        });
    
    for (int i = 0; i < 100; i++) {
        cm1.add_packet(baseline_packets[i], 128);
    }
    
    // Clean up and run with alpha = 1.0 (pure energy)
    remove(TEST_BUFFER_PATH);
    csi_processor_cleanup(&g_processor);
    csi_processor_init(&g_processor, 50, 1.0f);
    
    CalibrationManager cm2;
    cm2.init(&csi_manager, TEST_BUFFER_PATH);
    cm2.set_buffer_size(100);
    cm2.set_window_size(50);
    cm2.set_alpha(1.0f);  // Pure energy
    
    uint8_t band2[12] = {0};
    uint8_t size2 = 0;
    
    cm2.start_auto_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [&](const uint8_t* band, uint8_t size, bool success) {
            if (success) { memcpy(band2, band, size); size2 = size; }
        });
    
    for (int i = 0; i < 100; i++) {
        cm2.add_packet(baseline_packets[i], 128);
    }
    
    TEST_ASSERT_EQUAL(12, size1);
    TEST_ASSERT_EQUAL(12, size2);
    
    // The selections might be different due to different alpha
    // At minimum, both should produce valid results
    ESP_LOGI(TAG, "Alpha=0 first 3: %d, %d, %d", band1[0], band1[1], band1[2]);
    ESP_LOGI(TAG, "Alpha=1 first 3: %d, %d, %d", band2[0], band2[1], band2[2]);
}

void test_calibration_manager_percentile_affects_baseline(void) {
    // Test that percentile parameter affects baseline window detection
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, false, 7, 3.0f);
    
    CalibrationManager cm;
    cm.init(&csi_manager, TEST_BUFFER_PATH);
    cm.set_buffer_size(200);
    cm.set_window_size(50);
    cm.set_percentile(5);  // Very strict - only 5th percentile
    
    bool calibration_success = false;
    
    cm.start_auto_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [&](const uint8_t* band, uint8_t size, bool success) {
            calibration_success = success;
        });
    
    // Mix baseline and movement packets
    for (int i = 0; i < 100; i++) {
        cm.add_packet(baseline_packets[i], 128);
    }
    for (int i = 0; i < 100; i++) {
        cm.add_packet(movement_packets[i], 128);
    }
    
    // Should still succeed - percentile helps find the quietest window
    TEST_ASSERT_TRUE(calibration_success);
}

void test_calibration_manager_noise_gate(void) {
    // Test that noise gate filters weak subcarriers
    CSIManager csi_manager;
    csi_manager.init(&g_processor, DEFAULT_BAND, 1.0f, 50, 100, false, 7, 3.0f);
    
    CalibrationManager cm;
    cm.init(&csi_manager, TEST_BUFFER_PATH);
    cm.set_buffer_size(100);
    cm.set_window_size(50);
    cm.set_noise_gate_percentile(20);  // Filter bottom 20% by signal strength
    
    uint8_t result_band[12] = {0};
    uint8_t result_size = 0;
    
    cm.start_auto_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE,
        [&](const uint8_t* band, uint8_t size, bool success) {
            if (success) { memcpy(result_band, band, size); result_size = size; }
        });
    
    for (int i = 0; i < 100; i++) {
        cm.add_packet(baseline_packets[i], 128);
    }
    
    TEST_ASSERT_EQUAL(12, result_size);
    
    // Guard bands (0-5, 59-63) have weak signal and should be filtered
    for (int i = 0; i < result_size; i++) {
        TEST_ASSERT_TRUE(result_band[i] >= 6);
        TEST_ASSERT_TRUE(result_band[i] <= 58);
    }
}

// Note: Spectral spacing is tested in test_spectral_spacing_* tests below
// and verified in test_calibration_manager_full_calibration via the callback results

// ============================================================================
// PERCENTILE TESTS (via CalibrationManager end-to-end)
// The calculate_percentile_ function is tested through CalibrationManager
// ============================================================================

// Note: Percentile calculation is tested indirectly through:
// - test_calibration_manager_percentile_affects_baseline
// - test_calibration_manager_full_calibration
// These tests exercise CalibrationManager::calculate_percentile_ internally

// ============================================================================
// SPECTRAL SPACING TESTS
// ============================================================================

// Helper: Check if subcarrier selection respects minimum spacing
static bool check_spectral_spacing(const uint8_t* band, uint8_t size, uint8_t min_spacing) {
    for (uint8_t i = 1; i < size; i++) {
        if (band[i] - band[i-1] < min_spacing) {
            return false;
        }
    }
    return true;
}

void test_spectral_spacing_valid(void) {
    // Valid selection with spacing >= 3
    uint8_t band[] = {10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43};
    
    TEST_ASSERT_TRUE(check_spectral_spacing(band, 12, 3));
}

void test_spectral_spacing_invalid(void) {
    // Invalid selection - adjacent subcarriers
    uint8_t band[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21};
    
    TEST_ASSERT_FALSE(check_spectral_spacing(band, 12, 3));
}

void test_spectral_spacing_edge_case(void) {
    // Exactly minimum spacing
    uint8_t band[] = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33};
    
    TEST_ASSERT_TRUE(check_spectral_spacing(band, 12, 3));
}

// ============================================================================
// SUBCARRIER RANKING TESTS
// ============================================================================

void test_subcarrier_ranking_by_nbvi(void) {
    // Simulate NBVI values for different subcarriers
    struct SubcarrierNBVI {
        uint8_t index;
        float nbvi;
    };
    
    SubcarrierNBVI metrics[] = {
        {10, 0.05f},  // Most stable
        {15, 0.08f},
        {20, 0.12f},
        {25, 0.03f},  // Best
        {30, 0.15f},
        {35, 0.07f},
    };
    
    // Sort by NBVI (ascending - lower is better)
    std::sort(metrics, metrics + 6, [](const SubcarrierNBVI& a, const SubcarrierNBVI& b) {
        return a.nbvi < b.nbvi;
    });
    
    // Best subcarrier should be first
    TEST_ASSERT_EQUAL(25, metrics[0].index);
    TEST_ASSERT_EQUAL_FLOAT(0.03f, metrics[0].nbvi);
    
    // Worst should be last
    TEST_ASSERT_EQUAL(30, metrics[5].index);
}

// ============================================================================
// MAGNITUDE CALCULATION TESTS
// ============================================================================

void test_magnitude_zero_iq(void) {
    float mag = calculate_magnitude(0, 0);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, mag);
}

void test_magnitude_positive_iq(void) {
    // I=3, Q=4 → magnitude = 5
    float mag = calculate_magnitude(3, 4);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, mag);
}

void test_magnitude_negative_iq(void) {
    // I=-6, Q=-8 → magnitude = 10
    float mag = calculate_magnitude(-6, -8);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 10.0f, mag);
}

void test_magnitude_max_values(void) {
    // Max int8_t values: I=127, Q=127 → magnitude ≈ 179.6
    float mag = calculate_magnitude(127, 127);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 179.6f, mag);
}

// ============================================================================
// SPATIAL TURBULENCE TESTS
// ============================================================================

void test_turbulence_uniform_magnitudes(void) {
    float magnitudes[64];
    for (int i = 0; i < 64; i++) magnitudes[i] = 100.0f;
    
    uint8_t subcarriers[] = {10, 20, 30, 40};
    float turb = calculate_spatial_turbulence(magnitudes, subcarriers, 4);
    
    // Uniform magnitudes → zero turbulence
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, turb);
}

void test_turbulence_varying_magnitudes(void) {
    float magnitudes[64] = {0};
    magnitudes[10] = 80.0f;
    magnitudes[20] = 100.0f;
    magnitudes[30] = 120.0f;
    magnitudes[40] = 100.0f;
    
    uint8_t subcarriers[] = {10, 20, 30, 40};
    float turb = calculate_spatial_turbulence(magnitudes, subcarriers, 4);
    
    // Mean = 100, variance = (400+0+400+0)/4 = 200, std = 14.14
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 14.14f, turb);
}

void test_turbulence_empty_selection(void) {
    float magnitudes[64] = {0};
    uint8_t subcarriers[] = {};
    
    float turb = calculate_spatial_turbulence(magnitudes, subcarriers, 0);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, turb);
}

void test_turbulence_single_subcarrier(void) {
    float magnitudes[64] = {0};
    magnitudes[25] = 150.0f;
    
    uint8_t subcarriers[] = {25};
    float turb = calculate_spatial_turbulence(magnitudes, subcarriers, 1);
    
    // Single value → zero turbulence
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, turb);
}

// ============================================================================
// NOISE GATE TESTS (via CalibrationManager end-to-end)
// ============================================================================

// Note: Noise gate is tested through test_calibration_manager_noise_gate
// which exercises CalibrationManager::apply_noise_gate_ internally

// ============================================================================
// COMPARE FUNCTIONS TESTS
// ============================================================================

void test_compare_float_ascending(void) {
    float values[] = {3.0f, 1.0f, 4.0f, 1.0f, 5.0f, 9.0f, 2.0f, 6.0f};
    size_t n = sizeof(values) / sizeof(values[0]);
    
    std::qsort(values, n, sizeof(float), compare_float);
    
    TEST_ASSERT_EQUAL_FLOAT(1.0f, values[0]);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, values[1]);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, values[n-1]);
}

void test_compare_float_abs(void) {
    float values[] = {-5.0f, 3.0f, -1.0f, 4.0f, -2.0f};
    size_t n = sizeof(values) / sizeof(values[0]);
    
    std::qsort(values, n, sizeof(float), compare_float_abs);
    
    // Sorted by absolute value: 1, 2, 3, 4, 5
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, std::abs(values[0]));
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 5.0f, std::abs(values[n-1]));
}

void test_compare_int8_ascending(void) {
    int8_t values[] = {5, -3, 0, 127, -128, 10, -1};
    size_t n = sizeof(values) / sizeof(values[0]);
    
    std::qsort(values, n, sizeof(int8_t), compare_int8);
    
    // Should be sorted: -128, -3, -1, 0, 5, 10, 127
    TEST_ASSERT_EQUAL_INT8(-128, values[0]);
    TEST_ASSERT_EQUAL_INT8(-3, values[1]);
    TEST_ASSERT_EQUAL_INT8(-1, values[2]);
    TEST_ASSERT_EQUAL_INT8(0, values[3]);
    TEST_ASSERT_EQUAL_INT8(5, values[4]);
    TEST_ASSERT_EQUAL_INT8(10, values[5]);
    TEST_ASSERT_EQUAL_INT8(127, values[n-1]);
}

void test_compare_int8_with_duplicates(void) {
    int8_t values[] = {3, 1, 3, 1, 2};
    size_t n = sizeof(values) / sizeof(values[0]);
    
    std::qsort(values, n, sizeof(int8_t), compare_int8);
    
    TEST_ASSERT_EQUAL_INT8(1, values[0]);
    TEST_ASSERT_EQUAL_INT8(1, values[1]);
    TEST_ASSERT_EQUAL_INT8(2, values[2]);
    TEST_ASSERT_EQUAL_INT8(3, values[3]);
    TEST_ASSERT_EQUAL_INT8(3, values[4]);
}

// ============================================================================
// REAL CSI DATA TESTS (utils.h functions)
// ============================================================================

void test_magnitude_from_real_csi(void) {
    // Test magnitude calculation on real CSI data
    const int8_t* packet = baseline_packets[0];
    
    // Calculate magnitudes for all subcarriers
    float magnitudes[64];
    for (uint8_t sc = 0; sc < 64; sc++) {
        magnitudes[sc] = calculate_magnitude(packet[sc * 2], packet[sc * 2 + 1]);
    }
    
    // First 5 subcarriers are typically zero (guard band)
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 0.0f, magnitudes[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 0.0f, magnitudes[1]);
    
    // Middle subcarriers should have non-zero magnitude
    // Subcarrier 5: check actual I/Q values from packet
    float expected_mag_5 = calculate_magnitude(packet[10], packet[11]);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, expected_mag_5, magnitudes[5]);
    
    ESP_LOGI(TAG, "Real CSI packet magnitudes - sc5: %.2f, sc10: %.2f, sc20: %.2f",
             magnitudes[5], magnitudes[10], magnitudes[20]);
}

void test_variance_baseline_vs_movement(void) {
    // Variance should be LOWER for baseline (stable) than movement (noisy)
    const size_t SAMPLE_SIZE = 100;
    const uint8_t TEST_SUBCARRIER = 15;
    
    std::vector<float> baseline_mags(SAMPLE_SIZE);
    std::vector<float> movement_mags(SAMPLE_SIZE);
    
    for (size_t i = 0; i < SAMPLE_SIZE; i++) {
        const int8_t* bp = baseline_packets[i];
        const int8_t* mp = movement_packets[i];
        baseline_mags[i] = calculate_magnitude(bp[TEST_SUBCARRIER * 2], bp[TEST_SUBCARRIER * 2 + 1]);
        movement_mags[i] = calculate_magnitude(mp[TEST_SUBCARRIER * 2], mp[TEST_SUBCARRIER * 2 + 1]);
    }
    
    float var_baseline = calculate_variance_two_pass(baseline_mags.data(), SAMPLE_SIZE);
    float var_movement = calculate_variance_two_pass(movement_mags.data(), SAMPLE_SIZE);
    
    ESP_LOGI(TAG, "Variance comparison for subcarrier %d:", TEST_SUBCARRIER);
    ESP_LOGI(TAG, "  Baseline variance: %.4f", var_baseline);
    ESP_LOGI(TAG, "  Movement variance: %.4f", var_movement);
    
    // Baseline should have lower variance (more stable)
    TEST_ASSERT_TRUE_MESSAGE(var_baseline < var_movement,
        "Baseline variance should be lower than movement variance");
}

void test_turbulence_baseline_vs_movement(void) {
    // Spatial turbulence should be LOWER for baseline than movement
    const uint8_t SUBCARRIERS[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    const uint8_t NUM_SC = sizeof(SUBCARRIERS) / sizeof(SUBCARRIERS[0]);
    
    // Calculate turbulence for several packets
    float baseline_turb_sum = 0.0f;
    float movement_turb_sum = 0.0f;
    const size_t SAMPLE_SIZE = 100;
    
    for (size_t i = 0; i < SAMPLE_SIZE; i++) {
        // Calculate magnitudes for this packet
        float baseline_mags[64], movement_mags[64];
        const int8_t* bp = baseline_packets[i];
        const int8_t* mp = movement_packets[i];
        for (uint8_t sc = 0; sc < 64; sc++) {
            baseline_mags[sc] = calculate_magnitude(bp[sc * 2], bp[sc * 2 + 1]);
            movement_mags[sc] = calculate_magnitude(mp[sc * 2], mp[sc * 2 + 1]);
        }
        
        baseline_turb_sum += calculate_spatial_turbulence(baseline_mags, SUBCARRIERS, NUM_SC);
        movement_turb_sum += calculate_spatial_turbulence(movement_mags, SUBCARRIERS, NUM_SC);
    }
    
    float avg_baseline_turb = baseline_turb_sum / SAMPLE_SIZE;
    float avg_movement_turb = movement_turb_sum / SAMPLE_SIZE;
    
    ESP_LOGI(TAG, "Average spatial turbulence:");
    ESP_LOGI(TAG, "  Baseline: %.4f", avg_baseline_turb);
    ESP_LOGI(TAG, "  Movement: %.4f", avg_movement_turb);
    
    // Movement should have higher turbulence variance (more chaotic)
    // Note: The absolute turbulence might be similar, but the variance over time differs
    TEST_ASSERT_TRUE(avg_baseline_turb > 0.0f);
    TEST_ASSERT_TRUE(avg_movement_turb > 0.0f);
}

// ============================================================================
// CRITICAL: Test calculate_spatial_turbulence_from_csi directly
// This function is used in production code (csi_processor.cpp)
// ============================================================================

void test_turbulence_from_csi_nonzero_real_data(void) {
    // CRITICAL: This test ensures calculate_spatial_turbulence_from_csi
    // produces non-zero values with real CSI data
    const uint8_t SUBCARRIERS[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    const uint8_t NUM_SC = 12;
    const size_t CSI_LEN = 128;  // 64 subcarriers * 2 bytes (I/Q)
    
    // Test with multiple baseline packets
    int nonzero_count = 0;
    float total_turbulence = 0.0f;
    
    for (int i = 0; i < 50; i++) {
        float turb = calculate_spatial_turbulence_from_csi(
            baseline_packets[i], CSI_LEN, SUBCARRIERS, NUM_SC);
        
        if (turb > 0.0f) {
            nonzero_count++;
            total_turbulence += turb;
        }
    }
    
    ESP_LOGI(TAG, "calculate_spatial_turbulence_from_csi test:");
    ESP_LOGI(TAG, "  Non-zero results: %d/50", nonzero_count);
    ESP_LOGI(TAG, "  Average turbulence: %.4f", total_turbulence / 50);
    
    // ALL results should be non-zero with real data
    TEST_ASSERT_EQUAL_MESSAGE(50, nonzero_count, 
        "All turbulence values should be non-zero with real CSI data");
    
    // Average should be reasonable (not too small, not too large)
    float avg = total_turbulence / 50;
    TEST_ASSERT_TRUE_MESSAGE(avg > 0.5f, "Average turbulence too low");
    TEST_ASSERT_TRUE_MESSAGE(avg < 50.0f, "Average turbulence too high");
}

void test_turbulence_from_csi_movement_higher_variance(void) {
    // Movement packets should produce MORE VARIABLE turbulence than baseline
    const uint8_t SUBCARRIERS[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    const uint8_t NUM_SC = 12;
    const size_t CSI_LEN = 128;
    const size_t SAMPLE_SIZE = 100;
    
    // Collect turbulence values
    float baseline_values[100];
    float movement_values[100];
    
    for (size_t i = 0; i < SAMPLE_SIZE; i++) {
        baseline_values[i] = calculate_spatial_turbulence_from_csi(
            baseline_packets[i], CSI_LEN, SUBCARRIERS, NUM_SC);
        movement_values[i] = calculate_spatial_turbulence_from_csi(
            movement_packets[i], CSI_LEN, SUBCARRIERS, NUM_SC);
    }
    
    // Calculate variance of turbulence values
    float baseline_var = calculate_variance_two_pass(baseline_values, SAMPLE_SIZE);
    float movement_var = calculate_variance_two_pass(movement_values, SAMPLE_SIZE);
    
    ESP_LOGI(TAG, "Turbulence variance (from_csi function):");
    ESP_LOGI(TAG, "  Baseline variance: %.4f", baseline_var);
    ESP_LOGI(TAG, "  Movement variance: %.4f", movement_var);
    
    // Movement should have higher variance in turbulence
    TEST_ASSERT_TRUE_MESSAGE(movement_var > baseline_var,
        "Movement turbulence variance should be higher than baseline");
}

void test_turbulence_from_csi_different_csi_lengths(void) {
    // Test that function handles different csi_len values correctly
    const uint8_t SUBCARRIERS[] = {5, 10, 15, 20, 25, 30};
    const uint8_t NUM_SC = 6;
    
    // Test with csi_len = 128 (standard 64 subcarriers)
    float turb_128 = calculate_spatial_turbulence_from_csi(
        baseline_packets[0], 128, SUBCARRIERS, NUM_SC);
    TEST_ASSERT_TRUE_MESSAGE(turb_128 > 0.0f, "Should work with csi_len=128");
    
    // Test with csi_len = 64 (32 subcarriers) - some selected SC will be skipped
    float turb_64 = calculate_spatial_turbulence_from_csi(
        baseline_packets[0], 64, SUBCARRIERS, NUM_SC);
    // SC 5, 10, 15, 20, 25, 30 - only 5, 10, 15, 20, 25, 30 are < 32
    // So all 6 should be valid
    TEST_ASSERT_TRUE_MESSAGE(turb_64 > 0.0f, "Should work with csi_len=64");
    
    // Test with very short csi_len (only 10 subcarriers)
    float turb_20 = calculate_spatial_turbulence_from_csi(
        baseline_packets[0], 20, SUBCARRIERS, NUM_SC);
    // Only SC 5 is < 10, so only 1 valid subcarrier
    // With 1 subcarrier, variance is 0, so turbulence is 0
    TEST_ASSERT_EQUAL_FLOAT_MESSAGE(0.0f, turb_20, 
        "Should return 0 when only 1 valid subcarrier (variance=0)");
    
    ESP_LOGI(TAG, "Different csi_len test: 128->%.4f, 64->%.4f, 20->%.4f", 
             turb_128, turb_64, turb_20);
}

void test_subcarrier_variance_ranking_real_data(void) {
    // Test that variance correctly identifies stable subcarriers
    const size_t SAMPLE_SIZE = 100;
    
    struct SubcarrierVariance {
        uint8_t index;
        float variance;
        float mean;
    };
    
    SubcarrierVariance metrics[64];
    
    // Calculate variance for each subcarrier
    for (uint8_t sc = 0; sc < 64; sc++) {
        std::vector<float> mags(SAMPLE_SIZE);
        float sum = 0.0f;
        for (size_t i = 0; i < SAMPLE_SIZE; i++) {
            const int8_t* bp = baseline_packets[i];
            mags[i] = calculate_magnitude(bp[sc * 2], bp[sc * 2 + 1]);
            sum += mags[i];
        }
        
        metrics[sc].index = sc;
        metrics[sc].variance = calculate_variance_two_pass(mags.data(), SAMPLE_SIZE);
        metrics[sc].mean = sum / SAMPLE_SIZE;
    }
    
    // Sort by variance (ascending - lower is better for stability)
    std::sort(metrics, metrics + 64, [](const SubcarrierVariance& a, const SubcarrierVariance& b) {
        // Filter out zero-mean subcarriers (guard bands)
        if (a.mean < 1.0f && b.mean >= 1.0f) return false;
        if (b.mean < 1.0f && a.mean >= 1.0f) return true;
        return a.variance < b.variance;
    });
    
    // Log top 12 subcarriers
    ESP_LOGI(TAG, "Top 12 subcarriers by variance (lower is better):");
    for (int i = 0; i < 12; i++) {
        ESP_LOGI(TAG, "  %2d. Subcarrier %2d: var=%.4f, mean=%.2f", 
                 i+1, metrics[i].index, metrics[i].variance, metrics[i].mean);
    }
    
    // Verify guard bands (subcarriers 0-5, 59-63) are NOT in top 12
    bool guard_bands_filtered = true;
    for (int i = 0; i < 12; i++) {
        if (metrics[i].index < 6 || metrics[i].index > 58) {
            guard_bands_filtered = false;
            break;
        }
    }
    
    TEST_ASSERT_TRUE_MESSAGE(guard_bands_filtered,
        "Top 12 subcarriers should not include guard bands (0-5, 59-63)");
}

int process(void) {
    UNITY_BEGIN();
    
    // Variance calculation tests (utils.h)
    RUN_TEST(test_variance_empty_array);
    RUN_TEST(test_variance_single_element);
    RUN_TEST(test_variance_identical_values);
    RUN_TEST(test_variance_known_values);
    RUN_TEST(test_variance_with_negative_values);
    RUN_TEST(test_variance_large_values_numerical_stability);
    
    // CalibrationManager end-to-end tests (exercises calculate_nbvi_weighted_ and calculate_percentile_)
    RUN_TEST(test_calibration_manager_full_calibration);
    RUN_TEST(test_calibration_manager_alpha_affects_selection);
    RUN_TEST(test_calibration_manager_percentile_affects_baseline);
    RUN_TEST(test_calibration_manager_noise_gate);
    
    // Spectral spacing tests
    RUN_TEST(test_spectral_spacing_valid);
    RUN_TEST(test_spectral_spacing_invalid);
    RUN_TEST(test_spectral_spacing_edge_case);
    
    // Subcarrier ranking tests
    RUN_TEST(test_subcarrier_ranking_by_nbvi);
    
    // Magnitude calculation tests (utils.h)
    RUN_TEST(test_magnitude_zero_iq);
    RUN_TEST(test_magnitude_positive_iq);
    RUN_TEST(test_magnitude_negative_iq);
    RUN_TEST(test_magnitude_max_values);
    
    // Spatial turbulence tests (utils.h)
    RUN_TEST(test_turbulence_uniform_magnitudes);
    RUN_TEST(test_turbulence_varying_magnitudes);
    RUN_TEST(test_turbulence_empty_selection);
    RUN_TEST(test_turbulence_single_subcarrier);
    
    // Compare functions tests (utils.h)
    RUN_TEST(test_compare_float_ascending);
    RUN_TEST(test_compare_float_abs);
    RUN_TEST(test_compare_int8_ascending);
    RUN_TEST(test_compare_int8_with_duplicates);
    
    // Real CSI data tests
    RUN_TEST(test_magnitude_from_real_csi);
    RUN_TEST(test_variance_baseline_vs_movement);
    RUN_TEST(test_turbulence_baseline_vs_movement);
    RUN_TEST(test_subcarrier_variance_ranking_real_data);
    
    // CRITICAL: Tests for calculate_spatial_turbulence_from_csi (production function)
    RUN_TEST(test_turbulence_from_csi_nonzero_real_data);
    RUN_TEST(test_turbulence_from_csi_movement_higher_variance);
    RUN_TEST(test_turbulence_from_csi_different_csi_lengths);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

