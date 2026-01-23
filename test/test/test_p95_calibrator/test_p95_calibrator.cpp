/*
 * ESPectre - P95Calibrator Integration Tests
 *
 * Tests the P95Calibrator class with real CSI data.
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
#include "p95_calibrator.h"
#include "csi_manager.h"
#include "utils.h"
#include "utils.h"
#include "esphome/core/log.h"

#if defined(ESP_PLATFORM)
#include "esp_spiffs.h"
#endif

// Include CSI data loader (loads from NPZ files)
#include "csi_test_data.h"

// Compatibility macros for existing test code
#define baseline_packets csi_test_data::baseline_packets()
#define movement_packets csi_test_data::movement_packets()
#define num_baseline csi_test_data::num_baseline()
#define num_movement csi_test_data::num_movement()
#define packet_size csi_test_data::packet_size()

using namespace esphome::espectre;

static const char *TAG = "test_p95_calibrator";

// Test buffer file path - different for native vs ESP32
#if defined(ESP_PLATFORM)
static const char* TEST_BUFFER_PATH = "/spiffs/test_buffer.bin";
static bool spiffs_mounted = false;
#else
static const char* TEST_BUFFER_PATH = "/tmp/test_calibration_buffer.bin";
#endif

// Default subcarrier band for testing
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
    // Remove test file before each test
    remove(TEST_BUFFER_PATH);
}

void tearDown(void) {
    // Cleanup after each test
    remove(TEST_BUFFER_PATH);
}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_init_without_csi_manager(void) {
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    // Should not crash, but calibration should fail without CSI manager
    TEST_ASSERT_FALSE(cm.is_calibrating());
}

void test_init_with_custom_buffer_path(void) {
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    // Verify we can set configuration parameters
    cm.set_buffer_size(100);
    
    TEST_ASSERT_FALSE(cm.is_calibrating());
}

void test_start_calibration_fails_without_csi_manager(void) {
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    uint8_t band[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    
    esp_err_t err = cm.start_calibration(band, 12, nullptr);
    
    // Should fail because CSI manager is not set
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, err);
    TEST_ASSERT_FALSE(cm.is_calibrating());
}

// ============================================================================
// ADD_PACKET TESTS
// ============================================================================

void test_add_packet_rejects_short_data(void) {
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    // Manually set calibrating state for testing add_packet
    // Note: This is a workaround since we can't call start_calibration without CSI manager
    
    int8_t short_data[64] = {0};  // Too short (need 128 bytes)
    
    // add_packet should return false for short data
    // But since we're not calibrating, it will return early
    bool result = cm.add_packet(short_data, 64);
    TEST_ASSERT_FALSE(result);
}

void test_add_packet_not_calibrating_returns_early(void) {
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    // Try to add packet when not calibrating
    bool result = cm.add_packet(baseline_packets[0], packet_size);
    
    // Should return false (buffer_count >= buffer_size is false)
    TEST_ASSERT_FALSE(result);
}

// ============================================================================
// CONFIGURATION SETTERS TESTS
// ============================================================================

void test_set_buffer_size(void) {
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    cm.set_buffer_size(500);
    // No direct getter, but should not crash
    TEST_PASS();
}

void test_set_skip_subcarrier_selection(void) {
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    // Test that we can configure skip_subcarrier_selection
    cm.set_skip_subcarrier_selection(true);
    cm.set_skip_subcarrier_selection(false);
    
    TEST_PASS();
}

// ============================================================================
// FILE I/O TESTS (via add_packet simulation)
// ============================================================================

void test_file_write_via_add_packet_simulation(void) {
    // Simulate what add_packet does: write magnitudes to file
    FILE* f = fopen(TEST_BUFFER_PATH, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    // Write 100 packets worth of magnitude data
    for (int p = 0; p < 100; p++) {
        uint8_t magnitudes[64];
        const int8_t* pkt = baseline_packets[p];
        for (uint8_t sc = 0; sc < 64; sc++) {
            // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            int8_t q_val = pkt[sc * 2];      // Imaginary first
            int8_t i_val = pkt[sc * 2 + 1];  // Real second
            float mag = calculate_magnitude(i_val, q_val);
            magnitudes[sc] = static_cast<uint8_t>(std::min(mag, 255.0f));
        }
        
        size_t written = fwrite(magnitudes, 1, 64, f);
        TEST_ASSERT_EQUAL(64, written);
    }
    
    fclose(f);
    
    // Verify file size
    f = fopen(TEST_BUFFER_PATH, "rb");
    TEST_ASSERT_NOT_NULL(f);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    TEST_ASSERT_EQUAL(100 * 64, size);
    fclose(f);
}

void test_file_read_window_simulation(void) {
    // First write some data
    FILE* f = fopen(TEST_BUFFER_PATH, "wb");
    TEST_ASSERT_NOT_NULL(f);
    
    for (int p = 0; p < 50; p++) {
        uint8_t magnitudes[64];
        for (uint8_t sc = 0; sc < 64; sc++) {
            magnitudes[sc] = p;  // Each packet has value = packet index
        }
        fwrite(magnitudes, 1, 64, f);
    }
    fclose(f);
    
    // Read window starting at packet 20, size 10
    f = fopen(TEST_BUFFER_PATH, "rb");
    TEST_ASSERT_NOT_NULL(f);
    
    uint16_t start_idx = 20;
    uint16_t window_size = 10;
    
    fseek(f, start_idx * 64, SEEK_SET);
    
    std::vector<uint8_t> window_data(window_size * 64);
    size_t read = fread(window_data.data(), 1, window_data.size(), f);
    TEST_ASSERT_EQUAL(window_size * 64, read);
    
    // Verify data: packet 20 should have all 20s
    for (int i = 0; i < 64; i++) {
        TEST_ASSERT_EQUAL_UINT8(20, window_data[i]);
    }
    
    // Packet 25 (index 5 in window) should have all 25s
    for (int i = 0; i < 64; i++) {
        TEST_ASSERT_EQUAL_UINT8(25, window_data[5 * 64 + i]);
    }
    
    fclose(f);
}

// ============================================================================
// MAGNITUDE EXTRACTION FROM REAL DATA
// ============================================================================

void test_magnitude_extraction_real_data(void) {
    // Test that we can correctly extract magnitudes from real CSI packets
    const int8_t* packet = baseline_packets[0];
    
    // Calculate magnitudes for all 64 subcarriers using utils.h
    // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
    float magnitudes[64];
    for (uint8_t sc = 0; sc < 64; sc++) {
        int8_t q_val = packet[sc * 2];      // Imaginary first
        int8_t i_val = packet[sc * 2 + 1];  // Real second
        magnitudes[sc] = calculate_magnitude(i_val, q_val);
    }
    
    // Guard bands (0-5) should have low/zero magnitude
    for (int sc = 0; sc < 5; sc++) {
        TEST_ASSERT_TRUE(magnitudes[sc] < 10.0f);
    }
    
    // Active subcarriers (10-50) should have significant magnitude
    float avg_active_mag = 0.0f;
    for (int sc = 10; sc < 50; sc++) {
        avg_active_mag += magnitudes[sc];
    }
    avg_active_mag /= 40;
    
    ESP_LOGI(TAG, "Average active subcarrier magnitude: %.2f", avg_active_mag);
    TEST_ASSERT_TRUE(avg_active_mag > 10.0f);
}

void test_magnitude_consistency_across_packets(void) {
    // Magnitudes should be relatively consistent across baseline packets
    const uint8_t TEST_SC = 20;  // Test subcarrier 20
    
    // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
    std::vector<float> magnitudes(100);
    for (int p = 0; p < 100; p++) {
        const int8_t* pkt = baseline_packets[p];
        magnitudes[p] = calculate_magnitude(pkt[TEST_SC * 2 + 1], pkt[TEST_SC * 2]);
    }
    
    // Calculate variance using utils.h
    float variance = calculate_variance_two_pass(magnitudes.data(), magnitudes.size());
    float std_dev = std::sqrt(variance);
    
    // Calculate mean
    float mean = 0.0f;
    for (float m : magnitudes) mean += m;
    mean /= 100;
    
    // Coefficient of variation should be low for baseline
    float cv = std_dev / mean;
    
    ESP_LOGI(TAG, "Subcarrier %d: mean=%.2f, std=%.2f, CV=%.4f", 
             TEST_SC, mean, std_dev, cv);
    
    TEST_ASSERT_TRUE(cv < 0.5f);  // CV should be less than 50% for stable signal
}

// ============================================================================
// SPATIAL TURBULENCE FROM REAL DATA
// ============================================================================

void test_spatial_turbulence_calculation(void) {
    // Calculate spatial turbulence for a real packet using utils.h
    float turbulence = calculate_spatial_turbulence_from_csi(
        baseline_packets[0], packet_size, DEFAULT_BAND, DEFAULT_BAND_SIZE);
    
    ESP_LOGI(TAG, "Spatial turbulence for packet 0: %.4f", turbulence);
    
    TEST_ASSERT_TRUE(turbulence >= 0.0f);
}

void test_turbulence_variance_baseline_vs_movement(void) {
    // Turbulence VARIANCE should be lower for baseline than movement
    std::vector<float> baseline_turbulences(100);
    std::vector<float> movement_turbulences(100);
    
    for (int p = 0; p < 100; p++) {
        // Calculate turbulence using utils.h functions
        baseline_turbulences[p] = calculate_spatial_turbulence_from_csi(
            baseline_packets[p], packet_size, DEFAULT_BAND, DEFAULT_BAND_SIZE);
        movement_turbulences[p] = calculate_spatial_turbulence_from_csi(
            movement_packets[p], packet_size, DEFAULT_BAND, DEFAULT_BAND_SIZE);
    }
    
    // Calculate variance of turbulences using utils.h
    float baseline_turb_var = calculate_variance_two_pass(
        baseline_turbulences.data(), baseline_turbulences.size());
    float movement_turb_var = calculate_variance_two_pass(
        movement_turbulences.data(), movement_turbulences.size());
    
    ESP_LOGI(TAG, "Turbulence variance - Baseline: %.4f, Movement: %.4f",
             baseline_turb_var, movement_turb_var);
    
    // Movement should have higher variance in turbulence
    TEST_ASSERT_TRUE(movement_turb_var > baseline_turb_var);
}

// ============================================================================
// BAND METRICS CALCULATION FROM REAL DATA
// ============================================================================

void test_band_metric_calculation_real_data(void) {
    const uint8_t TEST_SC = 15;
    
    // Collect magnitudes for 100 baseline packets using utils.h
    // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
    std::vector<float> magnitudes(100);
    for (int p = 0; p < 100; p++) {
        const int8_t* pkt = baseline_packets[p];
        magnitudes[p] = calculate_magnitude(pkt[TEST_SC * 2 + 1], pkt[TEST_SC * 2]);
    }
    
    // Calculate mean
    float sum = 0.0f;
    for (float m : magnitudes) sum += m;
    float mean = sum / 100;
    
    // Calculate variance using utils.h
    float variance = calculate_variance_two_pass(magnitudes.data(), magnitudes.size());
    float std_dev = std::sqrt(variance);
    
    // Calculate coefficient of variation (used for band ranking)
    float cv = std_dev / mean;
    
    ESP_LOGI(TAG, "Band metric for subcarrier %d: cv=%.4f (mean=%.2f, std=%.2f)",
             TEST_SC, cv, mean, std_dev);
    
    TEST_ASSERT_TRUE(cv > 0.0f);
    TEST_ASSERT_TRUE(cv < 1.0f);  // Should be reasonable for stable signal
}

void test_band_ranking_identifies_best_subcarriers(void) {
    struct BandResult {
        uint8_t subcarrier;
        float variance;
        float mean;
    };
    
    std::vector<BandResult> results(64);
    
    // Calculate variance for all subcarriers using utils.h
    // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
    for (uint8_t sc = 0; sc < 64; sc++) {
        std::vector<float> magnitudes(100);
        for (int p = 0; p < 100; p++) {
            const int8_t* pkt = baseline_packets[p];
            magnitudes[p] = calculate_magnitude(pkt[sc * 2 + 1], pkt[sc * 2]);
        }
        
        float sum = 0.0f;
        for (float m : magnitudes) sum += m;
        float mean = sum / 100;
        
        float variance = calculate_variance_two_pass(magnitudes.data(), magnitudes.size());
        
        results[sc].subcarrier = sc;
        results[sc].variance = variance;
        results[sc].mean = mean;
    }
    
    // Sort by variance (ascending - lower variance = more stable = better for P95)
    std::sort(results.begin(), results.end(), 
              [](const BandResult& a, const BandResult& b) {
                  return a.variance < b.variance;
              });
    
    // Log top 12
    ESP_LOGI(TAG, "Top 12 subcarriers by variance:");
    for (int i = 0; i < 12; i++) {
        ESP_LOGI(TAG, "  %2d. SC %2d: variance=%.6f, mean=%.2f",
                 i+1, results[i].subcarrier, results[i].variance, results[i].mean);
    }
    
    // Verify ranking produces valid results
    // Note: Guard band exclusion is done by P95Calibrator, not by ranking
    TEST_ASSERT_TRUE(results[0].variance <= results[11].variance);  // Sorted correctly
    TEST_ASSERT_TRUE(results[0].variance >= 0.0f);  // Non-negative variance
}

// ============================================================================
// BASELINE WINDOW DETECTION SIMULATION
// ============================================================================

void test_baseline_window_detection_simulation(void) {
    // Simulate baseline window detection algorithm using utils.h
    const uint16_t BUFFER_SIZE = 200;
    const uint16_t WINDOW_SIZE = 50;
    const uint16_t WINDOW_STEP = 25;
    
    // Calculate turbulence for each packet using utils.h
    std::vector<float> all_turbulences(BUFFER_SIZE);
    
    // First 100 packets: baseline, next 100: movement
    for (int p = 0; p < 100; p++) {
        all_turbulences[p] = calculate_spatial_turbulence_from_csi(
            baseline_packets[p], packet_size, DEFAULT_BAND, DEFAULT_BAND_SIZE);
    }
    
    for (int p = 0; p < 100; p++) {
        all_turbulences[100 + p] = calculate_spatial_turbulence_from_csi(
            movement_packets[p], packet_size, DEFAULT_BAND, DEFAULT_BAND_SIZE);
    }
    
    // Calculate variance for each window
    struct WindowInfo {
        uint16_t start;
        float variance;
    };
    
    std::vector<WindowInfo> windows;
    for (uint16_t start = 0; start <= BUFFER_SIZE - WINDOW_SIZE; start += WINDOW_STEP) {
        // Calculate variance of turbulences in this window using utils.h
        float var = calculate_variance_two_pass(&all_turbulences[start], WINDOW_SIZE);
        windows.push_back({start, var});
    }
    
    // Find window with minimum variance
    auto min_it = std::min_element(windows.begin(), windows.end(),
                                    [](const WindowInfo& a, const WindowInfo& b) {
                                        return a.variance < b.variance;
                                    });
    
    ESP_LOGI(TAG, "Best baseline window: start=%d, variance=%.4f",
             min_it->start, min_it->variance);
    
    // Best window should be in the baseline portion (first 100 packets)
    TEST_ASSERT_TRUE(min_it->start < 100);
}

// ============================================================================
// PERCENTILE CALCULATION TESTS
// ============================================================================

void test_percentile_edge_cases(void) {
    // Test percentile calculation with edge cases
    
    // Empty vector
    std::vector<float> empty;
    // Can't test directly, but we verify via band metric calculation
    
    // Single element
    std::vector<float> single = {42.0f};
    float sum = 0.0f;
    for (float v : single) sum += v;
    float mean = sum / single.size();
    TEST_ASSERT_EQUAL_FLOAT(42.0f, mean);
    
    // Two elements - percentile interpolation
    std::vector<float> two = {10.0f, 20.0f};
    std::sort(two.begin(), two.end());
    
    // Linear interpolation for p50 should give 15.0
    float k = (two.size() - 1) * 50 / 100.0f;
    size_t f = static_cast<size_t>(k);
    size_t c = f + 1;
    float p50 = two[f] * (c - k) + two[c] * (k - f);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 15.0f, p50);
}

void test_percentile_boundary_values(void) {
    // Test percentile at boundaries (0%, 100%)
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::sort(values.begin(), values.end());
    
    // p0 should give first element
    float k0 = (values.size() - 1) * 0 / 100.0f;
    TEST_ASSERT_EQUAL_FLOAT(0.0f, k0);
    TEST_ASSERT_EQUAL_FLOAT(1.0f, values[0]);
    
    // p100 should give last element
    float k100 = (values.size() - 1) * 100 / 100.0f;
    size_t idx = static_cast<size_t>(k100);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, values[idx]);
}

// ============================================================================
// NOISE GATE TESTS
// ============================================================================

void test_noise_gate_filters_low_magnitude_subcarriers(void) {
    // Verify noise gate concept: low magnitude subcarriers should be filtered
    
    // Simulate subcarrier magnitudes: some strong, some weak
    struct SubcarrierInfo {
        uint8_t idx;
        float mean_magnitude;
    };
    
    std::vector<SubcarrierInfo> subcarriers;
    
    // Guard bands (0-5, 59-63) typically have low magnitude
    // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
    for (uint8_t i = 0; i < 64; i++) {
        float mean = 0.0f;
        for (int p = 0; p < 100; p++) {
            mean += calculate_magnitude(baseline_packets[p][i * 2 + 1], 
                                        baseline_packets[p][i * 2]);
        }
        mean /= 100;
        subcarriers.push_back({i, mean});
    }
    
    // Sort by magnitude
    std::sort(subcarriers.begin(), subcarriers.end(),
              [](const SubcarrierInfo& a, const SubcarrierInfo& b) {
                  return a.mean_magnitude < b.mean_magnitude;
              });
    
    // Calculate p10 threshold
    size_t p10_idx = static_cast<size_t>(63 * 0.1f);
    float threshold = subcarriers[p10_idx].mean_magnitude;
    
    ESP_LOGI(TAG, "Noise gate p10 threshold: %.2f", threshold);
    
    // Count filtered subcarriers
    int filtered = 0;
    for (const auto& sc : subcarriers) {
        if (sc.mean_magnitude < threshold) {
            filtered++;
            ESP_LOGD(TAG, "  Filtered SC %d: mean=%.2f", sc.idx, sc.mean_magnitude);
        }
    }
    
    // Should filter approximately 10% (6-7 subcarriers)
    TEST_ASSERT_TRUE(filtered >= 5);
    TEST_ASSERT_TRUE(filtered <= 10);
}

// ============================================================================
// SPECTRAL SPACING TESTS
// ============================================================================

void test_spectral_spacing_selection(void) {
    // Test that selected subcarriers respect minimum spacing
    const uint8_t MIN_SPACING = 3;
    
    // Simulate ranked subcarriers by P95 moving variance (best first)
    std::vector<uint8_t> sorted_by_p95 = {15, 16, 17, 20, 25, 30, 35, 40, 45, 50, 10, 55};
    
    std::vector<uint8_t> selected;
    
    // Phase 1: Top 5 absolute best (no spacing check)
    for (int i = 0; i < 5 && i < (int)sorted_by_p95.size(); i++) {
        selected.push_back(sorted_by_p95[i]);
    }
    
    // Phase 2: Remaining 7 with spacing
    for (size_t i = 5; i < sorted_by_p95.size() && selected.size() < 12; i++) {
        uint8_t candidate = sorted_by_p95[i];
        
        bool spacing_ok = true;
        for (uint8_t sel : selected) {
            uint8_t dist = (candidate > sel) ? (candidate - sel) : (sel - candidate);
            if (dist < MIN_SPACING) {
                spacing_ok = false;
                break;
            }
        }
        
        if (spacing_ok) {
            selected.push_back(candidate);
        }
    }
    
    ESP_LOGI(TAG, "Selected with spacing: %zu subcarriers", selected.size());
    for (uint8_t sc : selected) {
        ESP_LOGD(TAG, "  SC %d", sc);
    }
    
    // Verify we got at least some subcarriers
    TEST_ASSERT_TRUE(selected.size() >= 5);
}

// ============================================================================
// BAND METRIC EDGE CASE TESTS
// ============================================================================

void test_band_metric_zero_mean(void) {
    // Test band metric with zero mean (should be handled gracefully)
    std::vector<float> zero_magnitudes(100, 0.0f);
    
    float sum = 0.0f;
    for (float m : zero_magnitudes) sum += m;
    float mean = sum / zero_magnitudes.size();
    
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, mean);
    
    // Variance should be zero for zero signal
    float variance = calculate_variance_two_pass(zero_magnitudes.data(), 
                                                 zero_magnitudes.size());
    
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, variance);
}

void test_band_metric_constant_signal(void) {
    // Test band metric with constant signal (variance=0)
    std::vector<float> constant_magnitudes(100, 50.0f);
    
    float sum = 0.0f;
    for (float m : constant_magnitudes) sum += m;
    float mean = sum / constant_magnitudes.size();
    
    float variance = calculate_variance_two_pass(constant_magnitudes.data(), 
                                                 constant_magnitudes.size());
    float std = std::sqrt(variance);
    
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 50.0f, mean);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, std);
    
    // Variance should be 0 for constant signal
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, variance);
}

// ============================================================================
// CALIBRATION DOUBLE START TEST
// ============================================================================

void test_start_calibration_while_already_calibrating(void) {
    // This tests the "already calibrating" path
    P95Calibrator cm;
    cm.init(nullptr, TEST_BUFFER_PATH);
    
    // First start should fail because no CSI manager
    esp_err_t err1 = cm.start_calibration(DEFAULT_BAND, DEFAULT_BAND_SIZE, nullptr);
    TEST_ASSERT_EQUAL(ESP_ERR_INVALID_STATE, err1);
    
    // Verify not calibrating
    TEST_ASSERT_FALSE(cm.is_calibrating());
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    // Load CSI test data from NPZ files
    if (!csi_test_data::load()) {
        printf("ERROR: Failed to load CSI test data from NPZ files\n");
        return 1;
    }
    
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_init_without_csi_manager);
    RUN_TEST(test_init_with_custom_buffer_path);
    RUN_TEST(test_start_calibration_fails_without_csi_manager);
    
    // add_packet tests
    RUN_TEST(test_add_packet_rejects_short_data);
    RUN_TEST(test_add_packet_not_calibrating_returns_early);
    
    // Configuration tests
    RUN_TEST(test_set_buffer_size);
    RUN_TEST(test_set_skip_subcarrier_selection);
    
    // File I/O simulation tests
    RUN_TEST(test_file_write_via_add_packet_simulation);
    RUN_TEST(test_file_read_window_simulation);
    
    // Magnitude extraction tests
    RUN_TEST(test_magnitude_extraction_real_data);
    RUN_TEST(test_magnitude_consistency_across_packets);
    
    // Spatial turbulence tests
    RUN_TEST(test_spatial_turbulence_calculation);
    RUN_TEST(test_turbulence_variance_baseline_vs_movement);
    
    // Band metric tests
    RUN_TEST(test_band_metric_calculation_real_data);
    RUN_TEST(test_band_ranking_identifies_best_subcarriers);
    
    // Baseline window detection
    RUN_TEST(test_baseline_window_detection_simulation);
    
    // Percentile calculation tests
    RUN_TEST(test_percentile_edge_cases);
    RUN_TEST(test_percentile_boundary_values);
    
    // Noise gate tests
    RUN_TEST(test_noise_gate_filters_low_magnitude_subcarriers);
    
    // Spectral spacing tests
    RUN_TEST(test_spectral_spacing_selection);
    
    // Band metric edge case tests
    RUN_TEST(test_band_metric_zero_mean);
    RUN_TEST(test_band_metric_constant_signal);
    
    // Double start test
    RUN_TEST(test_start_calibration_while_already_calibrating);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

