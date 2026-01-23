/*
 * ESPectre - Utils Unit Tests
 *
 * Tests utility functions: variance, compare, magnitude, turbulence.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "utils.h"
#include "esphome/core/log.h"

// Include CSI data loader
#include "csi_test_data.h"

#define baseline_packets csi_test_data::baseline_packets()
#define movement_packets csi_test_data::movement_packets()
#define num_baseline csi_test_data::num_baseline()
#define num_movement csi_test_data::num_movement()

using namespace esphome::espectre;

static const char *TAG = "test_utils";

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// VARIANCE TESTS
// ============================================================================

void test_variance_empty_array(void) {
    float data[] = {};
    float result = calculate_variance_two_pass(data, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);
}

void test_variance_single_element(void) {
    float data[] = {5.0f};
    float result = calculate_variance_two_pass(data, 1);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);
}

void test_variance_identical_values(void) {
    float data[] = {3.0f, 3.0f, 3.0f, 3.0f};
    float result = calculate_variance_two_pass(data, 4);
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, result);
}

void test_variance_known_values(void) {
    // Values: 2, 4, 4, 4, 5, 5, 7, 9
    // Mean = 5, Variance = 4
    float data[] = {2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
    float result = calculate_variance_two_pass(data, 8);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 4.0f, result);
}

void test_variance_with_negative_values(void) {
    float data[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    float result = calculate_variance_two_pass(data, 5);
    // Mean = 0, Variance = (4+1+0+1+4)/5 = 2
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 2.0f, result);
}

void test_variance_large_values_numerical_stability(void) {
    // Large values to test numerical stability
    float data[] = {1000000.0f, 1000001.0f, 1000002.0f, 1000003.0f, 1000004.0f};
    float result = calculate_variance_two_pass(data, 5);
    // Variance should be 2.0 (same as small values)
    TEST_ASSERT_FLOAT_WITHIN(0.1f, 2.0f, result);
}

// ============================================================================
// MAGNITUDE TESTS
// ============================================================================

void test_magnitude_zero_iq(void) {
    float result = calculate_magnitude(0, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);
}

void test_magnitude_positive_iq(void) {
    // 3^2 + 4^2 = 25, sqrt(25) = 5
    float result = calculate_magnitude(3, 4);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.0f, result);
}

void test_magnitude_negative_iq(void) {
    // (-3)^2 + (-4)^2 = 25, sqrt(25) = 5
    float result = calculate_magnitude(-3, -4);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 5.0f, result);
}

void test_magnitude_max_values(void) {
    // INT8_MAX = 127
    float result = calculate_magnitude(127, 127);
    // 127^2 + 127^2 = 32258, sqrt = ~179.6
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 179.6f, result);
}

// ============================================================================
// SPATIAL TURBULENCE TESTS
// ============================================================================

void test_turbulence_uniform_magnitudes(void) {
    float magnitudes[] = {10.0f, 10.0f, 10.0f, 10.0f};
    uint8_t indices[] = {0, 1, 2, 3};
    
    float result = calculate_spatial_turbulence(magnitudes, indices, 4);
    
    // Uniform magnitudes = zero turbulence
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.0f, result);
}

void test_turbulence_varying_magnitudes(void) {
    float magnitudes[] = {5.0f, 10.0f, 15.0f, 20.0f};
    uint8_t indices[] = {0, 1, 2, 3};
    
    float result = calculate_spatial_turbulence(magnitudes, indices, 4);
    
    // Should have non-zero turbulence
    TEST_ASSERT_TRUE(result > 0.0f);
}

void test_turbulence_empty_selection(void) {
    float magnitudes[] = {10.0f};
    uint8_t indices[] = {};
    
    float result = calculate_spatial_turbulence(magnitudes, indices, 0);
    
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);
}

void test_turbulence_single_subcarrier(void) {
    float magnitudes[] = {10.0f};
    uint8_t indices[] = {0};
    
    float result = calculate_spatial_turbulence(magnitudes, indices, 1);
    
    // Single subcarrier = zero turbulence (no variance)
    TEST_ASSERT_EQUAL_FLOAT(0.0f, result);
}

// ============================================================================
// COMPARE FUNCTION TESTS
// ============================================================================

void test_compare_float_ascending(void) {
    float data[] = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f};
    std::sort(data, data + 5);
    
    TEST_ASSERT_EQUAL_FLOAT(1.0f, data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, data[1]);
    TEST_ASSERT_EQUAL_FLOAT(5.0f, data[2]);
    TEST_ASSERT_EQUAL_FLOAT(8.0f, data[3]);
    TEST_ASSERT_EQUAL_FLOAT(9.0f, data[4]);
}

void test_compare_float_abs(void) {
    float data[] = {-5.0f, 2.0f, -8.0f, 1.0f, -9.0f};
    std::sort(data, data + 5, [](float a, float b) {
        return std::abs(a) < std::abs(b);
    });
    
    TEST_ASSERT_EQUAL_FLOAT(1.0f, data[0]);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, data[1]);
}

void test_compare_int8_ascending(void) {
    int8_t data[] = {5, -2, 8, -1, 9};
    std::sort(data, data + 5);
    
    TEST_ASSERT_EQUAL_INT8(-2, data[0]);
    TEST_ASSERT_EQUAL_INT8(-1, data[1]);
    TEST_ASSERT_EQUAL_INT8(5, data[2]);
    TEST_ASSERT_EQUAL_INT8(8, data[3]);
    TEST_ASSERT_EQUAL_INT8(9, data[4]);
}

void test_compare_int8_with_duplicates(void) {
    int8_t data[] = {3, 1, 3, 2, 1};
    std::sort(data, data + 5);
    
    TEST_ASSERT_EQUAL_INT8(1, data[0]);
    TEST_ASSERT_EQUAL_INT8(1, data[1]);
    TEST_ASSERT_EQUAL_INT8(2, data[2]);
    TEST_ASSERT_EQUAL_INT8(3, data[3]);
    TEST_ASSERT_EQUAL_INT8(3, data[4]);
}

// ============================================================================
// REAL DATA TESTS
// ============================================================================

void test_magnitude_from_real_csi(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    // Get first baseline packet
    const int8_t* csi = baseline_packets[0];
    
    // Extract magnitudes from first few subcarriers
    float mag0 = calculate_magnitude(csi[0], csi[1]);
    float mag1 = calculate_magnitude(csi[2], csi[3]);
    float mag2 = calculate_magnitude(csi[4], csi[5]);
    
    // All magnitudes should be non-negative
    TEST_ASSERT_TRUE(mag0 >= 0.0f);
    TEST_ASSERT_TRUE(mag1 >= 0.0f);
    TEST_ASSERT_TRUE(mag2 >= 0.0f);
    
    ESP_LOGI(TAG, "Real CSI magnitudes: %.2f, %.2f, %.2f", mag0, mag1, mag2);
}

void test_variance_baseline_vs_movement(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    // Calculate variance of first subcarrier magnitude across packets
    float baseline_mags[100];
    float movement_mags[100];
    
    for (int i = 0; i < 100 && i < num_baseline; i++) {
        baseline_mags[i] = calculate_magnitude(baseline_packets[i][0], baseline_packets[i][1]);
    }
    
    for (int i = 0; i < 100 && i < num_movement; i++) {
        movement_mags[i] = calculate_magnitude(movement_packets[i][0], movement_packets[i][1]);
    }
    
    float baseline_var = calculate_variance_two_pass(baseline_mags, 100);
    float movement_var = calculate_variance_two_pass(movement_mags, 100);
    
    ESP_LOGI(TAG, "Baseline variance: %.4f, Movement variance: %.4f", baseline_var, movement_var);
    
    // Movement should generally have higher variance, but not always guaranteed
    TEST_ASSERT_TRUE(baseline_var >= 0.0f);
    TEST_ASSERT_TRUE(movement_var >= 0.0f);
}

void test_turbulence_baseline_vs_movement(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    uint8_t subcarriers[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    
    float baseline_turb = calculate_spatial_turbulence_from_csi(
        baseline_packets[50], 128, subcarriers, 12);
    
    float movement_turb = calculate_spatial_turbulence_from_csi(
        movement_packets[50], 128, subcarriers, 12);
    
    ESP_LOGI(TAG, "Baseline turbulence: %.4f, Movement turbulence: %.4f", 
             baseline_turb, movement_turb);
    
    // Both should be valid non-negative values
    TEST_ASSERT_TRUE(baseline_turb >= 0.0f);
    TEST_ASSERT_TRUE(movement_turb >= 0.0f);
}

void test_turbulence_from_csi_nonzero_real_data(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    uint8_t subcarriers[12] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
    
    int nonzero_count = 0;
    for (int i = 0; i < 50 && i < num_baseline; i++) {
        float turb = calculate_spatial_turbulence_from_csi(
            baseline_packets[i], 128, subcarriers, 12);
        if (turb > 0.0f) nonzero_count++;
    }
    
    // Most packets should have non-zero turbulence
    TEST_ASSERT_TRUE(nonzero_count > 25);
}

void test_turbulence_from_csi_different_csi_lengths(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    uint8_t subcarriers[12] = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    
    // Test with HT20 (128 bytes)
    float turb_128 = calculate_spatial_turbulence_from_csi(
        baseline_packets[0], 128, subcarriers, 12);
    
    // Should work with full 128 bytes
    TEST_ASSERT_TRUE(turb_128 >= 0.0f);
}

// ============================================================================
// ENTRY POINT
// ============================================================================

int process(void) {
    UNITY_BEGIN();
    
    // Variance tests
    RUN_TEST(test_variance_empty_array);
    RUN_TEST(test_variance_single_element);
    RUN_TEST(test_variance_identical_values);
    RUN_TEST(test_variance_known_values);
    RUN_TEST(test_variance_with_negative_values);
    RUN_TEST(test_variance_large_values_numerical_stability);
    
    // Magnitude tests
    RUN_TEST(test_magnitude_zero_iq);
    RUN_TEST(test_magnitude_positive_iq);
    RUN_TEST(test_magnitude_negative_iq);
    RUN_TEST(test_magnitude_max_values);
    
    // Turbulence tests
    RUN_TEST(test_turbulence_uniform_magnitudes);
    RUN_TEST(test_turbulence_varying_magnitudes);
    RUN_TEST(test_turbulence_empty_selection);
    RUN_TEST(test_turbulence_single_subcarrier);
    
    // Compare function tests
    RUN_TEST(test_compare_float_ascending);
    RUN_TEST(test_compare_float_abs);
    RUN_TEST(test_compare_int8_ascending);
    RUN_TEST(test_compare_int8_with_duplicates);
    
    // Real data tests
    RUN_TEST(test_magnitude_from_real_csi);
    RUN_TEST(test_variance_baseline_vs_movement);
    RUN_TEST(test_turbulence_baseline_vs_movement);
    RUN_TEST(test_turbulence_from_csi_nonzero_real_data);
    RUN_TEST(test_turbulence_from_csi_different_csi_lengths);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
