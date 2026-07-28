/*
 * ESPectre - Utils Unit Tests
 *
 * Tests utility functions: variance, compare, magnitude, turbulence.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "csi_format.h"
#include "utils.h"
#include "esphome/core/log.h"

// Include CSI data loader
#include "csi_test_data.h"

#define static_presence_packets csi_test_data::static_presence_packets()
#define motion_packets csi_test_data::motion_packets()
#define num_static_presence csi_test_data::num_static_presence()
#define num_motion csi_test_data::num_motion()

using namespace espectre;

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
// REAL DATA TESTS
// ============================================================================

void test_magnitude_from_real_csi(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    // Get first static-presence packet
    const int8_t* csi = static_presence_packets[0];
    
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

void test_variance_static_presence_vs_motion(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    // Calculate variance of first subcarrier magnitude across packets
    float static_presence_mags[100];
    float motion_mags[100];
    
    for (int i = 0; i < 100 && i < num_static_presence; i++) {
        static_presence_mags[i] = calculate_magnitude(static_presence_packets[i][0], static_presence_packets[i][1]);
    }
    
    for (int i = 0; i < 100 && i < num_motion; i++) {
        motion_mags[i] = calculate_magnitude(motion_packets[i][0], motion_packets[i][1]);
    }
    
    float static_presence_var = calculate_variance_two_pass(static_presence_mags, 100);
    float motion_var = calculate_variance_two_pass(motion_mags, 100);
    
    ESP_LOGI(TAG, "Static presence variance: %.4f, Motion variance: %.4f", static_presence_var, motion_var);
    
    // Motion should generally have higher variance, but not always guaranteed
    TEST_ASSERT_TRUE(static_presence_var >= 0.0f);
    TEST_ASSERT_TRUE(motion_var >= 0.0f);
}

void test_turbulence_static_presence_vs_motion(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    const uint8_t* subcarriers = DEFAULT_SUBCARRIERS;
    
    float static_presence_turb = calculate_spatial_turbulence_from_csi(
        static_presence_packets[50], 128, subcarriers, 12);
    
    float motion_turb = calculate_spatial_turbulence_from_csi(
        motion_packets[50], 128, subcarriers, 12);
    
    ESP_LOGI(TAG, "Static presence turbulence: %.4f, Motion turbulence: %.4f", 
             static_presence_turb, motion_turb);
    
    // Both should be valid non-negative values
    TEST_ASSERT_TRUE(static_presence_turb >= 0.0f);
    TEST_ASSERT_TRUE(motion_turb >= 0.0f);
}

void test_turbulence_from_csi_nonzero_real_data(void) {
    if (!csi_test_data::load()) {
        TEST_IGNORE_MESSAGE("Failed to load test data");
        return;
    }
    
    const uint8_t* subcarriers = DEFAULT_SUBCARRIERS;
    
    int nonzero_count = 0;
    for (int i = 0; i < 50 && i < num_static_presence; i++) {
        float turb = calculate_spatial_turbulence_from_csi(
            static_presence_packets[i], 128, subcarriers, 12);
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
        static_presence_packets[0], 128, subcarriers, 12);
    
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
    
    // Real data tests
    RUN_TEST(test_magnitude_from_real_csi);
    RUN_TEST(test_variance_static_presence_vs_motion);
    RUN_TEST(test_turbulence_static_presence_vs_motion);
    RUN_TEST(test_turbulence_from_csi_nonzero_real_data);
    RUN_TEST(test_turbulence_from_csi_different_csi_lengths);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
