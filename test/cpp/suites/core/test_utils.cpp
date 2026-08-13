/*
 * ESPectre - Utils Unit Tests
 *
 * Tests utility functions: variance, compare, magnitude, turbulence.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "csi_format.h"
#include "utils.h"

using namespace espectre;

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
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
