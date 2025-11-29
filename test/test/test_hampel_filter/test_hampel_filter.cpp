/*
 * ESPectre - Hampel Filter Unit Tests
 *
 * Unit tests for Hampel outlier removal filter used in MVS preprocessing
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cmath>
#include "csi_processor.h"
#include "esp_log.h"

using namespace esphome::espectre;

static const char *TAG = "test_hampel";

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_hampel_init_default_values(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, HAMPEL_TURBULENCE_WINDOW_DEFAULT, 
                          HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, true);
    
    TEST_ASSERT_EQUAL(HAMPEL_TURBULENCE_WINDOW_DEFAULT, state.window_size);
    TEST_ASSERT_EQUAL_FLOAT(HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, state.threshold);
    TEST_ASSERT_TRUE(state.enabled);
    TEST_ASSERT_EQUAL(0, state.count);
    TEST_ASSERT_EQUAL(0, state.index);
}

void test_hampel_init_minimum_window(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, HAMPEL_TURBULENCE_WINDOW_MIN, 3.0f, true);
    
    TEST_ASSERT_EQUAL(HAMPEL_TURBULENCE_WINDOW_MIN, state.window_size);
}

void test_hampel_init_maximum_window(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, HAMPEL_TURBULENCE_WINDOW_MAX, 3.0f, true);
    
    TEST_ASSERT_EQUAL(HAMPEL_TURBULENCE_WINDOW_MAX, state.window_size);
}

void test_hampel_init_invalid_window_uses_default(void) {
    hampel_turbulence_state_t state;
    
    // Too small
    hampel_turbulence_init(&state, 1, 3.0f, true);
    TEST_ASSERT_EQUAL(HAMPEL_TURBULENCE_WINDOW_DEFAULT, state.window_size);
    
    // Too large
    hampel_turbulence_init(&state, 20, 3.0f, true);
    TEST_ASSERT_EQUAL(HAMPEL_TURBULENCE_WINDOW_DEFAULT, state.window_size);
}

void test_hampel_init_disabled(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 7, 3.0f, false);
    
    TEST_ASSERT_FALSE(state.enabled);
}

// ============================================================================
// FILTER BEHAVIOR TESTS
// ============================================================================

void test_hampel_disabled_returns_raw_value(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 7, 3.0f, false);  // disabled
    
    float result = hampel_filter_turbulence(&state, 100.0f);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, result);
    
    result = hampel_filter_turbulence(&state, 999.0f);
    TEST_ASSERT_EQUAL_FLOAT(999.0f, result);
}

void test_hampel_returns_raw_until_buffer_fills(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 5, 3.0f, true);
    
    // First 2 values should be returned as-is (need 3 for filtering)
    float result1 = hampel_filter_turbulence(&state, 10.0f);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, result1);
    TEST_ASSERT_EQUAL(1, state.count);
    
    float result2 = hampel_filter_turbulence(&state, 11.0f);
    TEST_ASSERT_EQUAL_FLOAT(11.0f, result2);
    TEST_ASSERT_EQUAL(2, state.count);
}

void test_hampel_passes_normal_values(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 5, 3.0f, true);
    
    // Fill buffer with stable values
    hampel_filter_turbulence(&state, 10.0f);
    hampel_filter_turbulence(&state, 10.0f);
    hampel_filter_turbulence(&state, 10.0f);
    hampel_filter_turbulence(&state, 10.0f);
    
    // Normal value within threshold should pass through
    float result = hampel_filter_turbulence(&state, 10.5f);
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 10.5f, result);
}

void test_hampel_replaces_outlier(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 5, 3.0f, true);
    
    // Fill buffer with stable values around 10
    hampel_filter_turbulence(&state, 10.0f);
    hampel_filter_turbulence(&state, 10.0f);
    hampel_filter_turbulence(&state, 10.0f);
    hampel_filter_turbulence(&state, 10.0f);
    hampel_filter_turbulence(&state, 10.0f);
    
    // Extreme outlier should be replaced with median
    float result = hampel_filter_turbulence(&state, 1000.0f);
    
    // Result should be close to median (10.0), not the outlier (1000.0)
    TEST_ASSERT_FLOAT_WITHIN(5.0f, 10.0f, result);
}

void test_hampel_circular_buffer_wraps(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 5, 3.0f, true);
    
    // Fill buffer completely
    for (int i = 0; i < 5; i++) {
        hampel_filter_turbulence(&state, 10.0f);
    }
    TEST_ASSERT_EQUAL(5, state.count);
    
    // Add more values - buffer should wrap
    for (int i = 0; i < 10; i++) {
        hampel_filter_turbulence(&state, 11.0f);
    }
    
    // Count should stay at window_size
    TEST_ASSERT_EQUAL(5, state.count);
}

// ============================================================================
// STANDALONE HAMPEL_FILTER FUNCTION TESTS
// ============================================================================

void test_hampel_filter_with_null_window(void) {
    float result = hampel_filter(NULL, 5, 10.0f, 3.0f);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, result);  // Returns input unchanged
}

void test_hampel_filter_with_small_window(void) {
    float window[] = {10.0f, 11.0f};
    float result = hampel_filter(window, 2, 10.5f, 3.0f);
    TEST_ASSERT_EQUAL_FLOAT(10.5f, result);  // Returns input unchanged (window < 3)
}

void test_hampel_filter_detects_outlier(void) {
    float window[] = {10.0f, 10.0f, 10.0f, 10.0f, 10.0f};
    
    // Extreme outlier
    float result = hampel_filter(window, 5, 100.0f, 3.0f);
    
    // Should return median (10.0), not the outlier
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 10.0f, result);
}

void test_hampel_filter_passes_normal_value(void) {
    float window[] = {9.0f, 10.0f, 11.0f, 10.0f, 10.0f};
    
    // Normal value close to median (with some variance in window)
    float result = hampel_filter(window, 5, 10.5f, 3.0f);
    
    // Should pass through unchanged (within tolerance of median)
    TEST_ASSERT_FLOAT_WITHIN(1.0f, 10.5f, result);
}

// ============================================================================
// EDGE CASES
// ============================================================================

void test_hampel_with_all_same_values(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 5, 3.0f, true);
    
    // All same values - MAD will be 0
    for (int i = 0; i < 5; i++) {
        hampel_filter_turbulence(&state, 10.0f);
    }
    
    // Any different value might be considered outlier when MAD=0
    // Implementation should handle this gracefully
    float result = hampel_filter_turbulence(&state, 10.0f);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, result);
}

void test_hampel_with_varying_values(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 7, 3.0f, true);
    
    // Natural variation
    float values[] = {10.0f, 11.0f, 9.5f, 10.5f, 10.2f, 9.8f, 10.3f};
    for (int i = 0; i < 7; i++) {
        hampel_filter_turbulence(&state, values[i]);
    }
    
    // Value within normal range should pass
    float result = hampel_filter_turbulence(&state, 10.1f);
    TEST_ASSERT_FLOAT_WITHIN(2.0f, 10.1f, result);
}

int process(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_hampel_init_default_values);
    RUN_TEST(test_hampel_init_minimum_window);
    RUN_TEST(test_hampel_init_maximum_window);
    RUN_TEST(test_hampel_init_invalid_window_uses_default);
    RUN_TEST(test_hampel_init_disabled);
    
    // Filter behavior tests
    RUN_TEST(test_hampel_disabled_returns_raw_value);
    RUN_TEST(test_hampel_returns_raw_until_buffer_fills);
    RUN_TEST(test_hampel_passes_normal_values);
    RUN_TEST(test_hampel_replaces_outlier);
    RUN_TEST(test_hampel_circular_buffer_wraps);
    
    // Standalone function tests
    RUN_TEST(test_hampel_filter_with_null_window);
    RUN_TEST(test_hampel_filter_with_small_window);
    RUN_TEST(test_hampel_filter_detects_outlier);
    RUN_TEST(test_hampel_filter_passes_normal_value);
    
    // Edge cases
    RUN_TEST(test_hampel_with_all_same_values);
    RUN_TEST(test_hampel_with_varying_values);
    
    return UNITY_END();
}

#ifdef ARDUINO
void setup() { delay(2000); process(); }
void loop() {}
#else
int main(int argc, char **argv) { return process(); }
#endif

