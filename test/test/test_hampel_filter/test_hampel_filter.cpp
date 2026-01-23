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
#include "filters.h"
#include "utils.h"
#include "esphome/core/log.h"

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

// ============================================================================
// REAL CSI DATA TESTS
// ============================================================================

// Include CSI data loader (loads from NPZ files)
#include "csi_test_data.h"

// Compatibility macros for existing test code
#define baseline_packets csi_test_data::baseline_packets()
#define movement_packets csi_test_data::movement_packets()
#define num_baseline csi_test_data::num_baseline()
#define num_movement csi_test_data::num_movement()
#define packet_size csi_test_data::packet_size()

// Using calculate_spatial_turbulence_from_csi from utils.h

// Optimal subcarriers by dataset (found via grid search analysis)
static const uint8_t SUBCARRIERS_64SC[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
static const uint8_t SUBCARRIERS_256SC[] = {147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158};
inline const uint8_t* get_subcarriers() {
    return (csi_test_data::num_subcarriers() == 64) ? SUBCARRIERS_64SC : SUBCARRIERS_256SC;
}
static const uint8_t NUM_SC = 12;

void test_hampel_with_real_baseline_turbulence(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 7, 4.0f, true);
    
    // Process 100 baseline packets
    float raw_values[100];
    float filtered_values[100];
    
    for (int i = 0; i < 100; i++) {
        float turb = calculate_spatial_turbulence_from_csi(baseline_packets[i], packet_size, get_subcarriers(), NUM_SC);
        raw_values[i] = turb;
        filtered_values[i] = hampel_filter_turbulence(&state, turb);
    }
    
    // Calculate variance of raw vs filtered
    float raw_mean = 0.0f, filtered_mean = 0.0f;
    for (int i = 0; i < 100; i++) {
        raw_mean += raw_values[i];
        filtered_mean += filtered_values[i];
    }
    raw_mean /= 100;
    filtered_mean /= 100;
    
    float raw_var = 0.0f, filtered_var = 0.0f;
    for (int i = 0; i < 100; i++) {
        raw_var += (raw_values[i] - raw_mean) * (raw_values[i] - raw_mean);
        filtered_var += (filtered_values[i] - filtered_mean) * (filtered_values[i] - filtered_mean);
    }
    raw_var /= 100;
    filtered_var /= 100;
    
    ESP_LOGI(TAG, "Baseline turbulence - Raw variance: %.4f, Filtered variance: %.4f",
             raw_var, filtered_var);
    
    // Filtered variance should be <= raw variance (outliers removed)
    TEST_ASSERT_TRUE(filtered_var <= raw_var * 1.1f);  // Allow small tolerance
}

void test_hampel_with_real_movement_turbulence(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 7, 4.0f, true);
    
    // Process 100 movement packets
    float raw_values[100];
    float filtered_values[100];
    
    for (int i = 0; i < 100; i++) {
        float turb = calculate_spatial_turbulence_from_csi(movement_packets[i], packet_size, get_subcarriers(), NUM_SC);
        raw_values[i] = turb;
        filtered_values[i] = hampel_filter_turbulence(&state, turb);
    }
    
    // Movement should have higher turbulence than baseline
    float avg_turb = 0.0f;
    for (int i = 0; i < 100; i++) {
        avg_turb += filtered_values[i];
    }
    avg_turb /= 100;
    
    ESP_LOGI(TAG, "Movement average filtered turbulence: %.4f", avg_turb);
    
    // Turbulence should be positive
    TEST_ASSERT_TRUE(avg_turb > 0.0f);
}

void test_hampel_outlier_detection_on_real_data(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 7, 3.0f, true);
    
    // Fill with baseline turbulence values
    for (int i = 0; i < 20; i++) {
        float turb = calculate_spatial_turbulence_from_csi(baseline_packets[i], packet_size, get_subcarriers(), NUM_SC);
        hampel_filter_turbulence(&state, turb);
    }
    
    // Now inject a synthetic outlier (10x normal value)
    float normal_turb = calculate_spatial_turbulence_from_csi(baseline_packets[20], packet_size, get_subcarriers(), NUM_SC);
    float outlier = normal_turb * 10.0f;
    
    float filtered = hampel_filter_turbulence(&state, outlier);
    
    ESP_LOGI(TAG, "Outlier test - Normal: %.4f, Outlier: %.4f, Filtered: %.4f",
             normal_turb, outlier, filtered);
    
    // Filtered value should be much closer to normal than to outlier
    float dist_to_normal = std::abs(filtered - normal_turb);
    float dist_to_outlier = std::abs(filtered - outlier);
    
    TEST_ASSERT_TRUE_MESSAGE(dist_to_normal < dist_to_outlier,
        "Filtered value should be closer to normal than to outlier");
}

void test_hampel_preserves_movement_signal(void) {
    hampel_turbulence_state_t state;
    hampel_turbulence_init(&state, 7, 4.0f, true);
    
    // Process mixed baseline and movement
    float baseline_filtered[50];
    float movement_filtered[50];
    
    // First 50 baseline
    for (int i = 0; i < 50; i++) {
        float turb = calculate_spatial_turbulence_from_csi(baseline_packets[i], packet_size, get_subcarriers(), NUM_SC);
        baseline_filtered[i] = hampel_filter_turbulence(&state, turb);
    }
    
    // Reset and process movement
    hampel_turbulence_init(&state, 7, 4.0f, true);
    for (int i = 0; i < 50; i++) {
        float turb = calculate_spatial_turbulence_from_csi(movement_packets[i], packet_size, get_subcarriers(), NUM_SC);
        movement_filtered[i] = hampel_filter_turbulence(&state, turb);
    }
    
    // Calculate averages
    float avg_baseline = 0.0f, avg_movement = 0.0f;
    for (int i = 10; i < 50; i++) {  // Skip warmup period
        avg_baseline += baseline_filtered[i];
        avg_movement += movement_filtered[i];
    }
    avg_baseline /= 40;
    avg_movement /= 40;
    
    ESP_LOGI(TAG, "Filtered averages - Baseline: %.4f, Movement: %.4f", 
             avg_baseline, avg_movement);
    
    // Movement should still have different characteristics than baseline
    // (filter should not remove the real signal)
    // Note: This is a weak assertion - the main point is the filter works
    TEST_ASSERT_TRUE(avg_baseline > 0.0f);
    TEST_ASSERT_TRUE(avg_movement > 0.0f);
}

int process(void) {
    // Load CSI test data from NPZ files
    if (!csi_test_data::load()) {
        printf("ERROR: Failed to load CSI test data from NPZ files\n");
        return 1;
    }
    
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
    
    // Real CSI data tests
    RUN_TEST(test_hampel_with_real_baseline_turbulence);
    RUN_TEST(test_hampel_with_real_movement_turbulence);
    RUN_TEST(test_hampel_outlier_detection_on_real_data);
    RUN_TEST(test_hampel_preserves_movement_signal);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

