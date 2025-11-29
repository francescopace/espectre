/*
 * ESPectre - Calibration Algorithm Unit Tests
 *
 * Unit tests for NBVI (Normalized Baseline Variability Index) calculations
 * and utility functions used in auto-calibration.
 *
 * NBVI Formula: nbvi_weighted = α × (σ/μ²) + (1-α) × (σ/μ)
 * Where:
 *   - α = 0.3 (default weighting factor)
 *   - σ = standard deviation
 *   - μ = mean
 *   - σ/μ = Coefficient of Variation (CV)
 *   - σ/μ² = Energy-normalized variability
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cmath>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include "utils.h"
#include "esp_log.h"

using namespace esphome::espectre;

static const char *TAG = "test_calibration";

void setUp(void) {}
void tearDown(void) {}

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
// NBVI CALCULATION TESTS (Mathematical Validation)
// ============================================================================

// Helper: Calculate NBVI weighted (same formula as calibration_manager)
static float calculate_nbvi_weighted(const float* values, size_t count, float alpha = 0.3f) {
    if (count == 0 || !values) {
        return INFINITY;
    }
    
    // Calculate mean
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    float mean = sum / count;
    
    if (mean < 1e-6f) {
        return INFINITY;
    }
    
    // Calculate standard deviation
    float variance = calculate_variance_two_pass(values, count);
    float std = std::sqrt(variance);
    
    // NBVI Weighted
    float cv = std / mean;                      // Coefficient of variation
    float nbvi_energy = std / (mean * mean);    // Energy normalization
    float nbvi_weighted = alpha * nbvi_energy + (1.0f - alpha) * cv;
    
    return nbvi_weighted;
}

void test_nbvi_stable_signal_low_value(void) {
    // Stable signal: low variance relative to mean → low NBVI
    float stable[] = {100.0f, 100.0f, 100.0f, 100.0f, 100.0f};
    float nbvi = calculate_nbvi_weighted(stable, 5);
    
    // With zero variance, NBVI should be 0
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 0.0f, nbvi);
}

void test_nbvi_noisy_signal_high_value(void) {
    // Noisy signal: high variance relative to mean → high NBVI
    float noisy[] = {50.0f, 150.0f, 50.0f, 150.0f, 50.0f};
    float nbvi = calculate_nbvi_weighted(noisy, 5);
    
    // NBVI should be significantly higher than 0 for noisy signal
    TEST_ASSERT_TRUE(nbvi > 0.1f);
    
    ESP_LOGI(TAG, "Noisy signal NBVI: %.6f", nbvi);
}

void test_nbvi_comparison_stable_vs_noisy(void) {
    // Stable subcarrier (good for motion detection)
    float stable[] = {100.0f, 101.0f, 99.0f, 100.0f, 100.0f};
    float nbvi_stable = calculate_nbvi_weighted(stable, 5);
    
    // Noisy subcarrier (bad for motion detection)
    float noisy[] = {80.0f, 120.0f, 90.0f, 110.0f, 100.0f};
    float nbvi_noisy = calculate_nbvi_weighted(noisy, 5);
    
    ESP_LOGI(TAG, "NBVI stable: %.6f, noisy: %.6f", nbvi_stable, nbvi_noisy);
    
    // Stable should have lower NBVI than noisy
    TEST_ASSERT_TRUE(nbvi_stable < nbvi_noisy);
}

void test_nbvi_zero_mean_returns_infinity(void) {
    float zero_mean[] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float nbvi = calculate_nbvi_weighted(zero_mean, 5);
    
    TEST_ASSERT_TRUE(std::isinf(nbvi));
}

void test_nbvi_alpha_weighting_effect(void) {
    float values[] = {90.0f, 95.0f, 100.0f, 105.0f, 110.0f};
    
    // Different alpha values
    float nbvi_alpha_0 = calculate_nbvi_weighted(values, 5, 0.0f);   // Pure CV
    float nbvi_alpha_1 = calculate_nbvi_weighted(values, 5, 1.0f);   // Pure energy
    float nbvi_alpha_03 = calculate_nbvi_weighted(values, 5, 0.3f);  // Default
    
    // Alpha=0.3 should be between pure CV and pure energy
    // (or at least different from both extremes)
    ESP_LOGI(TAG, "NBVI α=0: %.6f, α=0.3: %.6f, α=1: %.6f", 
             nbvi_alpha_0, nbvi_alpha_03, nbvi_alpha_1);
    
    TEST_ASSERT_NOT_EQUAL(nbvi_alpha_0, nbvi_alpha_1);
}

// ============================================================================
// PERCENTILE CALCULATION TESTS
// ============================================================================

// Helper: Calculate percentile (same algorithm as calibration_manager)
static float calculate_percentile(std::vector<float> values, uint8_t percentile) {
    if (values.empty()) return 0.0f;
    
    std::sort(values.begin(), values.end());
    
    size_t n = values.size();
    float k = (percentile / 100.0f) * (n - 1);
    size_t f = (size_t)k;
    size_t c = f + 1;
    
    if (c >= n) {
        return values[n - 1];
    }
    
    float d0 = values[f] * (c - k);
    float d1 = values[c] * (k - f);
    return d0 + d1;
}

void test_percentile_p50_is_median(void) {
    std::vector<float> values = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    float p50 = calculate_percentile(values, 50);
    
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 3.0f, p50);
}

void test_percentile_p0_is_minimum(void) {
    std::vector<float> values = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f};
    float p0 = calculate_percentile(values, 0);
    
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, p0);
}

void test_percentile_p100_is_maximum(void) {
    std::vector<float> values = {5.0f, 2.0f, 8.0f, 1.0f, 9.0f};
    float p100 = calculate_percentile(values, 100);
    
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 9.0f, p100);
}

void test_percentile_p10_baseline_detection(void) {
    // Simulate variance values from sliding windows
    // Lower percentile = more stable baseline
    std::vector<float> variances = {0.5f, 0.8f, 1.2f, 0.3f, 2.0f, 0.4f, 1.5f, 0.6f, 0.7f, 1.0f};
    float p10 = calculate_percentile(variances, 10);
    
    ESP_LOGI(TAG, "p10 of variances: %.4f (min=0.3, max=2.0)", p10);
    
    // p10 should be close to minimum values (between min and median)
    TEST_ASSERT_TRUE(p10 < 0.6f);  // Should be in lower range
    TEST_ASSERT_TRUE(p10 >= 0.3f); // At least the minimum
}

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

int process(void) {
    UNITY_BEGIN();
    
    // Variance calculation tests
    RUN_TEST(test_variance_empty_array);
    RUN_TEST(test_variance_single_element);
    RUN_TEST(test_variance_identical_values);
    RUN_TEST(test_variance_known_values);
    RUN_TEST(test_variance_with_negative_values);
    RUN_TEST(test_variance_large_values_numerical_stability);
    
    // NBVI calculation tests
    RUN_TEST(test_nbvi_stable_signal_low_value);
    RUN_TEST(test_nbvi_noisy_signal_high_value);
    RUN_TEST(test_nbvi_comparison_stable_vs_noisy);
    RUN_TEST(test_nbvi_zero_mean_returns_infinity);
    RUN_TEST(test_nbvi_alpha_weighting_effect);
    
    // Percentile calculation tests
    RUN_TEST(test_percentile_p50_is_median);
    RUN_TEST(test_percentile_p0_is_minimum);
    RUN_TEST(test_percentile_p100_is_maximum);
    RUN_TEST(test_percentile_p10_baseline_detection);
    
    // Spectral spacing tests
    RUN_TEST(test_spectral_spacing_valid);
    RUN_TEST(test_spectral_spacing_invalid);
    RUN_TEST(test_spectral_spacing_edge_case);
    
    // Subcarrier ranking tests
    RUN_TEST(test_subcarrier_ranking_by_nbvi);
    
    // Compare functions tests
    RUN_TEST(test_compare_float_ascending);
    RUN_TEST(test_compare_float_abs);
    
    return UNITY_END();
}

#ifdef ARDUINO
void setup() { delay(2000); process(); }
void loop() {}
#else
int main(int argc, char **argv) { return process(); }
#endif

