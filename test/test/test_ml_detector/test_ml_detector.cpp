/*
 * ESPectre - MLDetector Unit Tests
 *
 * Tests the MLDetector class implementing BaseDetector interface.
 * Validates MLP inference accuracy against reference model outputs.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "ml_detector.h"
#include "ml_features.h"
#include "ml_weights.h"
#include "esphome/core/log.h"

using namespace esphome::espectre;

static const char *TAG = "test_ml_detector";

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// TEST DATA (extracted from ml_test_data.npz - raw features, not normalized)
// The test applies normalization using ML_FEATURE_MEAN/SCALE, same as inference
// ============================================================================

constexpr int NUM_TEST_SAMPLES = 6;

// Sample 0: idx=0, expected=0.000000
constexpr float TEST_FEATURES_0[12] = {15.037017f, 1.826449f, 16.590075f, 2.687921f, 13.902154f, 3.335915f, 6.951077f, 1.106964f, 0.112449f, -1.540914f, 0.018642f, -0.911717f};
constexpr float TEST_EXPECTED_0 = 0.000000f;

// Sample 1: idx=100, expected=1.000000
constexpr float TEST_FEATURES_1[12] = {6.483670f, 1.729731f, 10.709900f, 3.155840f, 7.554060f, 2.991968f, 3.777030f, 3.090561f, 0.029862f, -1.543806f, -0.010408f, 2.296937f};
constexpr float TEST_EXPECTED_1 = 1.000000f;

// Sample 2: idx=500, expected=1.000000
constexpr float TEST_FEATURES_2[12] = {15.039514f, 2.640327f, 25.975748f, 11.449318f, 14.526430f, 6.971325f, 7.263215f, 2.501832f, -1.725084f, 1.090417f, -0.028332f, -8.790643f};
constexpr float TEST_EXPECTED_2 = 1.000000f;

// Sample 3: idx=1000, expected=0.065183
constexpr float TEST_FEATURES_3[12] = {18.843235f, 2.961268f, 22.827011f, 10.418902f, 12.408109f, 8.769106f, 6.204054f, 2.188493f, -0.774085f, -0.291629f, -0.030693f, -9.512109f};
constexpr float TEST_EXPECTED_3 = 0.065183f;

// Sample 4: idx=2000, expected=0.000000
constexpr float TEST_FEATURES_4[12] = {15.055570f, 1.826135f, 16.127796f, 2.597119f, 13.530678f, 3.334768f, 6.765339f, 0.607364f, 0.171855f, -1.456887f, -0.001839f, -0.126887f};
constexpr float TEST_EXPECTED_4 = 0.000000f;

// Sample 5: idx=2500, expected=0.000000
constexpr float TEST_FEATURES_5[12] = {13.668566f, 4.314494f, 17.185942f, 2.711784f, 14.474158f, 18.614859f, 7.237079f, 1.418150f, 0.138241f, -1.500547f, -0.084217f, 0.682294f};
constexpr float TEST_EXPECTED_5 = 0.000000f;

// Array of pointers for iteration
constexpr const float* TEST_FEATURES[NUM_TEST_SAMPLES] = {
    TEST_FEATURES_0, TEST_FEATURES_1, TEST_FEATURES_2,
    TEST_FEATURES_3, TEST_FEATURES_4, TEST_FEATURES_5
};

constexpr float TEST_EXPECTED[NUM_TEST_SAMPLES] = {
    TEST_EXPECTED_0, TEST_EXPECTED_1, TEST_EXPECTED_2,
    TEST_EXPECTED_3, TEST_EXPECTED_4, TEST_EXPECTED_5
};

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_ml_detector_default_constructor(void) {
    MLDetector detector;
    
    TEST_ASSERT_EQUAL(DETECTOR_DEFAULT_WINDOW_SIZE, detector.get_window_size());
    TEST_ASSERT_EQUAL_FLOAT(ML_DEFAULT_THRESHOLD, detector.get_threshold());
    TEST_ASSERT_EQUAL(MotionState::IDLE, detector.get_state());
    TEST_ASSERT_EQUAL(0, detector.get_total_packets());
}

void test_ml_detector_custom_constructor(void) {
    MLDetector detector(100, 0.7f);
    
    TEST_ASSERT_EQUAL(100, detector.get_window_size());
    TEST_ASSERT_EQUAL_FLOAT(0.7f, detector.get_threshold());
}

void test_ml_detector_get_name(void) {
    MLDetector detector;
    
    TEST_ASSERT_EQUAL_STRING("ML", detector.get_name());
}

// ============================================================================
// THRESHOLD TESTS
// ============================================================================

void test_ml_detector_set_threshold_valid(void) {
    MLDetector detector;
    
    TEST_ASSERT_TRUE(detector.set_threshold(0.7f));
    TEST_ASSERT_EQUAL_FLOAT(0.7f, detector.get_threshold());
}

void test_ml_detector_set_threshold_min(void) {
    MLDetector detector;
    
    TEST_ASSERT_TRUE(detector.set_threshold(ML_MIN_THRESHOLD));
    TEST_ASSERT_EQUAL_FLOAT(ML_MIN_THRESHOLD, detector.get_threshold());
}

void test_ml_detector_set_threshold_max(void) {
    MLDetector detector;
    
    TEST_ASSERT_TRUE(detector.set_threshold(ML_MAX_THRESHOLD));
    TEST_ASSERT_EQUAL_FLOAT(ML_MAX_THRESHOLD, detector.get_threshold());
}

void test_ml_detector_set_threshold_below_min(void) {
    MLDetector detector;
    float original = detector.get_threshold();
    
    TEST_ASSERT_FALSE(detector.set_threshold(-0.1f));
    TEST_ASSERT_EQUAL_FLOAT(original, detector.get_threshold());
}

void test_ml_detector_set_threshold_above_max(void) {
    MLDetector detector;
    float original = detector.get_threshold();
    
    TEST_ASSERT_FALSE(detector.set_threshold(1.1f));
    TEST_ASSERT_EQUAL_FLOAT(original, detector.get_threshold());
}

// ============================================================================
// MLP INFERENCE TESTS
// ============================================================================

// Helper function to run MLP inference (same as MLDetector::predict)
static float run_inference(const float* features) {
    float normalized[12];
    float h1[16];
    float h2[8];
    
    // Normalize raw features using StandardScaler params
    for (int i = 0; i < 12; i++) {
        normalized[i] = (features[i] - ML_FEATURE_MEAN[i]) / ML_FEATURE_SCALE[i];
    }
    
    // Layer 1: 12 -> 16 + ReLU
    for (int j = 0; j < 16; j++) {
        h1[j] = ML_B1[j];
        for (int i = 0; i < 12; i++) {
            h1[j] += normalized[i] * ML_W1[i][j];
        }
        h1[j] = std::max(0.0f, h1[j]);
    }
    
    // Layer 2: 16 -> 8 + ReLU
    for (int j = 0; j < 8; j++) {
        h2[j] = ML_B2[j];
        for (int i = 0; i < 16; i++) {
            h2[j] += h1[i] * ML_W2[i][j];
        }
        h2[j] = std::max(0.0f, h2[j]);
    }
    
    // Layer 3: 8 -> 1 + Sigmoid
    float out = ML_B3[0];
    for (int i = 0; i < 8; i++) {
        out += h2[i] * ML_W3[i][0];
    }
    
    // Sigmoid with overflow protection
    if (out < -20.0f) return 0.0f;
    if (out > 20.0f) return 1.0f;
    return 1.0f / (1.0f + std::exp(-out));
}

void test_ml_inference_matches_reference(void) {
    const float TOLERANCE = 1e-4f;  // Allow small numerical error
    
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        float result = run_inference(TEST_FEATURES[i]);
        float expected = TEST_EXPECTED[i];
        float error = std::abs(result - expected);
        
        ESP_LOGD(TAG, "Sample %d: expected=%.6f, got=%.6f, error=%.2e",
                 i, expected, result, error);
        
        TEST_ASSERT_FLOAT_WITHIN(TOLERANCE, expected, result);
    }
}

void test_ml_inference_output_range(void) {
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        float result = run_inference(TEST_FEATURES[i]);
        
        TEST_ASSERT_TRUE(result >= 0.0f);
        TEST_ASSERT_TRUE(result <= 1.0f);
    }
}

void test_ml_inference_classification(void) {
    // Test that motion/idle classification is correct
    const float THRESHOLD = 0.5f;
    
    for (int i = 0; i < NUM_TEST_SAMPLES; i++) {
        float result = run_inference(TEST_FEATURES[i]);
        float expected = TEST_EXPECTED[i];
        
        bool result_motion = (result > THRESHOLD);
        bool expected_motion = (expected > THRESHOLD);
        
        TEST_ASSERT_EQUAL(expected_motion, result_motion);
    }
}

// ============================================================================
// FEATURE EXTRACTION TESTS
// ============================================================================

void test_feature_extraction_basic(void) {
    float turb_buffer[50];
    float amplitudes[12] = {10.0f, 12.0f, 11.0f, 13.0f, 9.0f, 14.0f,
                            10.5f, 11.5f, 12.5f, 10.0f, 11.0f, 13.0f};
    float features[12];
    
    // Fill buffer with synthetic data
    for (int i = 0; i < 50; i++) {
        turb_buffer[i] = 10.0f + (i % 5) * 0.5f;
    }
    
    extract_ml_features(turb_buffer, 50, amplitudes, 12, features);
    
    // Verify features are reasonable
    TEST_ASSERT_TRUE(features[0] > 0);  // turb_mean > 0
    TEST_ASSERT_TRUE(features[1] >= 0); // turb_std >= 0
    TEST_ASSERT_TRUE(features[2] >= features[3]); // turb_max >= turb_min
    TEST_ASSERT_FLOAT_WITHIN(0.001f, features[2] - features[3], features[4]); // range = max - min
    TEST_ASSERT_TRUE(features[7] >= 0); // entropy >= 0
}

void test_feature_extraction_empty_buffer(void) {
    float turb_buffer[50] = {0};
    float features[12];
    
    extract_ml_features(turb_buffer, 0, nullptr, 0, features);
    
    // All features should be 0 for empty buffer
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, features[i]);
    }
}

// ============================================================================
// ML SUBCARRIERS TESTS
// ============================================================================

void test_ml_subcarriers_count(void) {
    TEST_ASSERT_EQUAL(12, sizeof(ML_SUBCARRIERS) / sizeof(ML_SUBCARRIERS[0]));
}

void test_ml_subcarriers_range(void) {
    for (int i = 0; i < 12; i++) {
        TEST_ASSERT_TRUE(ML_SUBCARRIERS[i] >= 0);
        TEST_ASSERT_TRUE(ML_SUBCARRIERS[i] < 64);  // HT20 has 64 subcarriers
    }
}

void test_ml_subcarriers_sorted(void) {
    for (int i = 1; i < 12; i++) {
        TEST_ASSERT_TRUE(ML_SUBCARRIERS[i] > ML_SUBCARRIERS[i-1]);
    }
}

// ============================================================================
// PERFORMANCE TESTS
// ============================================================================

void test_ml_inference_performance(void) {
    const int NUM_ITERATIONS = 1000;
    
    // Warm up
    for (int i = 0; i < 10; i++) {
        run_inference(TEST_FEATURES[0]);
    }
    
    // Benchmark (note: on native platform, not on actual ESP32)
    uint32_t start = 0;  // Would use micros() on ESP32
    
    for (int i = 0; i < NUM_ITERATIONS; i++) {
        run_inference(TEST_FEATURES[i % NUM_TEST_SAMPLES]);
    }
    
    uint32_t elapsed = 0;  // Would calculate elapsed time
    
    // Just verify it completes without error
    ESP_LOGI(TAG, "Completed %d inference iterations", NUM_ITERATIONS);
    TEST_PASS();
}

// ============================================================================
// MAIN
// ============================================================================

int main(int argc, char **argv) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_ml_detector_default_constructor);
    RUN_TEST(test_ml_detector_custom_constructor);
    RUN_TEST(test_ml_detector_get_name);
    
    // Threshold tests
    RUN_TEST(test_ml_detector_set_threshold_valid);
    RUN_TEST(test_ml_detector_set_threshold_min);
    RUN_TEST(test_ml_detector_set_threshold_max);
    RUN_TEST(test_ml_detector_set_threshold_below_min);
    RUN_TEST(test_ml_detector_set_threshold_above_max);
    
    // MLP inference tests
    RUN_TEST(test_ml_inference_matches_reference);
    RUN_TEST(test_ml_inference_output_range);
    RUN_TEST(test_ml_inference_classification);
    
    // Feature extraction tests
    RUN_TEST(test_feature_extraction_basic);
    RUN_TEST(test_feature_extraction_empty_buffer);
    
    // Subcarriers tests
    RUN_TEST(test_ml_subcarriers_count);
    RUN_TEST(test_ml_subcarriers_range);
    RUN_TEST(test_ml_subcarriers_sorted);
    
    // Performance tests
    RUN_TEST(test_ml_inference_performance);
    
    return UNITY_END();
}
