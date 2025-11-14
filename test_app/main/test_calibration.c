/*
 * ESPectre - Calibration End-to-End Tests
 * 
 * Tests complete calibration flow with synthetic data
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "calibration.h"
#include "csi_processor.h"
#include "detection_engine.h"
#include "filters.h"
#include "mock_csi_data.h"
#include "config_manager.h"
#include <math.h>
#include <string.h>

// Helper: extract all features for testing
static const uint8_t test_all_features[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
#define EXTRACT_ALL_FEATURES(data, len, features) \
    csi_extract_features(data, len, features, test_all_features, 10)

// Test: End-to-end calibration with synthetic baseline and movement data
TEST_CASE_ESP(calibration_end_to_end_with_mock_data, "[calibration]")
{
    // Initialize calibration system
    calibration_init();
    
    // Create mock config
    runtime_config_t config;
    config_init_defaults(&config);
    config.traffic_generator_rate = 15;  // Set traffic rate for sample-based calibration
    
    // Create mock normalizer
    adaptive_normalizer_t normalizer;
    adaptive_normalizer_init(&normalizer, 0.02f);
    
    // Start calibration (100 samples minimum required)
    bool started = calibration_start(100, &config, &normalizer, false);
    TEST_ASSERT_TRUE(started);
    
    // Initialize filters
    filter_buffer_t filter_buffer = {0};
    filter_buffer_init(&filter_buffer);
    
    // Simulate BASELINE phase
    wifi_csi_info_t csi_info;
    int8_t csi_buffer[384];
    csi_info.buf = csi_buffer;
    
    // Force phase to BASELINE (using test helper function)
    calibration_force_phase(CALIB_BASELINE);
    printf("Forced phase to BASELINE\n");
    
    // Simulate BASELINE phase (100 static packets)
    for (int i = 0; i < 100; i++) {
        generate_mock_csi_data(&csi_info, MOCK_CSI_STATIC);
        
        // Extract features (simplified - no history buffer)
        csi_features_t features;
        EXTRACT_ALL_FEATURES(csi_info.buf, csi_info.len, &features);
        
        // Feed to calibration
        feature_array_t feat_array;
        feat_array.features[0] = features.variance;
        feat_array.features[1] = features.skewness;
        feat_array.features[2] = features.kurtosis;
        feat_array.features[3] = features.entropy;
        feat_array.features[4] = features.iqr;
        feat_array.features[5] = features.spatial_variance;
        feat_array.features[6] = features.spatial_correlation;
        feat_array.features[7] = features.spatial_gradient;
        feat_array.features[8] = features.temporal_delta_mean;
        feat_array.features[9] = features.temporal_delta_variance;
        
        calibration_update(&feat_array);
    }
    
    // Force phase to MOVEMENT
    calibration_force_phase(CALIB_MOVEMENT);
    printf("Forced phase to MOVEMENT\n");
    
    // Simulate MOVEMENT phase (100 walking packets)
    csi_pattern_t walking = {
        .baseline_noise = 0.5f,
        .movement_amplitude = 8.0f,  // Strong movement
        .frequency = 2.0f
    };
    
    for (int i = 0; i < 100; i++) {
        // Generate movement data
        int8_t *movement_data = generate_csi_movement(128, &walking);
        TEST_ASSERT_NOT_NULL(movement_data);
        
        memcpy(csi_info.buf, movement_data, 128);
        csi_info.len = 128;
        free(movement_data);
        
        // Extract features
        csi_features_t features;
        EXTRACT_ALL_FEATURES(csi_info.buf, csi_info.len, &features);
        
        // Feed to calibration
        feature_array_t feat_array;
        feat_array.features[0] = features.variance;
        feat_array.features[1] = features.skewness;
        feat_array.features[2] = features.kurtosis;
        feat_array.features[3] = features.entropy;
        feat_array.features[4] = features.iqr;
        feat_array.features[5] = features.spatial_variance;
        feat_array.features[6] = features.spatial_correlation;
        feat_array.features[7] = features.spatial_gradient;
        feat_array.features[8] = features.temporal_delta_mean;
        feat_array.features[9] = features.temporal_delta_variance;
        
        calibration_update(&feat_array);
    }
    
    // Manually trigger analysis using test helper
    printf("Triggering calibration analysis...\n");
    calibration_trigger_analysis();
    
    // Get calibration results
    calibration_state_t calib_state;
    calibration_get_results(&calib_state);
    
    // Debug: Print calibration results
    printf("\n=== CALIBRATION RESULTS ===\n");
    printf("Baseline mean score: %.4f\n", calib_state.baseline_mean_score);
    printf("Movement mean score: %.4f\n", calib_state.movement_mean_score);
    printf("Separation ratio: %.4f\n", calib_state.separation_ratio);
    printf("Num features selected: %d\n", calib_state.num_selected);
    printf("Optimal threshold: %.4f\n", calib_state.optimal_threshold);
    
    // Print baseline stats for all 10 features
    printf("\n=== BASELINE STATS (all 10 features) ===\n");
    for (int i = 0; i < 10; i++) {
        printf("Feature %d: mean=%.4f, variance=%.4f, count=%zu\n", 
               i, 
               calib_state.baseline_stats[i].mean,
               calib_state.baseline_stats[i].m2 / (calib_state.baseline_stats[i].count > 0 ? calib_state.baseline_stats[i].count : 1),
               calib_state.baseline_stats[i].count);
    }
    
    printf("\n=== MOVEMENT STATS (all 10 features) ===\n");
    for (int i = 0; i < 10; i++) {
        printf("Feature %d: mean=%.4f, variance=%.4f, count=%zu\n", 
               i,
               calib_state.movement_stats[i].mean,
               calib_state.movement_stats[i].m2 / (calib_state.movement_stats[i].count > 0 ? calib_state.movement_stats[i].count : 1),
               calib_state.movement_stats[i].count);
    }
    printf("===========================\n\n");
    
    // Verify results
    TEST_ASSERT_GREATER_THAN(0, calib_state.num_selected);
    TEST_ASSERT_TRUE(calib_state.optimal_threshold > 0.0f);
    
    // CRITICAL TEST: Separation ratio should be >= 2.0 with strong movement
    TEST_ASSERT_TRUE(calib_state.separation_ratio >= 2.0f);
    
    // Movement should be > Baseline (movement has higher score)
    TEST_ASSERT_TRUE(calib_state.movement_mean_score > calib_state.baseline_mean_score);
    
    // Cleanup
    calibration_stop(&config);
}

// Test: Verify features are different between baseline and movement
TEST_CASE_ESP(features_differ_between_baseline_and_movement, "[calibration]")
{
    // Extract baseline features
    wifi_csi_info_t csi_info;
    int8_t csi_buffer[384];
    csi_info.buf = csi_buffer;
    generate_mock_csi_data(&csi_info, MOCK_CSI_STATIC);
    
    csi_features_t baseline_features;
    EXTRACT_ALL_FEATURES(csi_info.buf, csi_info.len, &baseline_features);
    
    // Extract movement features using MOCK_CSI_WALKING instead of generate_csi_movement
    generate_mock_csi_data(&csi_info, MOCK_CSI_WALKING);
    
    csi_features_t movement_features;
    EXTRACT_ALL_FEATURES(csi_info.buf, csi_info.len, &movement_features);
    
    // Debug: Print feature values
    printf("\n=== FEATURE COMPARISON ===\n");
    printf("Baseline variance: %.4f, Movement variance: %.4f\n", 
           baseline_features.variance, movement_features.variance);
    printf("Baseline spatial_gradient: %.4f, Movement spatial_gradient: %.4f\n",
           baseline_features.spatial_gradient, movement_features.spatial_gradient);
    printf("==========================\n\n");
    
    // Verify that movement features are significantly different from baseline
    // Movement should have higher variance due to sinusoidal pattern
    TEST_ASSERT_GREATER_THAN(baseline_features.variance, movement_features.variance);
    
    // Verify features are actually different (not just checking direction)
    float variance_ratio = movement_features.variance / (baseline_features.variance + 0.001f);
    TEST_ASSERT_TRUE(variance_ratio > 2.0f);  // Movement variance should be at least 2x baseline
}
