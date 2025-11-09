/*
 * ESPectre - Real CSI Data Calibration Test
 * 
 * Tests calibration with real CSI data captured from actual environment
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "calibration.h"
#include "csi_processor.h"
#include "filters.h"
#include "real_csi_data.h"
#include "config_manager.h"
#include <math.h>
#include <string.h>

// Helper: extract all features for testing
static const uint8_t test_all_features[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// Test: Calibration with real CSI data
TEST_CASE_ESP("Calibration with real CSI data", "[calibration][real]")
{
    // Initialize calibration
    calibration_init();
    
    runtime_config_t config;
    config_init_defaults(&config);
    config.traffic_generator_rate = 15;  // Set traffic rate for sample-based calibration
    
    adaptive_normalizer_t normalizer;
    adaptive_normalizer_init(&normalizer, 0.02f);
    
    // Start calibration with 50 samples (matching our dataset size)
    calibration_start(50, &config, &normalizer);
    
    // Initialize filters
    filter_buffer_t filter_buffer = {0};
    filter_buffer_init(&filter_buffer);
    
    // Force to BASELINE phase
    calibration_force_phase(CALIB_BASELINE);
    printf("Processing BASELINE phase with real CSI data...\n");
    
    // Array of all baseline packets
    const int8_t *baseline_packets[] = {
        real_baseline_0, real_baseline_1, real_baseline_2, real_baseline_3,
        real_baseline_4, real_baseline_5, real_baseline_6, real_baseline_7,
        real_baseline_8, real_baseline_9, real_baseline_10, real_baseline_11,
        real_baseline_12, real_baseline_13, real_baseline_14, real_baseline_15,
        real_baseline_16, real_baseline_17, real_baseline_18, real_baseline_19,
        real_baseline_20, real_baseline_21, real_baseline_22, real_baseline_23,
        real_baseline_24, real_baseline_25, real_baseline_26, real_baseline_27,
        real_baseline_28, real_baseline_29, real_baseline_30, real_baseline_31,
        real_baseline_32, real_baseline_33, real_baseline_34, real_baseline_35,
        real_baseline_36, real_baseline_37, real_baseline_38, real_baseline_39,
        real_baseline_40, real_baseline_41, real_baseline_42, real_baseline_43,
        real_baseline_44, real_baseline_45, real_baseline_46, real_baseline_47,
        real_baseline_48, real_baseline_49
    };
    
    // Process all 50 baseline packets WITHOUT replication
    for (int p = 0; p < 50; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, 10);
        
        // Log first packet features for debugging
        if (p == 0) {
            printf("\n=== BASELINE PACKET 0 FEATURES ===\n");
            printf("variance: %.4f\n", features.variance);
            printf("skewness: %.4f\n", features.skewness);
            printf("kurtosis: %.4f\n", features.kurtosis);
            printf("entropy: %.4f\n", features.entropy);
            printf("iqr: %.4f\n", features.iqr);
            printf("spatial_variance: %.4f\n", features.spatial_variance);
            printf("spatial_correlation: %.4f\n", features.spatial_correlation);
            printf("spatial_gradient: %.4f\n", features.spatial_gradient);
            printf("temporal_delta_mean: %.4f\n", features.temporal_delta_mean);
            printf("temporal_delta_variance: %.4f\n", features.temporal_delta_variance);
            printf("===================================\n\n");
        }
        
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
    
    printf("Baseline phase: collected %d samples\n", 50);
    
    // Check completion to advance to next phase
    calibration_check_completion();
    
    // Force to MOVEMENT phase (in case check_completion didn't advance due to min samples)
    calibration_force_phase(CALIB_MOVEMENT);
    printf("Processing MOVEMENT phase with real CSI data...\n");
    
    // Array of all 50 movement packets
    const int8_t *movement_packets[] = {
        real_movement_0, real_movement_1, real_movement_2, real_movement_3,
        real_movement_4, real_movement_5, real_movement_6, real_movement_7,
        real_movement_8, real_movement_9, real_movement_10, real_movement_11,
        real_movement_12, real_movement_13, real_movement_14, real_movement_15,
        real_movement_16, real_movement_17, real_movement_18, real_movement_19,
        real_movement_20, real_movement_21, real_movement_22, real_movement_23,
        real_movement_24, real_movement_25, real_movement_26, real_movement_27,
        real_movement_28, real_movement_29, real_movement_30, real_movement_31,
        real_movement_32, real_movement_33, real_movement_34, real_movement_35,
        real_movement_36, real_movement_37, real_movement_38, real_movement_39,
        real_movement_40, real_movement_41, real_movement_42, real_movement_43,
        real_movement_44, real_movement_45, real_movement_46, real_movement_47,
        real_movement_48, real_movement_49
    };
    
    // Process all 50 movement packets WITHOUT replication
    for (int p = 0; p < 50; p++) {
        csi_features_t features;
        csi_extract_features(movement_packets[p], 128, &features, test_all_features, 10);
        
        // Log first packet features for debugging
        if (p == 0) {
            printf("\n=== MOVEMENT PACKET 0 FEATURES ===\n");
            printf("variance: %.4f\n", features.variance);
            printf("skewness: %.4f\n", features.skewness);
            printf("kurtosis: %.4f\n", features.kurtosis);
            printf("entropy: %.4f\n", features.entropy);
            printf("iqr: %.4f\n", features.iqr);
            printf("spatial_variance: %.4f\n", features.spatial_variance);
            printf("spatial_correlation: %.4f\n", features.spatial_correlation);
            printf("spatial_gradient: %.4f\n", features.spatial_gradient);
            printf("temporal_delta_mean: %.4f\n", features.temporal_delta_mean);
            printf("temporal_delta_variance: %.4f\n", features.temporal_delta_variance);
            printf("===================================\n\n");
        }
        
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
    
    printf("Movement phase: collected %d samples\n", 50);
    
    // Trigger analysis
    printf("Analyzing real CSI data...\n");
    calibration_trigger_analysis();
    
    // Get results
    calibration_state_t calib_state;
    calibration_get_results(&calib_state);
    
    // Print detailed results
    printf("\n=== REAL CSI CALIBRATION RESULTS ===\n");
    printf("Baseline score: %.4f\n", calib_state.baseline_mean_score);
    printf("Movement score: %.4f\n", calib_state.movement_mean_score);
    printf("Separation ratio: %.4f\n", calib_state.separation_ratio);
    printf("Num features: %d\n", calib_state.num_selected);
    printf("Threshold: %.4f\n", calib_state.optimal_threshold);
    
    // Print selected features with their ranges
    printf("\n=== SELECTED FEATURES & RANGES ===\n");
    const char* feature_names[] = {"variance", "skewness", "kurtosis", "entropy", "iqr",
                                   "spatial_variance", "spatial_correlation", "spatial_gradient",
                                   "temporal_delta_mean", "temporal_delta_variance"};
    for (int i = 0; i < calib_state.num_selected; i++) {
        uint8_t feat_idx = calib_state.selected_features[i];
        printf("Feature %d: %s\n", i, feature_names[feat_idx]);
        printf("  Weight: %.4f\n", calib_state.optimized_weights[i]);
        printf("  Min: %.4f, Max: %.4f\n", calib_state.feature_min[i], calib_state.feature_max[i]);
        printf("  Baseline mean: %.4f\n", calib_state.baseline_stats[feat_idx].mean);
        printf("  Movement mean: %.4f\n", calib_state.movement_stats[feat_idx].mean);
    }
    printf("====================================\n\n");
    
    // Verify results
    TEST_ASSERT_GREATER_THAN(0, calib_state.num_selected);
    TEST_ASSERT_TRUE(calib_state.optimal_threshold > 0.0f);
    
    // Note: Real CSI data may have lower separation ratio due to environmental noise
    // We just verify it's positive (movement detected as different from baseline)
    TEST_ASSERT_TRUE(calib_state.separation_ratio > 0.0f);
    
    // NEW: Verify wavelet filter configuration
    bool butterworth, wavelet, hampel, savgol, adaptive_norm;
    int wavelet_level;
    float wavelet_threshold, hampel_threshold, norm_alpha;
    calibration_get_filter_config(&butterworth, &wavelet, &wavelet_level, &wavelet_threshold,
                                 &hampel, &hampel_threshold, &savgol, &adaptive_norm, &norm_alpha);
    
    printf("\n=== RECOMMENDED FILTER CONFIG ===\n");
    printf("Butterworth: %s\n", butterworth ? "ON" : "OFF");
    printf("Wavelet: %s", wavelet ? "ON" : "OFF");
    if (wavelet) {
        printf(" (level=%d, threshold=%.1f)\n", wavelet_level, wavelet_threshold);
    } else {
        printf("\n");
    }
    printf("Hampel: %s", hampel ? "ON" : "OFF");
    if (hampel) {
        printf(" (threshold=%.1f)\n", hampel_threshold);
    } else {
        printf("\n");
    }
    printf("Savitzky-Golay: %s\n", savgol ? "ON" : "OFF");
    printf("Adaptive Normalizer: %s", adaptive_norm ? "ON" : "OFF");
    if (adaptive_norm) {
        printf(" (alpha=%.3f)\n", norm_alpha);
    } else {
        printf("\n");
    }
    printf("=================================\n\n");
    
    // Butterworth should always be recommended
    TEST_ASSERT_TRUE(butterworth);
    
    // If baseline variance is high (>500), wavelet should be recommended
    float baseline_variance = calib_state.baseline_stats[0].mean;
    if (baseline_variance > 500.0f) {
        TEST_ASSERT_TRUE(wavelet);
        TEST_ASSERT_EQUAL(3, wavelet_level);
        TEST_ASSERT_TRUE(wavelet_threshold >= 0.5f && wavelet_threshold <= 2.0f);
        printf("✅ Wavelet correctly enabled for high variance (%.1f)\n", baseline_variance);
    } else {
        printf("ℹ️  Wavelet not needed - baseline variance is low (%.1f)\n", baseline_variance);
    }
    
    // Cleanup
    calibration_stop(&config);
}
