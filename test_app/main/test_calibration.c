/*
 * ESPectre - Calibration Tests (Real Data Only)
 * 
 * Tests calibration with real CSI data.
 * Mock data tests removed - use test_real_calibration.c instead.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "csi_processor.h"
#include "real_csi_data.h"
#include <math.h>
#include <string.h>

// Include CSI data arrays
#include "real_csi_arrays.inc"

// Helper: extract all features for testing
static const uint8_t test_all_features[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
#define EXTRACT_ALL_FEATURES(data, len, features) \
    csi_extract_features(data, len, features, test_all_features, 10)

// Test: Verify features are different between baseline and movement (using real data)
TEST_CASE_ESP(features_differ_between_baseline_and_movement, "[calibration]")
{
    // Extract baseline features from real data
    csi_features_t baseline_features;
    EXTRACT_ALL_FEATURES(baseline_packets[0], 128, &baseline_features);
    
    // Extract movement features from real data
    csi_features_t movement_features;
    EXTRACT_ALL_FEATURES(movement_packets[0], 128, &movement_features);
    
    // Debug: Print feature values
    printf("\n=== FEATURE COMPARISON ===\n");
    printf("Baseline variance: %.4f, Movement variance: %.4f\n", 
           baseline_features.variance, movement_features.variance);
    printf("Baseline spatial_gradient: %.4f, Movement spatial_gradient: %.4f\n",
           baseline_features.spatial_gradient, movement_features.spatial_gradient);
    printf("==========================\n\n");
    
    // Verify that movement features are different from baseline
    // With real data, movement should have higher variance
    TEST_ASSERT_GREATER_THAN(baseline_features.variance, movement_features.variance);
    
    // Verify features are actually different
    float variance_ratio = movement_features.variance / (baseline_features.variance + 0.001f);
    TEST_ASSERT_TRUE(variance_ratio > 1.5f);  // Movement variance should be at least 1.5x baseline (real data)
}
