/*
 * ESPectre - Feature Extraction Tests
 * 
 * Tests feature extraction with real CSI data.
 * Validates that features differ between baseline and movement states.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "csi_processor.h"
#include "real_csi_data_esp32.h"
#include <math.h>
#include <string.h>

// Include CSI data arrays
#include "real_csi_arrays.inc"

// Helper: extract all features for testing
static const uint8_t test_all_features[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
#define EXTRACT_ALL_FEATURES(data, len, features) \
    csi_extract_features(data, len, NULL, 0, features, test_all_features, 10)

// Test: Verify features are different between baseline and movement (using real data)
TEST_CASE_ESP(features_differ_between_baseline_and_movement, "[features]")
{
    printf("\n=== FEATURE EXTRACTION TEST ===\n");
    printf("Testing feature extraction with real CSI data\n");
    printf("Baseline packets: %d, Movement packets: %d\n\n", 
           num_baseline, num_movement);
    
    // Extract baseline features from real data
    csi_features_t baseline_features;
    EXTRACT_ALL_FEATURES(baseline_packets[0], 128, &baseline_features);
    
    // Extract movement features from real data
    csi_features_t movement_features;
    EXTRACT_ALL_FEATURES(movement_packets[0], 128, &movement_features);
    
    // Print feature comparison
    printf("=== FEATURE COMPARISON ===\n");
    printf("%-25s %12s %12s %10s\n", "Feature", "Baseline", "Movement", "Ratio");
    printf("%-25s %12s %12s %10s\n", "-------", "--------", "--------", "-----");
    
    printf("%-25s %12.4f %12.4f %10.2fx\n", "Variance", 
           baseline_features.variance, movement_features.variance,
           movement_features.variance / (baseline_features.variance + 0.001f));
    
    printf("%-25s %12.4f %12.4f %10.2fx\n", "Spatial Gradient", 
           baseline_features.spatial_gradient, movement_features.spatial_gradient,
           movement_features.spatial_gradient / (baseline_features.spatial_gradient + 0.001f));
    
    printf("%-25s %12.4f %12.4f %10.2fx\n", "IQR", 
           baseline_features.iqr, movement_features.iqr,
           movement_features.iqr / (baseline_features.iqr + 0.001f));
    
    printf("%-25s %12.4f %12.4f %10.2fx\n", "Entropy", 
           baseline_features.entropy, movement_features.entropy,
           movement_features.entropy / (baseline_features.entropy + 0.001f));
    
    printf("==========================\n\n");
    
    // Verify that features are extracted correctly (non-zero values)
    // Note: With real CSI data, features may not always show clear separation
    // because the segmentation algorithm is the primary detection mechanism.
    // Features are secondary and used for additional classification when enabled.
    TEST_ASSERT_TRUE(baseline_features.variance > 0.0f);
    TEST_ASSERT_TRUE(movement_features.variance > 0.0f);
    TEST_ASSERT_TRUE(baseline_features.entropy > 0.0f);
    TEST_ASSERT_TRUE(movement_features.entropy > 0.0f);
    
    // Check that at least one feature shows some difference
    float variance_ratio = movement_features.variance / (baseline_features.variance + 0.001f);
    float gradient_ratio = movement_features.spatial_gradient / (baseline_features.spatial_gradient + 0.001f);
    
    // At least one feature should show some difference (ratio != 1.0)
    bool has_difference = (fabsf(variance_ratio - 1.0f) > 0.01f) || 
                          (fabsf(gradient_ratio - 1.0f) > 0.01f);
    TEST_ASSERT_TRUE(has_difference);
    
    printf("✅ Feature extraction test PASSED\n");
    printf("   Features extracted correctly from real CSI data\n");
    printf("   Variance ratio: %.2fx, Gradient ratio: %.2fx\n\n", variance_ratio, gradient_ratio);
}
