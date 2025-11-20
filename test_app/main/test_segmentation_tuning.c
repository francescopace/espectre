/*
 * ESPectre - Segmentation Threshold Tuning Test
 * 
 * Tests segmentation threshold tuning with real CSI data.
 * Validates that the segmentation system can distinguish baseline from movement.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "csi_processor.h"
#include "segmentation.h"
#include "real_csi_data.h"
#include <math.h>
#include <string.h>

// Include CSI data arrays
#include "real_csi_arrays.inc"

// Test: Segmentation threshold tuning with real CSI data
TEST_CASE_ESP(segmentation_threshold_tuning_with_real_csi, "[segmentation][real]")
{
    printf("\n=== SEGMENTATION THRESHOLD TUNING TEST ===\n");
    printf("Testing segmentation with real CSI data\n");
    printf("Baseline packets: %d, Movement packets: %d\n\n", 
           num_baseline, num_movement);
    
    // Initialize segmentation context
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Process baseline packets to establish baseline turbulence
    printf("Processing BASELINE phase...\n");
    float baseline_turbulence_sum = 0.0f;
    int baseline_count = 0;
    
    for (int p = 0; p < num_baseline && p < 500; p++) {
        float turbulence = csi_calculate_spatial_turbulence(baseline_packets[p], 128);
        segmentation_add_turbulence(&ctx, turbulence);
        
        baseline_turbulence_sum += turbulence;
        baseline_count++;
        
        if (p == 0) {
            printf("  First baseline turbulence: %.4f\n", turbulence);
        }
    }
    
    float baseline_avg = baseline_turbulence_sum / baseline_count;
    float baseline_moving_variance = segmentation_get_moving_variance(&ctx);
    printf("  Baseline average turbulence: %.4f\n", baseline_avg);
    printf("  Baseline moving variance: %.4f\n", baseline_moving_variance);
    printf("  Baseline state: %s\n\n", 
           segmentation_get_state(&ctx) == SEG_STATE_IDLE ? "IDLE" : "MOTION");
    
    // Process movement packets
    printf("Processing MOVEMENT phase...\n");
    float movement_turbulence_sum = 0.0f;
    int movement_count = 0;
    int motion_detections = 0;
    
    for (int p = 0; p < num_movement && p < 500; p++) {
        float turbulence = csi_calculate_spatial_turbulence(movement_packets[p], 128);
        segmentation_add_turbulence(&ctx, turbulence);
        
        movement_turbulence_sum += turbulence;
        movement_count++;
        
        if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
            motion_detections++;
        }
        
        if (p == 0) {
            printf("  First movement turbulence: %.4f\n", turbulence);
        }
    }
    
    float movement_avg = movement_turbulence_sum / movement_count;
    float movement_moving_variance = segmentation_get_moving_variance(&ctx);
    printf("  Movement average turbulence: %.4f\n", movement_avg);
    printf("  Movement moving variance: %.4f\n", movement_moving_variance);
    printf("  Motion detections: %d/%d (%.1f%%)\n", 
           motion_detections, movement_count, 
           (motion_detections * 100.0f) / movement_count);
    
    // Calculate separation metrics
    float turbulence_ratio = movement_avg / (baseline_avg + 0.001f);
    float variance_ratio = movement_moving_variance / (baseline_moving_variance + 0.001f);
    
    printf("\n=== SEPARATION METRICS ===\n");
    printf("Turbulence ratio (movement/baseline): %.2fx\n", turbulence_ratio);
    printf("Moving variance ratio: %.2fx\n", variance_ratio);
    printf("Current threshold: %.4f\n", segmentation_get_threshold(&ctx));
    printf("Motion detection rate: %.1f%%\n", (motion_detections * 100.0f) / movement_count);
    printf("==========================\n\n");
    
    // Verify that movement is distinguishable from baseline
    // NOTE: Real CSI data has low turbulence separation (movement/baseline ~1.04x)
    // This is NORMAL - the key is the moving variance ratio (2.25x)
    // Segmentation works on variance of turbulence, not turbulence itself
    
    // Verify turbulence ratio is positive (movement >= baseline)
    TEST_ASSERT_GREATER_OR_EQUAL(baseline_avg, movement_avg);
    TEST_ASSERT_TRUE(turbulence_ratio >= 1.0f);
    
    // Verify that moving variance ratio is good (this is what matters!)
    TEST_ASSERT_TRUE(variance_ratio > 1.5f);  // At least 1.5x separation in variance
    
    // Verify that motion was detected
    TEST_ASSERT_GREATER_THAN(0, motion_detections);
    
    printf("✅ Segmentation threshold tuning test PASSED\n");
    printf("   Movement turbulence is %.2fx higher than baseline\n", turbulence_ratio);
    printf("   Motion detected in %.1f%% of movement packets\n\n", 
           (motion_detections * 100.0f) / movement_count);
}
