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
#include "real_csi_data_esp32_c6.h"
#include <math.h>
#include <string.h>

// Include CSI data arrays
#include "real_csi_arrays.inc"

// Default subcarrier selection for all tests (optimized based on PCA analysis)
static const uint8_t SELECTED_SUBCARRIERS[] = {53, 21, 52, 20, 58, 54, 22, 45, 46, 51, 19, 57};
static const uint8_t NUM_SUBCARRIERS = 12;

// Test: Segmentation threshold tuning with real CSI data
TEST_CASE_ESP(segmentation_threshold_tuning_with_real_csi, "[segmentation][real]")
{
    printf("\n=== SEGMENTATION TUNING TEST ===\n");
    
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    printf("Using default threshold: %.4f\n", segmentation_get_threshold(&ctx));
    
    // Test baseline (should have no or very few segments)
    int baseline_segments = 0;
    
    for (int p = 0; p < 500 && p < num_baseline; p++) {
        float turbulence = csi_calculate_spatial_turbulence(baseline_packets[p], 128,
                                                            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        if (segmentation_add_turbulence(&ctx, turbulence)) {
            baseline_segments++;
        }
    }
    
    // Test movement (should have segments)
    segmentation_reset(&ctx);
    int movement_segments = 0;
    int motion_packets = 0;
    
    for (int p = 0; p < 500 && p < num_movement; p++) {
        float turbulence = csi_calculate_spatial_turbulence(movement_packets[p], 128,
                                                            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        if (segmentation_add_turbulence(&ctx, turbulence)) {
            movement_segments++;
        }
        if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
            motion_packets++;
        }
    }
    
    printf("Results: baseline=%d FP, movement=%d segments (%.1f%% motion packets)\n",
           baseline_segments, movement_segments,
           (motion_packets * 100.0f) / 500);
    printf("Moving variance: %.4f, Threshold: %.4f\n",
           segmentation_get_moving_variance(&ctx), segmentation_get_threshold(&ctx));
    printf("================================\n\n");
    
    // Verify performance (updated based on real performance: 0 FP, 7 segments)
    TEST_ASSERT_LESS_THAN(1, baseline_segments);     // Expects 0 FP (actual: 0)
    TEST_ASSERT_GREATER_THAN(6, movement_segments);  // Expects ≥7 segments (actual: 7)
}

// Test: Different threshold values
TEST_CASE_ESP(segmentation_threshold_comparison, "[segmentation][tuning]")
{
    printf("\n=== THRESHOLD COMPARISON TEST ===\n");
    
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    float thresholds[] = {0.2f, 0.3f, 0.5f, 1.0f};
    int num_thresholds = sizeof(thresholds) / sizeof(thresholds[0]);
    
    for (int t = 0; t < num_thresholds; t++) {
        segmentation_context_t ctx;
        segmentation_init(&ctx);
        segmentation_set_threshold(&ctx, thresholds[t]);
        
        // Test with movement data
        int segments = 0;
        int motion_packets = 0;
        
        for (int p = 0; p < 500 && p < num_movement; p++) {
            float turbulence = csi_calculate_spatial_turbulence(movement_packets[p], 128,
                                                                SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
            if (segmentation_add_turbulence(&ctx, turbulence)) {
                segments++;
            }
            if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
                motion_packets++;
            }
        }
        
        printf("Threshold %.2f: %d segments, %.1f%% motion\n",
               thresholds[t], segments, (motion_packets * 100.0f) / 500);
    }
    
    printf("================================\n\n");
}
