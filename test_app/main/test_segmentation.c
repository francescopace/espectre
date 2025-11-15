/*
 * ESPectre - Segmentation Module Test
 * 
 * Tests the Moving Variance Segmentation (MVS) algorithm with real CSI data
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "segmentation.h"
#include "csi_processor.h"
#include "real_csi_data.h"
#include "esp_log.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static const char *TAG = "test_segmentation";

// Import CSI data arrays from real_csi_arrays.inc
#include "real_csi_arrays.inc"

// Use the arrays defined in real_csi_arrays.inc
#define NUM_BASELINE_PACKETS num_baseline
#define NUM_MOVEMENT_PACKETS num_movement

// Test: Initialize segmentation context
TEST_CASE_ESP(segmentation_init, "[segmentation]")
{
    segmentation_context_t ctx;
    
    segmentation_init(&ctx);
    
    TEST_ASSERT_EQUAL(SEG_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.buffer_count);
    TEST_ASSERT_EQUAL(0, ctx.num_segments);
    TEST_ASSERT_FALSE(ctx.threshold_calibrated);
    TEST_ASSERT_FALSE(ctx.calibrating);
}

// Test: Calibration with baseline data
TEST_CASE_ESP(segmentation_calibration, "[segmentation]")
{
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Start calibration with baseline data
    uint32_t num_samples = 200;  // Use first 200 baseline packets
    TEST_ASSERT_TRUE(segmentation_start_calibration(&ctx, num_samples));
    TEST_ASSERT_TRUE(ctx.calibrating);
    
    // Feed baseline packets
    for (int i = 0; i < num_samples && i < NUM_BASELINE_PACKETS; i++) {
        // Calculate spatial turbulence
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        
        // Add to segmentation
        segmentation_add_turbulence(&ctx, turbulence);
    }
    
    // Finalize calibration
    TEST_ASSERT_TRUE(segmentation_finalize_calibration(&ctx));
    TEST_ASSERT_TRUE(ctx.threshold_calibrated);
    TEST_ASSERT_FALSE(ctx.calibrating);
    
    // Check threshold is reasonable
    TEST_ASSERT_GREATER_THAN(0.0f, ctx.adaptive_threshold);
    TEST_ASSERT_LESS_THAN(10.0f, ctx.adaptive_threshold);
}

// Test: Segmentation with movement data
TEST_CASE_ESP(segmentation_movement_detection, "[segmentation]")
{
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Calibrate with baseline
    uint32_t calib_samples = 1000;
    segmentation_start_calibration(&ctx, calib_samples);
    
    // Store turbulence values for JSON output
    float *baseline_turbulence = malloc(calib_samples * sizeof(float));
    float *baseline_moving_var = malloc(calib_samples * sizeof(float));
    
    for (int i = 0; i < calib_samples && i < NUM_BASELINE_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        baseline_turbulence[i] = turbulence;
        segmentation_add_turbulence(&ctx, turbulence);
    }
    
    segmentation_finalize_calibration(&ctx);
    
    // Extract moving variance from calibration (for visualization)
    // Note: This is a simplified extraction - in real code, we'd need to access internal buffer
    for (int i = 0; i < calib_samples; i++) {
        baseline_moving_var[i] = 0.0f;  // Placeholder - will be calculated in Python
    }
    
    // CRITICAL: Reset the segmentation state after calibration
    segmentation_reset(&ctx);
    
    // Process movement data
    int segments_detected = 0;
    float *movement_turbulence = malloc(NUM_MOVEMENT_PACKETS * sizeof(float));
    float *movement_moving_var = malloc(NUM_MOVEMENT_PACKETS * sizeof(float));
    
    for (int i = 0; i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128);
        movement_turbulence[i] = turbulence;
        
        bool segment_completed = segmentation_add_turbulence(&ctx, turbulence);
        
        if (segment_completed) {
            segments_detected++;
            const segment_t *seg = segmentation_get_segment(&ctx, ctx.num_segments - 1);
            if (seg) {
                ESP_LOGI(TAG, "Segment #%d: start=%lu, length=%d (%.2fs), avg=%.2f, max=%.2f",
                         segments_detected,
                         (unsigned long)seg->start_index,
                         seg->length,
                         seg->length / 20.0f,
                         seg->avg_turbulence,
                         seg->max_turbulence);
            }
        }
        
        movement_moving_var[i] = 0.0f;  // Placeholder
    }
    
    // JSON output for Python visualization
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  JSON OUTPUT (for segmentation visualization)\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("{\n");
    printf("  \"test_name\": \"segmentation_movement_detection\",\n");
    printf("  \"threshold\": %.4f,\n", ctx.adaptive_threshold);
    printf("  \"mean_variance\": %.4f,\n", ctx.baseline_mean_variance);
    printf("  \"std_variance\": %.4f,\n", ctx.baseline_std_variance);
    printf("  \"window_size\": %d,\n", SEGMENTATION_WINDOW_SIZE);
    printf("  \"baseline_turbulence\": [");
    for (int i = 0; i < calib_samples && i < NUM_BASELINE_PACKETS; i++) {
        printf("%.4f%s", baseline_turbulence[i], i < calib_samples - 1 && i < NUM_BASELINE_PACKETS - 1 ? ", " : "");
    }
    printf("],\n");
    printf("  \"movement_turbulence\": [");
    for (int i = 0; i < NUM_MOVEMENT_PACKETS; i++) {
        printf("%.4f%s", movement_turbulence[i], i < NUM_MOVEMENT_PACKETS - 1 ? ", " : "");
    }
    printf("],\n");
    printf("  \"baseline_segments\": [],\n");
    printf("  \"movement_segments\": [\n");
    for (int i = 0; i < ctx.num_segments; i++) {
        const segment_t *seg = segmentation_get_segment(&ctx, i);
        if (seg) {
            printf("    {\"start\": %lu, \"length\": %d, \"avg\": %.4f, \"max\": %.4f}%s\n",
                   (unsigned long)seg->start_index,
                   seg->length,
                   seg->avg_turbulence,
                   seg->max_turbulence,
                   i < ctx.num_segments - 1 ? "," : "");
        }
    }
    printf("  ]\n");
    printf("}\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Cleanup
    free(baseline_turbulence);
    free(baseline_moving_var);
    free(movement_turbulence);
    free(movement_moving_var);
    
    // Verify segments were detected
    TEST_ASSERT_GREATER_THAN(0, ctx.num_segments);
    TEST_ASSERT_EQUAL(segments_detected, ctx.num_segments);
    TEST_ASSERT_GREATER_OR_EQUAL(10, ctx.num_segments);
    TEST_ASSERT_LESS_OR_EQUAL(20, ctx.num_segments);
}

// Test: No false positives on baseline
TEST_CASE_ESP(segmentation_no_false_positives, "[segmentation]")
{
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Calibrate with first half of baseline
    uint32_t calib_samples = NUM_BASELINE_PACKETS / 2;
    segmentation_start_calibration(&ctx, calib_samples);
    
    for (int i = 0; i < calib_samples; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        segmentation_add_turbulence(&ctx, turbulence);
    }
    
    segmentation_finalize_calibration(&ctx);
    
    // Test with second half of baseline (should have 0 segments)
    ESP_LOGI(TAG, "Testing baseline (should have 0 segments)...");
    
    for (int i = calib_samples; i < NUM_BASELINE_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        segmentation_add_turbulence(&ctx, turbulence);
    }
    
    // Should have 0 or very few segments (no false positives)
    TEST_ASSERT_LESS_OR_EQUAL(2, ctx.num_segments);  // Allow max 2 false positives
}

// Test: Spatial turbulence calculation
TEST_CASE_ESP(spatial_turbulence_calculation, "[segmentation]")
{
    // Test with multiple baseline packets to ensure we get valid turbulence values
    // With subcarrier filtering (47-58), some packets may have zero turbulence
    float turbulence_sum = 0.0f;
    int valid_count = 0;
    
    for (int i = 0; i < 10 && i < NUM_BASELINE_PACKETS; i++) {
        float turb = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        if (turb > 0.0f) {
            turbulence_sum += turb;
            valid_count++;
        }
    }
    
    // At least some packets should have positive turbulence
    TEST_ASSERT_GREATER_THAN(0, valid_count);
    float avg_baseline_turb = turbulence_sum / valid_count;
    TEST_ASSERT_GREATER_THAN(0.0f, avg_baseline_turb);
    TEST_ASSERT_LESS_THAN(20.0f, avg_baseline_turb);
    
    // Test with movement packets
    turbulence_sum = 0.0f;
    valid_count = 0;
    
    for (int i = 0; i < 10 && i < NUM_MOVEMENT_PACKETS; i++) {
        float turb = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128);
        if (turb > 0.0f) {
            turbulence_sum += turb;
            valid_count++;
        }
    }
    
    TEST_ASSERT_GREATER_THAN(0, valid_count);
    float avg_movement_turb = turbulence_sum / valid_count;
    TEST_ASSERT_GREATER_THAN(0.0f, avg_movement_turb);
}

// Test: Reset functionality
TEST_CASE_ESP(segmentation_reset, "[segmentation]")
{
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Calibrate and detect some segments
    // NOTE: Need to provide target + WINDOW_SIZE samples because the first
    //       WINDOW_SIZE samples are used to fill the circular buffer before
    //       variance collection begins. So for 100 target samples, we need
    //       to provide 130 total samples (100 + 30).
    segmentation_start_calibration(&ctx, 100);
    for (int i = 0; i < 130 && i < NUM_BASELINE_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        segmentation_add_turbulence(&ctx, turbulence);
    }
    segmentation_finalize_calibration(&ctx);
    
    // Add some movement
    for (int i = 0; i < 50 && i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128);
        segmentation_add_turbulence(&ctx, turbulence);
    }
    
    // Reset
    segmentation_reset(&ctx);
    
    // Verify reset (but threshold should be preserved)
    TEST_ASSERT_EQUAL(SEG_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.num_segments);
    TEST_ASSERT_EQUAL(0, ctx.packet_index);
    TEST_ASSERT_TRUE(ctx.threshold_calibrated);  // Threshold preserved
}
