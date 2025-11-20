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
    // Threshold is now initialized with default value, not calibrated
    TEST_ASSERT_TRUE(ctx.adaptive_threshold > 0.0f);
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
    
    for (int i = 0; i < calib_samples && i < NUM_BASELINE_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        segmentation_add_turbulence(&ctx, turbulence);
    }
    
    segmentation_finalize_calibration(&ctx);
    
    // CRITICAL: Reset the segmentation state after calibration
    segmentation_reset(&ctx);
    
    // Process movement data and count state transitions
    int motion_transitions = 0;
    int motion_packets = 0;
    segmentation_state_t prev_state = SEG_STATE_IDLE;
    
    for (int i = 0; i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128);
        
        bool segment_completed = segmentation_add_turbulence(&ctx, turbulence);
        segmentation_state_t current_state = segmentation_get_state(&ctx);
        
        // Count transitions to MOTION state
        if (current_state == SEG_STATE_MOTION && prev_state == SEG_STATE_IDLE) {
            motion_transitions++;
            ESP_LOGI(TAG, "Motion transition #%d at packet %d", motion_transitions, i);
        }
        
        if (current_state == SEG_STATE_MOTION) {
            motion_packets++;
        }
        
        if (segment_completed) {
            ESP_LOGD(TAG, "Motion segment completed at packet %d", i);
        }
        
        prev_state = current_state;
    }
    
    ESP_LOGI(TAG, "Movement detection results:");
    ESP_LOGI(TAG, "  Motion transitions: %d", motion_transitions);
    ESP_LOGI(TAG, "  Motion packets: %d/%d (%.1f%%)", 
             motion_packets, NUM_MOVEMENT_PACKETS,
             (motion_packets * 100.0f) / NUM_MOVEMENT_PACKETS);
    
    // Verify motion was detected
    TEST_ASSERT_GREATER_THAN(0, motion_transitions);
    TEST_ASSERT_GREATER_THAN(0, motion_packets);
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
    
    // Test with second half of baseline (should have minimal motion detection)
    ESP_LOGI(TAG, "Testing baseline (should have minimal motion detection)...");
    
    int motion_packets = 0;
    for (int i = calib_samples; i < NUM_BASELINE_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128);
        segmentation_add_turbulence(&ctx, turbulence);
        
        if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
            motion_packets++;
        }
    }
    
    int total_tested = NUM_BASELINE_PACKETS - calib_samples;
    float false_positive_rate = (motion_packets * 100.0f) / total_tested;
    
    ESP_LOGI(TAG, "False positive rate: %.1f%% (%d/%d packets)", 
             false_positive_rate, motion_packets, total_tested);
    
    // Should have very few false positives (< 5%)
    TEST_ASSERT_LESS_THAN(5.0f, false_positive_rate);
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
    
    // Calibrate and detect some motion
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
    TEST_ASSERT_EQUAL(0, ctx.packet_index);
    TEST_ASSERT_TRUE(ctx.threshold_calibrated);  // Threshold preserved
}
