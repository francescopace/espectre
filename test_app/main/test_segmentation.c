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
#include "real_csi_data_esp32_c6.h"
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

// Default subcarrier selection for all tests (optimized based on PCA analysis)
static const uint8_t SELECTED_SUBCARRIERS[] = {53, 21, 52, 20, 58, 54, 22, 45, 46, 51, 19, 57};
static const uint8_t NUM_SUBCARRIERS = 12;

// Test: Initialize segmentation context
TEST_CASE_ESP(segmentation_init, "[segmentation]")
{
    segmentation_context_t ctx;
    
    segmentation_init(&ctx);
    
    TEST_ASSERT_EQUAL(SEG_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.buffer_count);
    TEST_ASSERT_TRUE(ctx.adaptive_threshold > 0.0f);
    TEST_ASSERT_EQUAL(SEGMENTATION_DEFAULT_WINDOW_SIZE, ctx.window_size);
    TEST_ASSERT_EQUAL(SEGMENTATION_DEFAULT_MIN_LENGTH, ctx.min_length);
    TEST_ASSERT_EQUAL(SEGMENTATION_DEFAULT_MAX_LENGTH, ctx.max_length);
}

// Test: Parameter setters and getters
TEST_CASE_ESP(segmentation_parameters, "[segmentation]")
{
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Test K factor
    TEST_ASSERT_TRUE(segmentation_set_k_factor(&ctx, 1.5f));
    TEST_ASSERT_EQUAL_FLOAT(1.5f, segmentation_get_k_factor(&ctx));
    TEST_ASSERT_FALSE(segmentation_set_k_factor(&ctx, 0.1f));  // Too low
    TEST_ASSERT_FALSE(segmentation_set_k_factor(&ctx, 10.0f)); // Too high
    
    // Test window size
    TEST_ASSERT_TRUE(segmentation_set_window_size(&ctx, 10));
    TEST_ASSERT_EQUAL(10, segmentation_get_window_size(&ctx));
    TEST_ASSERT_FALSE(segmentation_set_window_size(&ctx, 2));   // Too low
    TEST_ASSERT_FALSE(segmentation_set_window_size(&ctx, 100)); // Too high
    
    // Test min length
    TEST_ASSERT_TRUE(segmentation_set_min_length(&ctx, 15));
    TEST_ASSERT_EQUAL(15, segmentation_get_min_length(&ctx));
    TEST_ASSERT_FALSE(segmentation_set_min_length(&ctx, 2));    // Too low
    TEST_ASSERT_FALSE(segmentation_set_min_length(&ctx, 200));  // Too high
    
    // Test max length
    TEST_ASSERT_TRUE(segmentation_set_max_length(&ctx, 50));
    TEST_ASSERT_EQUAL(50, segmentation_get_max_length(&ctx));
    TEST_ASSERT_TRUE(segmentation_set_max_length(&ctx, 0));     // 0 = no limit
    TEST_ASSERT_EQUAL(0, segmentation_get_max_length(&ctx));
    TEST_ASSERT_FALSE(segmentation_set_max_length(&ctx, 5));    // Too low
    TEST_ASSERT_FALSE(segmentation_set_max_length(&ctx, 300));  // Too high
    
    // Test threshold
    TEST_ASSERT_TRUE(segmentation_set_threshold(&ctx, 0.5f));
    TEST_ASSERT_EQUAL_FLOAT(0.5f, segmentation_get_threshold(&ctx));
    TEST_ASSERT_FALSE(segmentation_set_threshold(&ctx, 0.0f));  // Too low
    TEST_ASSERT_FALSE(segmentation_set_threshold(&ctx, 15.0f)); // Too high
}

// Test: Segmentation with movement data
TEST_CASE_ESP(segmentation_movement_detection, "[segmentation]")
{
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    ESP_LOGI(TAG, "Using default threshold: %.2f", segmentation_get_threshold(&ctx));
    
    // Process movement data
    int segments_completed = 0;
    int motion_packets = 0;
    
    for (int i = 0; i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        bool segment_completed = segmentation_add_turbulence(&ctx, turbulence);
        if (segment_completed) {
            segments_completed++;
        }
        
        if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
            motion_packets++;
        }
    }
    
    ESP_LOGI(TAG, "Movement: %d packets, %d motion (%.1f%%), %d segments", 
             NUM_MOVEMENT_PACKETS, motion_packets,
             (motion_packets * 100.0f) / NUM_MOVEMENT_PACKETS,
             segments_completed);
    
    // Verify motion was detected (at least 14 segments expected based on real performance)
    TEST_ASSERT_GREATER_THAN(13, segments_completed);  // Expects ≥14 (actual: ~15)
    TEST_ASSERT_GREATER_THAN(650, motion_packets);     // Expects ≥651 (actual: ~735)
}

// Test: No false positives on baseline
TEST_CASE_ESP(segmentation_no_false_positives, "[segmentation]")
{
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Use a higher threshold to ensure no false positives on baseline
    // Note: threshold=2.0 provides good balance between sensitivity and false positives
    segmentation_set_threshold(&ctx, 2.0f);
    
    ESP_LOGI(TAG, "Testing baseline with threshold: %.2f", segmentation_get_threshold(&ctx));
    
    // Test with baseline data (should have no false segments)
    int segments_completed = 0;
    int motion_packets = 0;
    
    for (int i = 0; i < NUM_BASELINE_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        bool segment_completed = segmentation_add_turbulence(&ctx, turbulence);
        if (segment_completed) {
            segments_completed++;
        }
        
        if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
            motion_packets++;
        }
    }
    
    ESP_LOGI(TAG, "Baseline test: %d packets, %d motion (%.1f%%), %d segments (FP)", 
             NUM_BASELINE_PACKETS, motion_packets, 
             (motion_packets * 100.0f) / NUM_BASELINE_PACKETS,
             segments_completed);
    
    // Should have zero or very few false positive segments
    TEST_ASSERT_LESS_THAN(3, segments_completed);
}

// Test: Spatial turbulence calculation
TEST_CASE_ESP(spatial_turbulence_calculation, "[segmentation]")
{
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    // Test with multiple baseline packets to ensure we get valid turbulence values
    float turbulence_sum = 0.0f;
    int valid_count = 0;
    
    for (int i = 0; i < 10 && i < NUM_BASELINE_PACKETS; i++) {
        float turb = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
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
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
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
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Add some movement
    for (int i = 0; i < 50 && i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        segmentation_add_turbulence(&ctx, turbulence);
    }
    
    // Store configured parameters
    float threshold_before = segmentation_get_threshold(&ctx);
    uint16_t window_before = segmentation_get_window_size(&ctx);
    
    // Reset
    segmentation_reset(&ctx);
    
    // Verify reset (but parameters should be preserved)
    TEST_ASSERT_EQUAL(SEG_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.packet_index);
    TEST_ASSERT_EQUAL_FLOAT(threshold_before, segmentation_get_threshold(&ctx));
    TEST_ASSERT_EQUAL(window_before, segmentation_get_window_size(&ctx));
}

// Test: Configurable window size affects detection
TEST_CASE_ESP(segmentation_window_size_effect, "[segmentation]")
{
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    segmentation_context_t ctx;
    
    // Test with small window (more reactive)
    segmentation_init(&ctx);
    segmentation_set_window_size(&ctx, 5);
    
    int segments_small = 0;
    for (int i = 0; i < 200 && i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        if (segmentation_add_turbulence(&ctx, turbulence)) {
            segments_small++;
        }
    }
    
    // Test with large window (more stable)
    segmentation_init(&ctx);
    segmentation_set_window_size(&ctx, 30);
    
    int segments_large = 0;
    for (int i = 0; i < 200 && i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        if (segmentation_add_turbulence(&ctx, turbulence)) {
            segments_large++;
        }
    }
    
    ESP_LOGI(TAG, "Window size effect: small=%d segments, large=%d segments", 
             segments_small, segments_large);
    
    // Both should detect motion
    TEST_ASSERT_GREATER_THAN(0, segments_small);
    TEST_ASSERT_GREATER_THAN(0, segments_large);
}
