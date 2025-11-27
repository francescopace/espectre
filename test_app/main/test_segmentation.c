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
#include "real_csi_data_esp32.h"
#include "espectre.h"
#include "esp_log.h"
#include "esp_system.h"
#include <stdio.h>
#include <string.h>
#include <math.h>

static const char *TAG = "test_segmentation";

// Import CSI data arrays from real_csi_arrays.inc
#include "real_csi_arrays.inc"

// Use the arrays defined in real_csi_arrays.inc
#define NUM_BASELINE_PACKETS num_baseline
#define NUM_MOVEMENT_PACKETS num_movement

// Default subcarrier selection from espectre.h (production configuration)
static const uint8_t SELECTED_SUBCARRIERS[] = DEFAULT_SUBCARRIERS;
static const uint8_t NUM_SUBCARRIERS = sizeof(SELECTED_SUBCARRIERS) / sizeof(SELECTED_SUBCARRIERS[0]);

// Test: Initialize segmentation context
TEST_CASE_ESP(segmentation_init, "[segmentation]")
{
    segmentation_context_t ctx;
    
    segmentation_init(&ctx);
    
    TEST_ASSERT_EQUAL(SEG_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.buffer_count);
    TEST_ASSERT_TRUE(ctx.threshold > 0.0f);
    TEST_ASSERT_EQUAL(SEGMENTATION_DEFAULT_WINDOW_SIZE, ctx.window_size);
}

// Test: Parameter setters and getters
TEST_CASE_ESP(segmentation_parameters, "[segmentation]")
{
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Test window size
    TEST_ASSERT_TRUE(segmentation_set_window_size(&ctx, 10));
    TEST_ASSERT_EQUAL(10, segmentation_get_window_size(&ctx));
    TEST_ASSERT_FALSE(segmentation_set_window_size(&ctx, 2));   // Too low
    TEST_ASSERT_FALSE(segmentation_set_window_size(&ctx, 100)); // Too high
    
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
    segmentation_state_t prev_state = SEG_STATE_IDLE;
    
    for (int i = 0; i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        segmentation_add_turbulence(&ctx, turbulence);
        
        segmentation_state_t current_state = segmentation_get_state(&ctx);
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == SEG_STATE_MOTION && current_state == SEG_STATE_IDLE) {
            segments_completed++;
        }
        
        if (current_state == SEG_STATE_MOTION) {
            motion_packets++;
        }
        
        prev_state = current_state;
    }
    
    ESP_LOGI(TAG, "Movement: %d packets, %d motion (%.1f%%), %d segments", 
             NUM_MOVEMENT_PACKETS, motion_packets,
             (motion_packets * 100.0f) / NUM_MOVEMENT_PACKETS,
             segments_completed);
    
    // Verify motion was detected
    // Count motion packets (not segments) - should have >94% recall
    TEST_ASSERT_GREATER_THAN(940, motion_packets);     // Expects >94% recall (actual: ~94.7%)
    TEST_ASSERT_GREATER_THAN(0, segments_completed);   // At least 1 segment completed
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
    segmentation_state_t prev_state = SEG_STATE_IDLE;
    
    for (int i = 0; i < NUM_BASELINE_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        segmentation_add_turbulence(&ctx, turbulence);
        
        segmentation_state_t current_state = segmentation_get_state(&ctx);
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == SEG_STATE_MOTION && current_state == SEG_STATE_IDLE) {
            segments_completed++;
        }
        
        if (current_state == SEG_STATE_MOTION) {
            motion_packets++;
        }
        
        prev_state = current_state;
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
    segmentation_state_t prev_state_small = SEG_STATE_IDLE;
    
    for (int i = 0; i < 200 && i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        segmentation_add_turbulence(&ctx, turbulence);
        
        segmentation_state_t current_state = segmentation_get_state(&ctx);
        if (prev_state_small == SEG_STATE_MOTION && current_state == SEG_STATE_IDLE) {
            segments_small++;
        }
        prev_state_small = current_state;
    }
    
    // Test with large window (more stable)
    segmentation_init(&ctx);
    segmentation_set_window_size(&ctx, 30);
    
    int segments_large = 0;
    segmentation_state_t prev_state_large = SEG_STATE_IDLE;
    
    for (int i = 0; i < 200 && i < NUM_MOVEMENT_PACKETS; i++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[i], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        segmentation_add_turbulence(&ctx, turbulence);
        
        segmentation_state_t current_state = segmentation_get_state(&ctx);
        if (prev_state_large == SEG_STATE_MOTION && current_state == SEG_STATE_IDLE) {
            segments_large++;
        }
        prev_state_large = current_state;
    }
    
    ESP_LOGI(TAG, "Window size effect: small=%d segments, large=%d segments", 
             segments_small, segments_large);
    
    // Both should detect motion
    TEST_ASSERT_GREATER_THAN(0, segments_small);
    TEST_ASSERT_GREATER_THAN(0, segments_large);
}

// Test: Segmentation handles invalid input gracefully
TEST_CASE_ESP(segmentation_handles_invalid_input, "[segmentation][edge]")
{
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Test with negative turbulence (should be treated as 0 or ignored)
    segmentation_add_turbulence(&ctx, -1.0f);
    TEST_ASSERT_EQUAL(SEG_STATE_IDLE, ctx.state);
    
    // Test with very large turbulence
    segmentation_add_turbulence(&ctx, 1000000.0f);
    // Should handle gracefully without crash
    TEST_ASSERT_TRUE(ctx.state == SEG_STATE_IDLE || ctx.state == SEG_STATE_MOTION);
    
    // Reset and test with NaN (if supported)
    segmentation_reset(&ctx);
    float nan_val = 0.0f / 0.0f;  // Generate NaN
    segmentation_add_turbulence(&ctx, nan_val);
    // Should not crash, state should remain valid
    TEST_ASSERT_TRUE(ctx.state == SEG_STATE_IDLE || ctx.state == SEG_STATE_MOTION);
    
    // Test with infinity
    segmentation_reset(&ctx);
    float inf_val = 1.0f / 0.0f;  // Generate infinity
    segmentation_add_turbulence(&ctx, inf_val);
    // Should not crash
    TEST_ASSERT_TRUE(ctx.state == SEG_STATE_IDLE || ctx.state == SEG_STATE_MOTION);
}

// Test: Segmentation stress test with many packets and memory leak check
TEST_CASE_ESP(segmentation_stress_test, "[segmentation][stress][memory]")
{
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    // Measure heap before test
    size_t heap_before = esp_get_free_heap_size();
    ESP_LOGI(TAG, "Heap before stress test: %d bytes", heap_before);
    
    segmentation_context_t ctx;
    segmentation_init(&ctx);
    
    // Process all available packets multiple times
    int total_packets = 0;
    int total_motion = 0;
    
    for (int round = 0; round < 5; round++) {
        // Process baseline
        for (int i = 0; i < NUM_BASELINE_PACKETS; i++) {
            float turbulence = csi_calculate_spatial_turbulence(
                (const int8_t*)baseline_packets[i], 128,
                SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
            segmentation_add_turbulence(&ctx, turbulence);
            total_packets++;
            if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
                total_motion++;
            }
        }
        
        // Process movement
        for (int i = 0; i < NUM_MOVEMENT_PACKETS; i++) {
            float turbulence = csi_calculate_spatial_turbulence(
                (const int8_t*)movement_packets[i], 128,
                SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
            segmentation_add_turbulence(&ctx, turbulence);
            total_packets++;
            if (segmentation_get_state(&ctx) == SEG_STATE_MOTION) {
                total_motion++;
            }
        }
    }
    
    // Measure heap after test
    size_t heap_after = esp_get_free_heap_size();
    int heap_diff = (int)heap_before - (int)heap_after;
    
    ESP_LOGI(TAG, "Stress test: %d packets processed, %d motion (%.1f%%)", 
             total_packets, total_motion, (total_motion * 100.0f) / total_packets);
    ESP_LOGI(TAG, "Heap after stress test: %d bytes (diff: %d bytes)", 
             heap_after, heap_diff);
    
    // Should have processed all packets without crash
    TEST_ASSERT_EQUAL(10000, total_packets);  // 5 rounds * 2000 packets
    // Should have detected some motion
    TEST_ASSERT_GREATER_THAN(0, total_motion);
    // Memory leak check: allow up to 1KB tolerance for fragmentation
    TEST_ASSERT_LESS_THAN(1024, heap_diff);
}
