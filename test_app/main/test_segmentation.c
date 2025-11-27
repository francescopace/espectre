/*
 * ESPectre - CSI Processor Unit Tests
 * 
 * Unit tests for the csi_processor module (initialization, parameters, reset, edge cases)
 * Motion detection performance is tested in test_performance_suite.c
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "csi_processor.h"
#include "real_csi_data_esp32.h"
#include "espectre.h"
#include "esp_log.h"
#include "esp_system.h"
#include <stdio.h>
#include <string.h>

static const char *TAG = "test_csi_processor";

// Import CSI data arrays from real_csi_arrays.inc
#include "real_csi_arrays.inc"

// Default subcarrier selection from espectre.h (production configuration)
static const uint8_t SELECTED_SUBCARRIERS[] = DEFAULT_SUBCARRIERS;
static const uint8_t NUM_SUBCARRIERS = sizeof(SELECTED_SUBCARRIERS) / sizeof(SELECTED_SUBCARRIERS[0]);

// Helper function to process a packet
static void process_packet(csi_processor_context_t *ctx, const int8_t *packet) {
    csi_process_packet(ctx, packet, 128, SELECTED_SUBCARRIERS, NUM_SUBCARRIERS, NULL, NULL, 0);
}

// Test: Initialize CSI processor context
TEST_CASE_ESP(segmentation_init, "[csi_processor]")
{
    csi_processor_context_t ctx;
    
    csi_processor_init(&ctx);
    
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.buffer_count);
    TEST_ASSERT_TRUE(ctx.threshold > 0.0f);
    TEST_ASSERT_EQUAL(SEGMENTATION_DEFAULT_WINDOW_SIZE, ctx.window_size);
}

// Test: Parameter setters and getters
TEST_CASE_ESP(segmentation_parameters, "[csi_processor]")
{
    csi_processor_context_t ctx;
    csi_processor_init(&ctx);
    
    // Test window size setter/getter
    TEST_ASSERT_TRUE(csi_processor_set_window_size(&ctx, 10));
    TEST_ASSERT_EQUAL(10, csi_processor_get_window_size(&ctx));
    
    TEST_ASSERT_TRUE(csi_processor_set_window_size(&ctx, 100));
    TEST_ASSERT_EQUAL(100, csi_processor_get_window_size(&ctx));
    
    // Test invalid window size (too small)
    TEST_ASSERT_FALSE(csi_processor_set_window_size(&ctx, 0));
    TEST_ASSERT_EQUAL(100, csi_processor_get_window_size(&ctx));  // Should remain unchanged
    
    // Test threshold setter/getter
    TEST_ASSERT_TRUE(csi_processor_set_threshold(&ctx, 0.5f));
    TEST_ASSERT_EQUAL_FLOAT(0.5f, csi_processor_get_threshold(&ctx));
    
    TEST_ASSERT_TRUE(csi_processor_set_threshold(&ctx, 2.0f));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, csi_processor_get_threshold(&ctx));
    
    // Test invalid threshold (negative)
    TEST_ASSERT_FALSE(csi_processor_set_threshold(&ctx, -1.0f));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, csi_processor_get_threshold(&ctx));  // Should remain unchanged
}

// Test: Reset functionality preserves parameters
TEST_CASE_ESP(segmentation_reset, "[csi_processor]")
{
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    csi_processor_context_t ctx;
    csi_processor_init(&ctx);
    
    // Configure custom parameters
    csi_processor_set_threshold(&ctx, 1.5f);
    csi_processor_set_window_size(&ctx, 30);
    
    // Process some packets to change state
    for (int i = 0; i < 50 && i < num_movement; i++) {
        process_packet(&ctx, (const int8_t*)movement_packets[i]);
    }
    
    // Store configured parameters
    float threshold_before = csi_processor_get_threshold(&ctx);
    uint16_t window_before = csi_processor_get_window_size(&ctx);
    
    // Reset
    csi_processor_reset(&ctx);
    
    // Verify state is reset
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.packet_index);
    
    // Verify parameters are preserved
    TEST_ASSERT_EQUAL_FLOAT(threshold_before, csi_processor_get_threshold(&ctx));
    TEST_ASSERT_EQUAL(window_before, csi_processor_get_window_size(&ctx));
}

// Test: Handles invalid input gracefully
TEST_CASE_ESP(segmentation_handles_invalid_input, "[csi_processor][edge]")
{
    csi_processor_context_t ctx;
    csi_processor_init(&ctx);
    
    // Test with NULL data - should handle gracefully without crash
    csi_process_packet(&ctx, NULL, 128, SELECTED_SUBCARRIERS, NUM_SUBCARRIERS, NULL, NULL, 0);
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    
    // Test with zero length
    int8_t dummy[128] = {0};
    csi_process_packet(&ctx, dummy, 0, SELECTED_SUBCARRIERS, NUM_SUBCARRIERS, NULL, NULL, 0);
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    
    // Test with NULL subcarriers
    csi_process_packet(&ctx, dummy, 128, NULL, 0, NULL, NULL, 0);
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
}

// Test: Stress test with many packets and memory leak check
TEST_CASE_ESP(segmentation_stress_test, "[csi_processor][stress][memory]")
{
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    // Measure heap before test
    size_t heap_before = esp_get_free_heap_size();
    ESP_LOGI(TAG, "Heap before stress test: %d bytes", heap_before);
    
    csi_processor_context_t ctx;
    csi_processor_init(&ctx);
    
    // Process all available packets multiple times
    int total_packets = 0;
    int total_motion = 0;
    
    for (int round = 0; round < 5; round++) {
        // Process baseline
        for (int i = 0; i < num_baseline; i++) {
            process_packet(&ctx, (const int8_t*)baseline_packets[i]);
            total_packets++;
            if (csi_processor_get_state(&ctx) == CSI_STATE_MOTION) {
                total_motion++;
            }
        }
        
        // Process movement
        for (int i = 0; i < num_movement; i++) {
            process_packet(&ctx, (const int8_t*)movement_packets[i]);
            total_packets++;
            if (csi_processor_get_state(&ctx) == CSI_STATE_MOTION) {
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
