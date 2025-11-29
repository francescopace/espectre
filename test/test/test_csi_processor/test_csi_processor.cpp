/*
 * ESPectre - CSI Processor Unit Tests
 *
 * Unit tests for csi_processor module: initialization, parameters, state machine
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <string.h>
#include "csi_processor.h"
#include "esp_log.h"
#include "esp_system.h"

using namespace esphome::espectre;

#include "real_csi_data_esp32.h"
#include "real_csi_arrays.inc"

static const char *TAG = "test_csi_processor";

static const uint8_t TEST_SUBCARRIERS[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
static const uint8_t NUM_TEST_SUBCARRIERS = sizeof(TEST_SUBCARRIERS) / sizeof(TEST_SUBCARRIERS[0]);

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// INITIALIZATION TESTS
// ============================================================================

void test_init_with_default_parameters(void) {
    csi_processor_context_t ctx;
    
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, SEGMENTATION_DEFAULT_WINDOW_SIZE, SEGMENTATION_DEFAULT_THRESHOLD));
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.buffer_count);
    TEST_ASSERT_EQUAL(SEGMENTATION_DEFAULT_WINDOW_SIZE, ctx.window_size);
    TEST_ASSERT_EQUAL_FLOAT(SEGMENTATION_DEFAULT_THRESHOLD, ctx.threshold);
    TEST_ASSERT_NOT_NULL(ctx.turbulence_buffer);
    
    csi_processor_cleanup(&ctx);
}

void test_init_with_minimum_window_size(void) {
    csi_processor_context_t ctx;
    
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, SEGMENTATION_MIN_WINDOW_SIZE, 1.0f));
    TEST_ASSERT_EQUAL(SEGMENTATION_MIN_WINDOW_SIZE, csi_processor_get_window_size(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_init_with_maximum_window_size(void) {
    csi_processor_context_t ctx;
    
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, SEGMENTATION_MAX_WINDOW_SIZE, 1.0f));
    TEST_ASSERT_EQUAL(SEGMENTATION_MAX_WINDOW_SIZE, csi_processor_get_window_size(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_init_fails_with_null_context(void) {
    TEST_ASSERT_FALSE(csi_processor_init(NULL, 50, 1.0f));
}

void test_init_fails_with_zero_window_size(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_FALSE(csi_processor_init(&ctx, 0, 1.0f));
}

void test_init_fails_with_window_size_too_small(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_FALSE(csi_processor_init(&ctx, SEGMENTATION_MIN_WINDOW_SIZE - 1, 1.0f));
}

void test_init_fails_with_window_size_too_large(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_FALSE(csi_processor_init(&ctx, SEGMENTATION_MAX_WINDOW_SIZE + 1, 1.0f));
}

void test_init_fails_with_negative_threshold(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_FALSE(csi_processor_init(&ctx, 50, -1.0f));
}

void test_init_fails_with_threshold_below_minimum(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_FALSE(csi_processor_init(&ctx, 50, 0.4f));  // min is 0.5
}

void test_init_fails_with_threshold_above_maximum(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_FALSE(csi_processor_init(&ctx, 50, 10.1f));  // max is 10.0
}

// ============================================================================
// THRESHOLD SETTER TESTS
// ============================================================================

void test_set_threshold_valid_values(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    TEST_ASSERT_TRUE(csi_processor_set_threshold(&ctx, 0.5f));
    TEST_ASSERT_EQUAL_FLOAT(0.5f, csi_processor_get_threshold(&ctx));
    
    TEST_ASSERT_TRUE(csi_processor_set_threshold(&ctx, 5.0f));
    TEST_ASSERT_EQUAL_FLOAT(5.0f, csi_processor_get_threshold(&ctx));
    
    TEST_ASSERT_TRUE(csi_processor_set_threshold(&ctx, 10.0f));
    TEST_ASSERT_EQUAL_FLOAT(10.0f, csi_processor_get_threshold(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_set_threshold_rejects_invalid_and_preserves_previous(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 2.0f));
    
    // Try invalid threshold
    TEST_ASSERT_FALSE(csi_processor_set_threshold(&ctx, -1.0f));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, csi_processor_get_threshold(&ctx));  // unchanged
    
    csi_processor_cleanup(&ctx);
}

void test_set_threshold_rejects_null_context(void) {
    TEST_ASSERT_FALSE(csi_processor_set_threshold(NULL, 1.0f));
}

// ============================================================================
// RESET TESTS
// ============================================================================

void test_reset_clears_state_but_preserves_parameters(void) {
    csi_set_subcarrier_selection(TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 30, 1.5f));
    
    // Process packets to change state
    for (int i = 0; i < 50 && i < num_movement; i++) {
        csi_process_packet(&ctx, (const int8_t*)movement_packets[i], 128, 
                          TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    }
    
    float threshold_before = csi_processor_get_threshold(&ctx);
    uint16_t window_before = csi_processor_get_window_size(&ctx);
    
    csi_processor_reset(&ctx);
    
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    TEST_ASSERT_EQUAL(0, ctx.packet_index);
    TEST_ASSERT_EQUAL_FLOAT(threshold_before, csi_processor_get_threshold(&ctx));
    TEST_ASSERT_EQUAL(window_before, csi_processor_get_window_size(&ctx));
    
    csi_processor_cleanup(&ctx);
}

// ============================================================================
// INPUT VALIDATION TESTS
// ============================================================================

void test_process_packet_handles_null_data(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    csi_process_packet(&ctx, NULL, 128, TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    
    csi_processor_cleanup(&ctx);
}

void test_process_packet_handles_zero_length(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    int8_t dummy[128] = {0};
    csi_process_packet(&ctx, dummy, 0, TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    
    csi_processor_cleanup(&ctx);
}

void test_process_packet_handles_null_subcarriers(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    int8_t dummy[128] = {0};
    csi_process_packet(&ctx, dummy, 128, NULL, 0);
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, ctx.state);
    
    csi_processor_cleanup(&ctx);
}

// ============================================================================
// MEMORY TESTS
// ============================================================================

void test_cleanup_deallocates_buffer(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    TEST_ASSERT_NOT_NULL(ctx.turbulence_buffer);
    
    csi_processor_cleanup(&ctx);
    TEST_ASSERT_NULL(ctx.turbulence_buffer);
}

void test_stress_no_memory_leak(void) {
    csi_set_subcarrier_selection(TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    
    size_t heap_before = esp_get_free_heap_size();
    ESP_LOGI(TAG, "Heap before stress test: %zu bytes", heap_before);
    
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    // Process 10,000 packets
    int total_packets = 0;
    for (int round = 0; round < 5; round++) {
        for (int i = 0; i < num_baseline; i++) {
            csi_process_packet(&ctx, (const int8_t*)baseline_packets[i], 128,
                              TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
            total_packets++;
        }
        for (int i = 0; i < num_movement; i++) {
            csi_process_packet(&ctx, (const int8_t*)movement_packets[i], 128,
                              TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
            total_packets++;
        }
    }
    
    csi_processor_cleanup(&ctx);
    
    size_t heap_after = esp_get_free_heap_size();
    int heap_diff = (int)heap_before - (int)heap_after;
    
    ESP_LOGI(TAG, "Stress test: %d packets, heap diff: %d bytes", total_packets, heap_diff);
    
    TEST_ASSERT_EQUAL(10000, total_packets);
    TEST_ASSERT_LESS_THAN(1024, heap_diff);  // Allow 1KB tolerance
}

int process(void) {
    UNITY_BEGIN();
    
    // Initialization tests
    RUN_TEST(test_init_with_default_parameters);
    RUN_TEST(test_init_with_minimum_window_size);
    RUN_TEST(test_init_with_maximum_window_size);
    RUN_TEST(test_init_fails_with_null_context);
    RUN_TEST(test_init_fails_with_zero_window_size);
    RUN_TEST(test_init_fails_with_window_size_too_small);
    RUN_TEST(test_init_fails_with_window_size_too_large);
    RUN_TEST(test_init_fails_with_negative_threshold);
    RUN_TEST(test_init_fails_with_threshold_below_minimum);
    RUN_TEST(test_init_fails_with_threshold_above_maximum);
    
    // Threshold setter tests
    RUN_TEST(test_set_threshold_valid_values);
    RUN_TEST(test_set_threshold_rejects_invalid_and_preserves_previous);
    RUN_TEST(test_set_threshold_rejects_null_context);
    
    // Reset tests
    RUN_TEST(test_reset_clears_state_but_preserves_parameters);
    
    // Input validation tests
    RUN_TEST(test_process_packet_handles_null_data);
    RUN_TEST(test_process_packet_handles_zero_length);
    RUN_TEST(test_process_packet_handles_null_subcarriers);
    
    // Memory tests
    RUN_TEST(test_cleanup_deallocates_buffer);
    RUN_TEST(test_stress_no_memory_leak);
    
    return UNITY_END();
}

#ifdef ARDUINO
void setup() { delay(2000); process(); }
void loop() {}
#else
int main(int argc, char **argv) { return process(); }
#endif

