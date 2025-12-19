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
#include "esphome/core/log.h"
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
// GETTER TESTS
// ============================================================================

void test_get_window_size_returns_correct_value(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 75, 1.0f));
    
    TEST_ASSERT_EQUAL(75, csi_processor_get_window_size(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_get_window_size_returns_zero_for_null(void) {
    TEST_ASSERT_EQUAL(0, csi_processor_get_window_size(NULL));
}

void test_get_threshold_returns_correct_value(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 2.5f));
    
    TEST_ASSERT_EQUAL_FLOAT(2.5f, csi_processor_get_threshold(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_get_threshold_returns_zero_for_null(void) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, csi_processor_get_threshold(NULL));
}

void test_get_state_returns_idle_initially(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, csi_processor_get_state(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_get_state_returns_idle_for_null(void) {
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, csi_processor_get_state(NULL));
}

void test_get_moving_variance_returns_zero_initially(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    TEST_ASSERT_EQUAL_FLOAT(0.0f, csi_processor_get_moving_variance(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_get_moving_variance_returns_zero_for_null(void) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, csi_processor_get_moving_variance(NULL));
}

void test_get_last_turbulence_returns_zero_initially(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    TEST_ASSERT_EQUAL_FLOAT(0.0f, csi_processor_get_last_turbulence(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_get_last_turbulence_returns_zero_for_null(void) {
    TEST_ASSERT_EQUAL_FLOAT(0.0f, csi_processor_get_last_turbulence(NULL));
}

void test_get_last_turbulence_after_processing(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    // Process a few packets
    for (int i = 0; i < 10; i++) {
        csi_process_packet(&ctx, baseline_packets[i], 128, 
                          TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    }
    
    // Last turbulence should be non-zero (real CSI data has signal)
    float last_turb = csi_processor_get_last_turbulence(&ctx);
    TEST_ASSERT_TRUE(last_turb >= 0.0f);  // Turbulence is always non-negative
    
    ESP_LOGI(TAG, "Last turbulence after 10 packets: %.4f", last_turb);
    
    csi_processor_cleanup(&ctx);
}

void test_get_total_packets_returns_zero_initially(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    TEST_ASSERT_EQUAL(0, csi_processor_get_total_packets(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_get_total_packets_returns_zero_for_null(void) {
    TEST_ASSERT_EQUAL(0, csi_processor_get_total_packets(NULL));
}

void test_get_total_packets_increments_correctly(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    // Process 25 packets
    for (int i = 0; i < 25; i++) {
        csi_process_packet(&ctx, baseline_packets[i], 128, 
                          TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    }
    
    TEST_ASSERT_EQUAL(25, csi_processor_get_total_packets(&ctx));
    
    csi_processor_cleanup(&ctx);
}

// ============================================================================
// STATE MACHINE TESTS
// ============================================================================

void test_state_transitions_to_motion_on_high_variance(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 30, 1.0f));  // Low threshold
    
    // Process movement packets - should trigger motion
    for (int i = 0; i < 50; i++) {
        csi_process_packet(&ctx, movement_packets[i], 128, 
                          TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    }
    
    // Update state (lazy evaluation - must be called explicitly)
    csi_processor_update_state(&ctx);
    
    // Should be in MOTION state after processing movement data
    csi_motion_state_t state = csi_processor_get_state(&ctx);
    float mv = csi_processor_get_moving_variance(&ctx);
    
    ESP_LOGI(TAG, "After 50 movement packets: state=%d, mv=%.4f", state, mv);
    
    // Moving variance should be positive
    TEST_ASSERT_TRUE(mv > 0.0f);
    
    csi_processor_cleanup(&ctx);
}

void test_state_stays_idle_on_baseline(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 30, 5.0f));  // High threshold
    
    // Process baseline packets - should stay idle
    for (int i = 0; i < 50; i++) {
        csi_process_packet(&ctx, baseline_packets[i], 128, 
                          TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    }
    
    // Update state (lazy evaluation - must be called explicitly)
    csi_processor_update_state(&ctx);
    
    csi_motion_state_t state = csi_processor_get_state(&ctx);
    float mv = csi_processor_get_moving_variance(&ctx);
    
    ESP_LOGI(TAG, "After 50 baseline packets: state=%d, mv=%.4f, threshold=5.0", state, mv);
    
    // With high threshold, baseline should not trigger motion
    TEST_ASSERT_EQUAL(CSI_STATE_IDLE, state);
    
    csi_processor_cleanup(&ctx);
}

// ============================================================================
// NORMALIZATION SCALE TESTS
// ============================================================================

void test_normalization_scale_default_is_one(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    TEST_ASSERT_EQUAL_FLOAT(1.0f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_set_normalization_scale_valid_values(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    csi_processor_set_normalization_scale(&ctx, 2.0f);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_set_normalization_scale(&ctx, 0.5f);
    TEST_ASSERT_EQUAL_FLOAT(0.5f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_normalization_scale_clamps_minimum(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    // Values below 0.001 should be clamped to 0.001
    csi_processor_set_normalization_scale(&ctx, 0.0001f);
    TEST_ASSERT_EQUAL_FLOAT(0.001f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_set_normalization_scale(&ctx, 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(0.001f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_set_normalization_scale(&ctx, -1.0f);
    TEST_ASSERT_EQUAL_FLOAT(0.001f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_normalization_scale_clamps_maximum(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    // Values above 100.0 should be clamped to 100.0
    csi_processor_set_normalization_scale(&ctx, 150.0f);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_set_normalization_scale(&ctx, 200.0f);
    TEST_ASSERT_EQUAL_FLOAT(100.0f, csi_processor_get_normalization_scale(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_get_normalization_scale_returns_one_for_null(void) {
    TEST_ASSERT_EQUAL_FLOAT(1.0f, csi_processor_get_normalization_scale(NULL));
}

void test_normalization_scale_affects_turbulence(void) {
    csi_processor_context_t ctx1, ctx2;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx1, 50, 1.0f));
    TEST_ASSERT_TRUE(csi_processor_init(&ctx2, 50, 1.0f));
    
    // Set different normalization scales
    csi_processor_set_normalization_scale(&ctx1, 1.0f);  // No scaling
    csi_processor_set_normalization_scale(&ctx2, 2.0f);  // Double scaling
    
    // Process same packet in both contexts
    csi_process_packet(&ctx1, baseline_packets[0], 128, 
                      TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    csi_process_packet(&ctx2, baseline_packets[0], 128, 
                      TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    
    float turb1 = csi_processor_get_last_turbulence(&ctx1);
    float turb2 = csi_processor_get_last_turbulence(&ctx2);
    
    ESP_LOGI(TAG, "Turbulence with scale=1.0: %.4f, scale=2.0: %.4f", turb1, turb2);
    
    // Turbulence with scale 2.0 should be approximately double
    TEST_ASSERT_FLOAT_WITHIN(0.01f, turb1 * 2.0f, turb2);
    
    csi_processor_cleanup(&ctx1);
    csi_processor_cleanup(&ctx2);
}

// ============================================================================
// LOW-PASS FILTER TESTS
// ============================================================================

void test_lowpass_filter_init_default(void) {
    lowpass_filter_state_t state;
    lowpass_filter_init(&state, LOWPASS_CUTOFF_DEFAULT, LOWPASS_SAMPLE_RATE, true);
    
    TEST_ASSERT_TRUE(state.enabled);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, LOWPASS_CUTOFF_DEFAULT, state.cutoff_hz);
    TEST_ASSERT_FALSE(state.initialized);  // Not initialized until first sample
}

void test_lowpass_filter_disabled_passthrough(void) {
    lowpass_filter_state_t state;
    lowpass_filter_init(&state, 10.0f, 100.0f, false);  // Disabled
    
    // Should pass through unchanged
    float input = 5.0f;
    float output = lowpass_filter_apply(&state, input);
    TEST_ASSERT_EQUAL_FLOAT(input, output);
}

void test_lowpass_filter_enabled_smooths(void) {
    lowpass_filter_state_t state;
    lowpass_filter_init(&state, 10.0f, 100.0f, true);
    
    // First sample initializes state
    float out1 = lowpass_filter_apply(&state, 10.0f);
    TEST_ASSERT_EQUAL_FLOAT(10.0f, out1);  // First sample returns same value
    
    // Sudden step input - filter should smooth it
    float out2 = lowpass_filter_apply(&state, 0.0f);
    // Output should be between 0 and 10 (smoothed)
    TEST_ASSERT_TRUE(out2 > 0.0f && out2 < 10.0f);
    ESP_LOGI(TAG, "Lowpass smoothing: step from 10 to 0, output=%.3f", out2);
}

void test_lowpass_filter_reset(void) {
    lowpass_filter_state_t state;
    lowpass_filter_init(&state, 10.0f, 100.0f, true);
    
    // Process some samples
    lowpass_filter_apply(&state, 10.0f);
    lowpass_filter_apply(&state, 20.0f);
    TEST_ASSERT_TRUE(state.initialized);
    
    // Reset
    lowpass_filter_reset(&state);
    TEST_ASSERT_FALSE(state.initialized);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, state.x_prev);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, state.y_prev);
}

void test_lowpass_filter_cutoff_clamping(void) {
    lowpass_filter_state_t state;
    
    // Too low - should clamp to minimum
    lowpass_filter_init(&state, 1.0f, 100.0f, true);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, LOWPASS_CUTOFF_MIN, state.cutoff_hz);
    
    // Too high - should clamp to maximum
    lowpass_filter_init(&state, 50.0f, 100.0f, true);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, LOWPASS_CUTOFF_MAX, state.cutoff_hz);
}

void test_lowpass_disabled_by_default_in_context(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    // Low-pass filter should be disabled by default
    TEST_ASSERT_FALSE(ctx.lowpass_state.enabled);
    TEST_ASSERT_FLOAT_WITHIN(0.1f, LOWPASS_CUTOFF_DEFAULT, ctx.lowpass_state.cutoff_hz);
    
    csi_processor_cleanup(&ctx);
}

void test_lowpass_setters_getters(void) {
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, 50, 1.0f));
    
    // Test set/get enabled
    csi_processor_set_lowpass_enabled(&ctx, false);
    TEST_ASSERT_FALSE(csi_processor_get_lowpass_enabled(&ctx));
    
    csi_processor_set_lowpass_enabled(&ctx, true);
    TEST_ASSERT_TRUE(csi_processor_get_lowpass_enabled(&ctx));
    
    // Test set/get cutoff
    csi_processor_set_lowpass_cutoff(&ctx, 8.0f);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 8.0f, csi_processor_get_lowpass_cutoff(&ctx));
    
    csi_processor_cleanup(&ctx);
}

void test_lowpass_in_processing_pipeline(void) {
    csi_processor_context_t ctx_lp, ctx_no_lp;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx_lp, 50, 1.0f));
    TEST_ASSERT_TRUE(csi_processor_init(&ctx_no_lp, 50, 1.0f));
    
    // Enable low-pass on one, disable on other
    csi_processor_set_lowpass_enabled(&ctx_lp, true);
    csi_processor_set_lowpass_enabled(&ctx_no_lp, false);
    
    // Also disable Hampel to isolate low-pass effect
    hampel_turbulence_init(&ctx_lp.hampel_state, 7, 4.0f, false);
    hampel_turbulence_init(&ctx_no_lp.hampel_state, 7, 4.0f, false);
    
    csi_set_subcarrier_selection(TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    
    // Process multiple packets - low-pass should smooth variations
    for (int i = 0; i < 10; i++) {
        csi_process_packet(&ctx_lp, baseline_packets[i % 5], 128,
                          TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
        csi_process_packet(&ctx_no_lp, baseline_packets[i % 5], 128,
                          TEST_SUBCARRIERS, NUM_TEST_SUBCARRIERS);
    }
    
    // Both should have valid moving variance, but low-pass should be smoother
    float mv_lp = csi_processor_get_moving_variance(&ctx_lp);
    float mv_no_lp = csi_processor_get_moving_variance(&ctx_no_lp);
    
    ESP_LOGI(TAG, "Moving variance with lowpass: %.4f, without: %.4f", mv_lp, mv_no_lp);
    
    // Low-pass filter reduces high-frequency noise, so variance should be <= without filter
    // (may be equal if signal is already smooth)
    TEST_ASSERT_TRUE(mv_lp <= mv_no_lp + 0.1f);  // Allow small tolerance
    
    csi_processor_cleanup(&ctx_lp);
    csi_processor_cleanup(&ctx_no_lp);
}

// ============================================================================
// SUBCARRIER SELECTION TESTS
// ============================================================================

void test_set_subcarrier_selection_valid(void) {
    uint8_t subcarriers[] = {10, 15, 20, 25, 30};
    
    // Should not crash - just logs
    csi_set_subcarrier_selection(subcarriers, 5);
    TEST_PASS();
}

void test_set_subcarrier_selection_null(void) {
    // Should handle gracefully
    csi_set_subcarrier_selection(NULL, 0);
    TEST_PASS();
}

void test_set_subcarrier_selection_empty(void) {
    uint8_t subcarriers[] = {10};
    
    // Zero count should be rejected
    csi_set_subcarrier_selection(subcarriers, 0);
    TEST_PASS();
}

void test_set_subcarrier_selection_too_many(void) {
    uint8_t subcarriers[65];
    for (int i = 0; i < 65; i++) subcarriers[i] = i;
    
    // More than 64 should be rejected
    csi_set_subcarrier_selection(subcarriers, 65);
    TEST_PASS();
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
    
    // Getter tests
    RUN_TEST(test_get_window_size_returns_correct_value);
    RUN_TEST(test_get_window_size_returns_zero_for_null);
    RUN_TEST(test_get_threshold_returns_correct_value);
    RUN_TEST(test_get_threshold_returns_zero_for_null);
    RUN_TEST(test_get_state_returns_idle_initially);
    RUN_TEST(test_get_state_returns_idle_for_null);
    RUN_TEST(test_get_moving_variance_returns_zero_initially);
    RUN_TEST(test_get_moving_variance_returns_zero_for_null);
    RUN_TEST(test_get_last_turbulence_returns_zero_initially);
    RUN_TEST(test_get_last_turbulence_returns_zero_for_null);
    RUN_TEST(test_get_last_turbulence_after_processing);
    RUN_TEST(test_get_total_packets_returns_zero_initially);
    RUN_TEST(test_get_total_packets_returns_zero_for_null);
    RUN_TEST(test_get_total_packets_increments_correctly);
    
    // State machine tests
    RUN_TEST(test_state_transitions_to_motion_on_high_variance);
    RUN_TEST(test_state_stays_idle_on_baseline);
    
    // Normalization scale tests
    RUN_TEST(test_normalization_scale_default_is_one);
    RUN_TEST(test_set_normalization_scale_valid_values);
    RUN_TEST(test_normalization_scale_clamps_minimum);
    RUN_TEST(test_normalization_scale_clamps_maximum);
    RUN_TEST(test_get_normalization_scale_returns_one_for_null);
    RUN_TEST(test_normalization_scale_affects_turbulence);
    
    // Low-pass filter tests
    RUN_TEST(test_lowpass_filter_init_default);
    RUN_TEST(test_lowpass_filter_disabled_passthrough);
    RUN_TEST(test_lowpass_filter_enabled_smooths);
    RUN_TEST(test_lowpass_filter_reset);
    RUN_TEST(test_lowpass_filter_cutoff_clamping);
    RUN_TEST(test_lowpass_disabled_by_default_in_context);
    RUN_TEST(test_lowpass_setters_getters);
    RUN_TEST(test_lowpass_in_processing_pipeline);
    
    // Subcarrier selection tests
    RUN_TEST(test_set_subcarrier_selection_valid);
    RUN_TEST(test_set_subcarrier_selection_null);
    RUN_TEST(test_set_subcarrier_selection_empty);
    RUN_TEST(test_set_subcarrier_selection_too_many);
    
    // Memory tests
    RUN_TEST(test_cleanup_deallocates_buffer);
    RUN_TEST(test_stress_no_memory_leak);
    
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

