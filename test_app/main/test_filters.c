/*
 * ESPectre - Filter Unit Tests
 * 
 * Tests for Butterworth, Hampel and Savitzky-Golay
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "filters.h"
#include "wavelet.h"
#include <math.h>
#include <string.h>


// Test: Butterworth filter initialization
TEST_CASE_ESP(butterworth_filter_initialization, "[filters]")
{
    butterworth_filter_t filter = {0};
    butterworth_init(&filter);
    
    TEST_ASSERT_TRUE(filter.initialized);
    TEST_ASSERT_FLOAT_WITHIN(0.001f, 1.0f, filter.a[0]);
}

// Test: Hampel filter removes outliers
TEST_CASE_ESP(hampel_filter_removes_outliers, "[filters]")
{
    float window[] = {1.0f, 1.1f, 1.0f, 100.0f, 0.9f};  // 100.0 is outlier
    float result = hampel_filter(window, 5, 100.0f, 2.0f);
    
    // Should replace outlier with median (~1.0)
    TEST_ASSERT_FLOAT_WITHIN(0.5f, 1.0f, result);
}

// Test: Filter buffer operations
TEST_CASE_ESP(filter_buffer_operations, "[filters]")
{
    filter_buffer_t fb;
    filter_buffer_init(&fb);
    
    // Add some samples
    for (int i = 0; i < 3; i++) {
        filter_buffer_add(&fb, (float)i);
    }
    
    TEST_ASSERT_EQUAL(3, fb.count);
    
    // Get window
    float window[5];
    size_t size;
    filter_buffer_get_window(&fb, window, 5, &size);
    
    TEST_ASSERT_EQUAL(3, size);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 0.0f, window[0]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 1.0f, window[1]);
    TEST_ASSERT_FLOAT_WITHIN(0.01f, 2.0f, window[2]);
}


// Test: Complete filter pipeline with wavelet
TEST_CASE_ESP(filter_pipeline_with_wavelet_integration, "[filters]")
{
    butterworth_filter_t butterworth = {0};
    wavelet_state_t wavelet = {0};
    wavelet_init(&wavelet, 3, 1.0f, WAVELET_THRESH_SOFT);
    filter_buffer_t buffer = {0};
    filter_buffer_init(&buffer);
    
    filter_config_t config = {
        .butterworth_enabled = true,
        .wavelet_enabled = true,
        .wavelet_level = 3,
        .wavelet_threshold = 1.0f,
        .hampel_enabled = true,
        .hampel_threshold = 2.0f,
        .savgol_enabled = true
    };
    
    // Apply pipeline with noisy input
    float noisy_input = 10.0f;
    float filtered = apply_filter_pipeline(noisy_input, &config, &butterworth, 
                                          &wavelet, &buffer);
    
    // Output should be valid
    TEST_ASSERT_FALSE(isnan(filtered));
    TEST_ASSERT_FALSE(isinf(filtered));
}

// Test: Wavelet can be disabled in pipeline
TEST_CASE_ESP(filter_pipeline_with_wavelet_disabled, "[filters]")
{
    butterworth_filter_t butterworth = {0};
    wavelet_state_t wavelet = {0};
    wavelet_init(&wavelet, 3, 1.0f, WAVELET_THRESH_SOFT);
    filter_buffer_t buffer = {0};
    filter_buffer_init(&buffer);
    
    filter_config_t config = {
        .butterworth_enabled = true,
        .wavelet_enabled = false,  // DISABLED
        .wavelet_level = 3,
        .wavelet_threshold = 1.0f,
        .hampel_enabled = false,
        .savgol_enabled = false
    };
    
    // Apply pipeline
    float input = 5.0f;
    float filtered = apply_filter_pipeline(input, &config, &butterworth, 
                                          &wavelet, &buffer);
    
    // Should work fine with wavelet disabled
    TEST_ASSERT_FALSE(isnan(filtered));
    TEST_ASSERT_FALSE(isinf(filtered));
}
