/*
 * ESPectre - Feature Extraction Unit Tests
 * 
 * Tests for CSI feature extraction and processing
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "csi_processor.h"
#include "mock_csi_data.h"
#include "esp_wifi_types.h"
#include <math.h>
#include <string.h>

// Test: CSI variance calculation
TEST_CASE_ESP("CSI variance calculation", "[features]")
{
    int8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float variance = csi_calculate_variance(data, 8);
    
    // Variance should be > 0 for non-constant data
    TEST_ASSERT_GREATER_THAN(0.0f, variance);
}

// Test: Skewness calculation
TEST_CASE_ESP("Skewness calculation", "[features]")
{
    int8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float skewness = csi_calculate_skewness(data, 10);
    
    // Symmetric data should have skewness close to 0
    TEST_ASSERT_FLOAT_WITHIN(0.5f, 0.0f, skewness);
}

// Test: Kurtosis calculation
TEST_CASE_ESP("Kurtosis calculation", "[features]")
{
    int8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float kurtosis = csi_calculate_kurtosis(data, 10);
    
    // Kurtosis should be calculated (value depends on distribution)
    TEST_ASSERT_NOT_EQUAL(0.0f, kurtosis);
}

// Test: Entropy calculation
TEST_CASE_ESP("Entropy calculation", "[features]")
{
    int8_t data[] = {1, 1, 1, 1, 1};  // Low entropy (constant)
    float entropy = csi_calculate_entropy(data, 5);
    
    // Constant data should have low entropy
    TEST_ASSERT_LESS_THAN(1.0f, entropy);
}

// Test: IQR calculation
TEST_CASE_ESP("IQR calculation", "[features]")
{
    int8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    float iqr = csi_calculate_iqr(data, 10);
    
    // IQR should be > 0 for varied data
    TEST_ASSERT_GREATER_THAN(0.0f, iqr);
}

// Test: Spatial variance
TEST_CASE_ESP("Spatial variance calculation", "[features]")
{
    // Use data with varying differences between adjacent elements
    int8_t data[] = {1, 3, 2, 6, 4, 8, 5, 10};
    float spatial_var = csi_calculate_spatial_variance(data, 8);
    
    TEST_ASSERT_GREATER_THAN(0.0f, spatial_var);
}

// Test: Spatial correlation
TEST_CASE_ESP("Spatial correlation calculation", "[features]")
{
    int8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float correlation = csi_calculate_spatial_correlation(data, 8);
    
    // Correlation should be between -1 and 1
    TEST_ASSERT_GREATER_OR_EQUAL(-1.0f, correlation);
    TEST_ASSERT_LESS_OR_EQUAL(1.0f, correlation);
}

// Test: Spatial gradient
TEST_CASE_ESP("Spatial gradient calculation", "[features]")
{
    int8_t data[] = {1, 2, 3, 4, 5, 6, 7, 8};
    float gradient = csi_calculate_spatial_gradient(data, 8);
    
    TEST_ASSERT_GREATER_THAN(0.0f, gradient);
}

// Test: Mock CSI data generation
TEST_CASE_ESP("Mock CSI data generation", "[features]")
{
    // Allocate buffer for CSI data
    int8_t buffer[384];  // Max CSI length
    wifi_csi_info_t csi_info = {
        .buf = buffer,
        .len = 0
    };
    
    generate_mock_csi_data(&csi_info, MOCK_CSI_STATIC);
    
    // Verify basic properties
    TEST_ASSERT_EQUAL(128, csi_info.len);
    
    // Verify that at least some data was generated (not all zeros)
    int non_zero_count = 0;
    for (int i = 0; i < csi_info.len; i++) {
        if (csi_info.buf[i] != 0) {
            non_zero_count++;
        }
    }
    TEST_ASSERT_GREATER_THAN(0, non_zero_count);
}
