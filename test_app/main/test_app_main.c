/*
 * ESPectre - Unit Test Application Main
 * 
 * Test suite focused on performance.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <stdio.h>
#include "unity.h"
#include "unity_test_runner.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

// ============================================================================
// PERFORMANCE TESTS (CORE)
// ============================================================================
// test_performance_suite.c
extern test_desc_t test_desc_performance_suite_comprehensive;
// test_threshold_optimization.c
extern test_desc_t test_desc_threshold_optimization_for_recall;
// test_temporal_robustness.c
extern test_desc_t test_desc_temporal_robustness_scenarios;
// test_home_assistant_integration.c
extern test_desc_t test_desc_home_assistant_integration_e2e;

// ============================================================================
// CALIBRATION & OPTIMIZATION TESTS
// ============================================================================
// test_calibration.c
extern test_desc_t test_desc_features_differ_between_baseline_and_movement;
// test_real_calibration.c
extern test_desc_t test_desc_calibration_with_real_csi_data;
// test_pca_subcarrier.c
extern test_desc_t test_desc_pca_subcarrier_analysis_on_real_data;

// ============================================================================
// SEGMENTATION TESTS
// ============================================================================
// test_segmentation.c
extern test_desc_t test_desc_segmentation_init;
extern test_desc_t test_desc_segmentation_calibration;
extern test_desc_t test_desc_segmentation_movement_detection;
extern test_desc_t test_desc_segmentation_no_false_positives;
extern test_desc_t test_desc_spatial_turbulence_calculation;
extern test_desc_t test_desc_segmentation_reset;

// ============================================================================
// COMPONENT TESTS (Minimal - for debugging only)
// ============================================================================
// test_filters.c
extern test_desc_t test_desc_filter_pipeline_with_wavelet_integration;
extern test_desc_t test_desc_filter_pipeline_with_wavelet_disabled;
// test_wavelet.c
extern test_desc_t test_desc_wavelet_denoising_reduces_noise;
extern test_desc_t test_desc_wavelet_streaming_mode;

void app_main(void)
{
    printf("\n");
    printf("╔═════════════════════════════════════════════════════════╗\n");
    printf("║                🛜  E S P e c t r e 👻                   ║\n");
    printf("║                      Test Suite                         ║\n");
    printf("╚═════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Test Suite Reorganized for Home Assistant Integration\n");
    printf("Focus: Security/Presence Detection (90%% Recall, 1-5 FP/hour)\n");
    printf("\n");
    
    // ========================================================================
    // PERFORMANCE TESTS (CORE) - Run these for Home Assistant validation
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  REGISTERING PERFORMANCE TESTS (CORE)\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    unity_testcase_register(&test_desc_performance_suite_comprehensive);
    unity_testcase_register(&test_desc_threshold_optimization_for_recall);
    unity_testcase_register(&test_desc_temporal_robustness_scenarios);
    unity_testcase_register(&test_desc_home_assistant_integration_e2e);
    
    printf("✅ Registered 4 performance tests\n\n");
    
    // ========================================================================
    // CALIBRATION & OPTIMIZATION TESTS
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  REGISTERING CALIBRATION & OPTIMIZATION TESTS\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    unity_testcase_register(&test_desc_features_differ_between_baseline_and_movement);
    unity_testcase_register(&test_desc_calibration_with_real_csi_data);
    unity_testcase_register(&test_desc_pca_subcarrier_analysis_on_real_data);
    
    printf("✅ Registered 3 calibration tests\n\n");
    
    // ========================================================================
    // SEGMENTATION TESTS
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  REGISTERING SEGMENTATION TESTS\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    unity_testcase_register(&test_desc_segmentation_init);
    unity_testcase_register(&test_desc_spatial_turbulence_calculation);
    unity_testcase_register(&test_desc_segmentation_calibration);
    unity_testcase_register(&test_desc_segmentation_no_false_positives);
    unity_testcase_register(&test_desc_segmentation_movement_detection);
    unity_testcase_register(&test_desc_segmentation_reset);
    
    printf("✅ Registered 6 segmentation tests\n\n");
    
    // ========================================================================
    // COMPONENT TESTS
    // ========================================================================
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  REGISTERING COMPONENT TESTS\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    unity_testcase_register(&test_desc_filter_pipeline_with_wavelet_integration);
    unity_testcase_register(&test_desc_filter_pipeline_with_wavelet_disabled);
    unity_testcase_register(&test_desc_wavelet_denoising_reduces_noise);
    unity_testcase_register(&test_desc_wavelet_streaming_mode);
    
    printf("✅ Registered 4 component tests\n\n");
    
    printf("═══════════════════════════════════════════════════════════\n");
    printf("  TOTAL: 17 tests registered (reduced from 38)\n");
    printf("  Removed: 21 redundant/mock tests\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    
    // Run all tests
    printf("Running all tests automatically...\n\n");
    unity_run_all_tests();
    
    // Check results
    if (Unity.TestFailures == 0) {
        printf("\n✅ All tests passed!\n");
    } else {
        printf("\n❌ Tests failed: %d failure(s)\n", Unity.TestFailures);
    }
    printf("Press Ctrl+] to exit monitor\n");
    
    // Prevent reboot loop
    while(1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
