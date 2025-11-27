/*
 * ESPectre - Unit Test Application Main
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "unity.h"
#include "unity_test_runner.h"
#include <stdio.h>

// test_performance_suite.c
extern test_desc_t test_desc_performance_suite_comprehensive;
// test_features.c
extern test_desc_t test_desc_features_differ_between_baseline_and_movement;
// test_pca_subcarrier.c
extern test_desc_t test_desc_pca_subcarrier_analysis_on_real_data;
// test_segmentation.c
extern test_desc_t test_desc_segmentation_init;
extern test_desc_t test_desc_segmentation_parameters;
extern test_desc_t test_desc_segmentation_reset;
extern test_desc_t test_desc_segmentation_handles_invalid_input;
extern test_desc_t test_desc_segmentation_stress_test;
// test_filters.c
extern test_desc_t test_desc_butterworth_filter_initialization;
extern test_desc_t test_desc_hampel_filter_removes_outliers;
extern test_desc_t test_desc_filter_buffer_operations;
extern test_desc_t test_desc_filter_pipeline_with_wavelet_integration;
extern test_desc_t test_desc_filter_pipeline_with_wavelet_disabled;
// test_wavelet.c
extern test_desc_t test_desc_wavelet_denoising_reduces_noise;
extern test_desc_t test_desc_wavelet_streaming_mode;
extern test_desc_t test_desc_wavelet_allocation_failure_protection;

void app_main(void) {
  printf("\n");
  printf("╔═════════════════════════════════════════════════════════╗\n");
  printf("║                🛜  E S P e c t r e 👻                   ║\n");
  printf("║                      Test Suite                         ║\n");
  printf("╚═════════════════════════════════════════════════════════╝\n");
  printf("\n");


  // ========================================================================
  // CSI PROCESSOR UNIT TESTS
  // ========================================================================

  unity_testcase_register(&test_desc_segmentation_init);
  unity_testcase_register(&test_desc_segmentation_parameters);
  unity_testcase_register(&test_desc_segmentation_reset);
  unity_testcase_register(&test_desc_segmentation_handles_invalid_input);
  unity_testcase_register(&test_desc_segmentation_stress_test);

  // ========================================================================
  // FEATURE TESTS
  // ========================================================================
  
  //unity_testcase_register(&test_desc_features_differ_between_baseline_and_movement);
  //unity_testcase_register(&test_desc_pca_subcarrier_analysis_on_real_data);

  // ========================================================================
  // FILTER TESTS
  // ========================================================================

  unity_testcase_register(&test_desc_butterworth_filter_initialization);
  unity_testcase_register(&test_desc_hampel_filter_removes_outliers);
  unity_testcase_register(&test_desc_filter_buffer_operations);
  unity_testcase_register(&test_desc_filter_pipeline_with_wavelet_integration);
  unity_testcase_register(&test_desc_filter_pipeline_with_wavelet_disabled);

  // ========================================================================
  // WAVELET TESTS
  // ========================================================================

  unity_testcase_register(&test_desc_wavelet_denoising_reduces_noise);
  unity_testcase_register(&test_desc_wavelet_streaming_mode);
  unity_testcase_register(&test_desc_wavelet_allocation_failure_protection);

  // ========================================================================
  // PERFORMANCE TESTS
  // ========================================================================

  unity_testcase_register(&test_desc_performance_suite_comprehensive);

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
  while (1) {
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
