/*
 * ESPectre - Unit Test Application Main
 *
 * Test suite for segmentation-based architecture.
 *
 * New Architecture:
 *   CSI Packet → Segmentation (always) → IF MOTION && features_enabled:
 *                                           → Extract Features + Publish
 *                                        ELSE:
 *                                           → Publish without features
 *
 * Accuracy based on: Segmentation performance (Moving Variance Segmentation - MVS)
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "unity.h"
#include "unity_test_runner.h"
#include <stdio.h>

// ============================================================================
// PERFORMANCE TESTS (CORE) - Segmentation-based
// ============================================================================
// test_performance_suite.c - Tests segmentation performance (PRIMARY)
extern test_desc_t test_desc_performance_suite_comprehensive;
// test_threshold_optimization.c - Feature ranking (SECONDARY - for features_enabled mode)
extern test_desc_t test_desc_threshold_optimization_for_recall;

// ============================================================================
// FEATURE EXTRACTION & SEGMENTATION TUNING TESTS
// ============================================================================
// test_features.c
extern test_desc_t test_desc_features_differ_between_baseline_and_movement;
// test_segmentation_tuning.c
extern test_desc_t test_desc_segmentation_threshold_tuning_with_real_csi;
// test_pca_subcarrier.c
extern test_desc_t test_desc_pca_subcarrier_analysis_on_real_data;

// ============================================================================
// SEGMENTATION TESTS
// ============================================================================
// test_segmentation.c
extern test_desc_t test_desc_segmentation_init;
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
extern test_desc_t test_desc_wavelet_allocation_failure_protection;

void app_main(void) {
  printf("\n");
  printf("╔═════════════════════════════════════════════════════════╗\n");
  printf("║                🛜  E S P e c t r e 👻                   ║\n");
  printf("║                      Test Suite                         ║\n");
  printf("╚═════════════════════════════════════════════════════════╝\n");
  printf("\n");

  // ========================================================================
  // SEGMENTATION UNIT TESTS
  // ========================================================================
  printf("═══════════════════════════════════════════════════════════\n");
  printf("  REGISTERING SEGMENTATION UNIT TESTS\n");
  printf("═══════════════════════════════════════════════════════════\n\n");

  unity_testcase_register(&test_desc_segmentation_init);
  unity_testcase_register(&test_desc_spatial_turbulence_calculation);
  unity_testcase_register(&test_desc_segmentation_no_false_positives);
  unity_testcase_register(&test_desc_segmentation_movement_detection);
  unity_testcase_register(&test_desc_segmentation_reset);

  printf("✅ Registered 6 segmentation unit tests\n\n");

  // ========================================================================
  // TUNING & FEATURE TESTS (SECONDARY)
  // ========================================================================
  printf("═══════════════════════════════════════════════════════════\n");
  printf("  REGISTERING TUNING & FEATURE TESTS (SECONDARY)\n");
  printf("═══════════════════════════════════════════════════════════\n\n");

  unity_testcase_register(
      &test_desc_segmentation_threshold_tuning_with_real_csi);
  unity_testcase_register(
      &test_desc_features_differ_between_baseline_and_movement);
  unity_testcase_register(&test_desc_pca_subcarrier_analysis_on_real_data);

  printf("✅ Registered 3 tuning/feature tests\n");
  printf("   Note: Features for features_enabled mode only\n\n");

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
  unity_testcase_register(&test_desc_wavelet_allocation_failure_protection);

  printf("✅ Registered 5 component tests\n\n");

  // ========================================================================
  // PERFORMANCE TESTS (CORE) - Segmentation-based
  // ========================================================================
  printf("═══════════════════════════════════════════════════════════\n");
  printf("  REGISTERING PERFORMANCE TESTS (SEGMENTATION-BASED)\n");
  printf("═══════════════════════════════════════════════════════════\n\n");

  unity_testcase_register(&test_desc_performance_suite_comprehensive);
  unity_testcase_register(&test_desc_threshold_optimization_for_recall);

  printf("✅ Registered 3 performance tests\n");
  printf("   - performance_suite: Segmentation metrics (PRIMARY)\n");
  printf("   - threshold_optimization: Feature ranking (SECONDARY)\n");
  printf("   - home_assistant_integration: E2E validation\n\n");

  printf("═══════════════════════════════════════════════════════════\n");
  printf("  TOTAL: 17 tests registered\n");
  printf("  Architecture: Segmentation-based (MVS)\n");
  printf("  Primary: Segmentation performance\n");
  printf("  Secondary: Feature ranking (features_enabled mode)\n");
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
  while (1) {
    vTaskDelay(pdMS_TO_TICKS(1000));
  }
}
