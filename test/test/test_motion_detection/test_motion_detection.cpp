/*
 * ESPectre - Motion Detection Integration Tests
 * 
 * Integration tests for MVS and ML motion detection algorithms.
 * Tests motion detection performance with real CSI data.
 * 
 * Test Categories:
 *   1. test_mvs_optimal_subcarriers - MVS with optimal (offline-tuned) subcarriers (best case)
 *   2. test_mvs_nbvi_calibration - MVS with NBVI auto-calibration (production case)
 *   3. test_ml_detection - ML neural network detection
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>

// Include headers from lib/espectre
#include "utils.h"
#include "filters.h"
#include "mvs_detector.h"
#include "ml_detector.h"
#include "csi_manager.h"
#include "nbvi_calibrator.h"
#include "threshold.h"
#include "esphome/core/log.h"
#include "esp_system.h"

using namespace esphome::espectre;

// Mock WiFi CSI for tests
class WiFiCSIMock : public IWiFiCSI {
 public:
  esp_err_t set_csi_config(const wifi_csi_config_t* config) override { return ESP_OK; }
  esp_err_t set_csi_rx_cb(wifi_csi_cb_t cb, void* ctx) override { return ESP_OK; }
  esp_err_t set_csi(bool enable) override { return ESP_OK; }
};
static WiFiCSIMock g_wifi_mock;

// Include CSI data loader (loads from NPZ files)
#include "csi_test_data.h"

// Compatibility macros for existing test code
#define baseline_packets csi_test_data::baseline_packets()
#define movement_packets csi_test_data::movement_packets()
#define num_baseline csi_test_data::num_baseline()
#define num_movement csi_test_data::num_movement()

static const char *TAG = "test_motion_detection";

// ============================================================================
// Performance Results Storage (for summary table)
// ============================================================================

struct PerformanceResult {
    float recall;
    float fp_rate;
    float precision;
    float f1;
    bool valid;
};

struct ChipResults {
    const char* chip_name;
    PerformanceResult mvs_optimal;
    PerformanceResult mvs_nbvi;
    PerformanceResult ml;
};

static ChipResults g_results[4];  // C3, C6, ESP32, S3
static int g_results_count = 0;

static void record_result(const char* algorithm, float recall, float fp_rate, float precision, float f1) {
    if (g_results_count == 0 || strcmp(g_results[g_results_count - 1].chip_name, 
            csi_test_data::chip_name(csi_test_data::current_chip())) != 0) {
        // New chip
        g_results[g_results_count].chip_name = csi_test_data::chip_name(csi_test_data::current_chip());
        g_results[g_results_count].mvs_optimal = {0, 0, 0, 0, false};
        g_results[g_results_count].mvs_nbvi = {0, 0, 0, 0, false};
        g_results[g_results_count].ml = {0, 0, 0, 0, false};
        g_results_count++;
    }
    
    ChipResults& current = g_results[g_results_count - 1];
    if (strcmp(algorithm, "mvs_optimal") == 0) {
        current.mvs_optimal = {recall, fp_rate, precision, f1, true};
    } else if (strcmp(algorithm, "mvs_nbvi") == 0) {
        current.mvs_nbvi = {recall, fp_rate, precision, f1, true};
    } else if (strcmp(algorithm, "ml") == 0) {
        current.ml = {recall, fp_rate, precision, f1, true};
    }
}

static void print_summary_table() {
    printf("\n");
    printf("================================================================================\n");
    printf("                      PERFORMANCE SUMMARY TABLE (C++)\n");
    printf("================================================================================\n");
    printf("\n");
    printf("| Chip   | MVS Optimal             | MVS + NBVI              | ML                      |\n");
    printf("|--------|-------------------------|-------------------------|-------------------------|\n");
    
    for (int i = 0; i < g_results_count; i++) {
        const ChipResults& r = g_results[i];
        
        char mvs_opt_str[32] = "N/A";
        char mvs_nbvi_str[32] = "N/A";
        char ml_str[32] = "N/A";
        
        if (r.mvs_optimal.valid) {
            snprintf(mvs_opt_str, sizeof(mvs_opt_str), "%.1f%% R, %.1f%% FP", 
                     r.mvs_optimal.recall, r.mvs_optimal.fp_rate);
        }
        if (r.mvs_nbvi.valid) {
            snprintf(mvs_nbvi_str, sizeof(mvs_nbvi_str), "%.1f%% R, %.1f%% FP",
                     r.mvs_nbvi.recall, r.mvs_nbvi.fp_rate);
        }
        if (r.ml.valid) {
            snprintf(ml_str, sizeof(ml_str), "%.1f%% R, %.1f%% FP",
                     r.ml.recall, r.ml.fp_rate);
        }
        
        printf("| %-6s | %-23s | %-23s | %-23s |\n", 
               r.chip_name, mvs_opt_str, mvs_nbvi_str, ml_str);
    }
    
    printf("\n");
    printf("Legend: R = Recall, FP = False Positive Rate\n");
    printf("Targets: MVS Recall >97%%, ML Recall >93%%, FP Rate <10%%\n");
    printf("================================================================================\n");
    
    // Detailed table for PERFORMANCE.md
    printf("\n");
    printf("                         DETAILED METRICS (for PERFORMANCE.md)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("| Chip   | Algorithm   | Recall  | Precision | FP Rate | F1-Score |\n");
    printf("|--------|-------------|---------|-----------|---------|----------|\n");
    
    for (int i = 0; i < g_results_count; i++) {
        const ChipResults& r = g_results[i];
        
        if (r.mvs_optimal.valid) {
            printf("| %-6s | MVS Optimal | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   r.chip_name, r.mvs_optimal.recall, r.mvs_optimal.precision,
                   r.mvs_optimal.fp_rate, r.mvs_optimal.f1);
        }
        if (r.mvs_nbvi.valid) {
            printf("| %-6s | MVS + NBVI  | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   r.chip_name, r.mvs_nbvi.recall, r.mvs_nbvi.precision,
                   r.mvs_nbvi.fp_rate, r.mvs_nbvi.f1);
        }
        if (r.ml.valid) {
            printf("| %-6s | ML          | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   r.chip_name, r.ml.recall, r.ml.precision,
                   r.ml.fp_rate, r.ml.f1);
        }
    }
    
    printf("--------------------------------------------------------------------------------\n");
}

// ============================================================================
// Optimal Subcarriers (found via offline grid search analysis)
// ============================================================================
// These represent the "best case" for each chip, found by analyzing datasets
static const uint8_t SUBCARRIERS_ESP32_64SC[] = {12, 13, 14, 17, 44, 45, 46, 48, 49, 50, 51, 52};
static const uint8_t SUBCARRIERS_C3_64SC[] = {18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29};
static const uint8_t SUBCARRIERS_C6_64SC[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
static const uint8_t SUBCARRIERS_S3_64SC[] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
static const uint8_t NUM_SELECTED_SUBCARRIERS = 12;

// ============================================================================
// Chip-Specific Configuration
// ============================================================================

inline bool is_s3_chip() {
    return csi_test_data::current_chip() == csi_test_data::ChipType::S3;
}

inline bool is_c3_chip() {
    return csi_test_data::current_chip() == csi_test_data::ChipType::C3;
}

inline bool is_esp32_chip() {
    return csi_test_data::current_chip() == csi_test_data::ChipType::ESP32;
}

// Chips without gain lock need CV normalization (std/mean)
// ESP32: hardware doesn't support gain lock
// C3: current dataset was collected without gain lock (signal too strong)
inline bool needs_cv_normalization() {
    return is_esp32_chip() || is_c3_chip();
}

inline const uint8_t* get_optimal_subcarriers() {
    switch (csi_test_data::current_chip()) {
        case csi_test_data::ChipType::C3: return SUBCARRIERS_C3_64SC;
        case csi_test_data::ChipType::ESP32: return SUBCARRIERS_ESP32_64SC;
        case csi_test_data::ChipType::S3: return SUBCARRIERS_S3_64SC;
        default: return SUBCARRIERS_C6_64SC;
    }
}

// MVS targets
// All chips achieve >96% recall
inline float get_fp_rate_target() { return 10.0f; }
inline float get_recall_target() { return 96.0f; }
inline float get_nbvi_recall_target() { return 96.0f; }

// Unified parameters for all chips (use production defaults)
inline uint16_t get_window_size() { return DETECTOR_DEFAULT_WINDOW_SIZE; }
inline bool get_enable_hampel() { return false; }

// ML targets
// All chips achieve >94% recall and <10% FP rate
inline float get_ml_fp_rate_target() { return 10.0f; }
inline float get_ml_recall_target() { return 93.0f; }

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// Test 1: MVS with Optimal Subcarriers (Best Case Reference)
// ============================================================================
// Uses offline-tuned subcarriers to establish best possible performance.
// This serves as a reference to measure NBVI degradation.

void test_mvs_optimal_subcarriers(void) {
    float fp_target = get_fp_rate_target();
    float recall_target = get_recall_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    bool cv_norm = needs_cv_normalization();
    const int pkt_size = csi_test_data::packet_size();
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: MVS with Optimal Subcarriers (Best Case)\n");
    printf("  Chip: %s, Window: %d, CV Norm: %s\n", 
           csi_test_data::chip_name(csi_test_data::current_chip()), 
           window_size, cv_norm ? "ON" : "OFF");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Get optimal subcarriers for this chip
    const uint8_t* optimal_band = get_optimal_subcarriers();
    printf("Optimal subcarriers: [");
    for (int i = 0; i < NUM_SELECTED_SUBCARRIERS; i++) {
        printf("%d", optimal_band[i]);
        if (i < NUM_SELECTED_SUBCARRIERS - 1) printf(", ");
    }
    printf("]\n\n");
    
    // Create calibration detector to calculate adaptive threshold
    MVSDetector cal_detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    cal_detector.configure_lowpass(false);
    cal_detector.configure_hampel(enable_hampel, 7, 4.0f);
    cal_detector.set_cv_normalization(cv_norm);
    
    std::vector<float> mv_values;
    int calibration_packets = std::min(num_baseline, static_cast<int>(CALIBRATION_DEFAULT_BUFFER_SIZE));
    for (int i = 0; i < calibration_packets; i++) {
        cal_detector.process_packet((const int8_t*)baseline_packets[i], pkt_size,
                          optimal_band, NUM_SELECTED_SUBCARRIERS);
        cal_detector.update_state();
        if (cal_detector.is_ready()) {
            mv_values.push_back(cal_detector.get_motion_metric());
        }
    }
    
    float adaptive_threshold = calculate_adaptive_threshold(mv_values, 95);
    printf("Adaptive threshold: %.6f (P95, from %zu MV values)\n\n", 
           adaptive_threshold, mv_values.size());
    
    // Create detector for evaluation
    MVSDetector detector(window_size, adaptive_threshold);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    detector.set_cv_normalization(cv_norm);
    
    // Process baseline
    int baseline_motion = 0;
    for (int p = 0; p < num_baseline; p++) {
        detector.process_packet((const int8_t*)baseline_packets[p], pkt_size, 
                          optimal_band, NUM_SELECTED_SUBCARRIERS);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            baseline_motion++;
        }
    }
    
    // Process movement
    int movement_motion = 0;
    for (int p = 0; p < num_movement; p++) {
        detector.process_packet((const int8_t*)movement_packets[p], pkt_size,
                          optimal_band, NUM_SELECTED_SUBCARRIERS);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            movement_motion++;
        }
    }
    
    // Calculate metrics
    float recall = (float)movement_motion / num_movement * 100.0f;
    float fp_rate = (float)baseline_motion / num_baseline * 100.0f;
    float precision = (movement_motion + baseline_motion > 0) ?
        (float)movement_motion / (movement_motion + baseline_motion) * 100.0f : 0.0f;
    float f1 = (precision + recall > 0) ?
        2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
    
    printf("Results:\n");
    printf("  * Recall:    %.1f%% (target: >%.0f%%)\n", recall, recall_target);
    printf("  * FP Rate:   %.1f%% (target: <%.0f%%)\n", fp_rate, fp_target);
    printf("  * Precision: %.1f%%\n", precision);
    printf("  * F1-Score:  %.1f%%\n\n", f1);
    
    // Record for summary table
    record_result("mvs_optimal", recall, fp_rate, precision, f1);
    
    TEST_ASSERT_TRUE_MESSAGE(recall > recall_target, "Recall too low");
    TEST_ASSERT_TRUE_MESSAGE(fp_rate < fp_target, "FP Rate too high");
}

// ============================================================================
// Test 2: MVS with NBVI Calibration (Production Case)
// ============================================================================
// Uses NBVI auto-calibration as in production.
// For chips requiring CV normalization (C3, ESP32), NBVI is skipped and optimal
// subcarriers are used instead (matches Python test behavior).

void test_mvs_nbvi_calibration(void) {
    float fp_target = get_fp_rate_target();
    float recall_target = get_nbvi_recall_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    bool cv_norm = needs_cv_normalization();
    const int pkt_size = csi_test_data::packet_size();
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: MVS with NBVI Calibration (Production Case)\n");
    printf("  Chip: %s, Window: %d, CV Norm: %s\n", 
           csi_test_data::chip_name(csi_test_data::current_chip()), 
           window_size, cv_norm ? "ON" : "OFF");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    MVSDetector detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    detector.set_cv_normalization(cv_norm);
    
    uint8_t calibrated_band[12] = {0};
    uint8_t calibrated_size = 0;
    float calibrated_threshold = 1.0f;
    
    // For chips with CV normalization, skip NBVI and use optimal subcarriers
    // (NBVI calibrator doesn't support CV normalization in Python, so we match that behavior)
    if (cv_norm) {
        printf("CV normalization enabled - using optimal subcarriers (NBVI skipped)\n");
        const uint8_t* optimal_band = get_optimal_subcarriers();
        memcpy(calibrated_band, optimal_band, NUM_SELECTED_SUBCARRIERS);
        calibrated_size = NUM_SELECTED_SUBCARRIERS;
        
        // Calculate adaptive threshold using calibration detector (same as test_mvs_optimal_subcarriers)
        MVSDetector cal_detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
        cal_detector.configure_lowpass(false);
        cal_detector.configure_hampel(enable_hampel, 7, 4.0f);
        cal_detector.set_cv_normalization(cv_norm);
        
        std::vector<float> mv_values;
        int calibration_packets = std::min(num_baseline, static_cast<int>(CALIBRATION_DEFAULT_BUFFER_SIZE));
        for (int i = 0; i < calibration_packets; i++) {
            cal_detector.process_packet((const int8_t*)baseline_packets[i], pkt_size,
                              optimal_band, NUM_SELECTED_SUBCARRIERS);
            cal_detector.update_state();
            if (cal_detector.is_ready()) {
                mv_values.push_back(cal_detector.get_motion_metric());
            }
        }
        
        calibrated_threshold = calculate_adaptive_threshold(mv_values, 95);
        printf("Adaptive threshold: %.6f (P95, from %zu MV values)\n\n", 
               calibrated_threshold, mv_values.size());
    } else {
        // Use NBVI calibration for chips without CV normalization (C6, S3)
        CSIManager csi_manager;
        csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
        
        NBVICalibrator nbvi;
        nbvi.init(&csi_manager, "/tmp/test_nbvi_buffer.bin");
        nbvi.set_mvs_window_size(window_size);
        nbvi.set_cv_normalization(cv_norm);
        
        uint16_t buffer_size = std::min(static_cast<int>(nbvi.get_buffer_size()), num_baseline);
        nbvi.set_buffer_size(buffer_size);
        
        bool calibration_success = false;
        
        esp_err_t err = nbvi.start_calibration(get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS,
            [&](const uint8_t* band, uint8_t size, const std::vector<float>& mv_values, bool success) {
                if (success && size > 0) {
                    memcpy(calibrated_band, band, size);
                    calibrated_size = size;
                    calibrated_threshold = calculate_adaptive_threshold(mv_values, 95);
                }
                calibration_success = success;
            });
        
        TEST_ASSERT_EQUAL(ESP_OK, err);
        
        printf("NBVI calibrating with %d baseline packets...\n", buffer_size);
        for (int i = 0; i < buffer_size && i < num_baseline; i++) {
            nbvi.add_packet(baseline_packets[i], pkt_size);
        }
        
        TEST_ASSERT_TRUE_MESSAGE(calibration_success, "NBVI calibration failed");
        
        printf("NBVI selected band: [");
        for (int i = 0; i < calibrated_size; i++) {
            printf("%d", calibrated_band[i]);
            if (i < calibrated_size - 1) printf(", ");
        }
        printf("]\n");
        printf("Adaptive threshold: %.6f (P95)\n\n", calibrated_threshold);
    }
    
    // Apply calibration
    detector.set_threshold(calibrated_threshold);
    detector.clear_buffer();
    
    // Process baseline
    int baseline_motion = 0;
    for (int i = 0; i < num_baseline; i++) {
        detector.process_packet((const int8_t*)baseline_packets[i], pkt_size, 
                          calibrated_band, calibrated_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            baseline_motion++;
        }
    }
    
    // Process movement
    int movement_motion = 0;
    for (int i = 0; i < num_movement; i++) {
        detector.process_packet((const int8_t*)movement_packets[i], pkt_size, 
                          calibrated_band, calibrated_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            movement_motion++;
        }
    }
    
    // Calculate metrics
    float recall = (float)movement_motion / num_movement * 100.0f;
    float fp_rate = (float)baseline_motion / num_baseline * 100.0f;
    float precision = (movement_motion + baseline_motion > 0) ?
        (float)movement_motion / (movement_motion + baseline_motion) * 100.0f : 0.0f;
    float f1 = (precision + recall > 0) ?
        2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
    
    printf("Results:\n");
    printf("  * Recall:    %.1f%% (target: >%.0f%%)\n", recall, recall_target);
    printf("  * FP Rate:   %.1f%% (target: <%.0f%%)\n", fp_rate, fp_target);
    printf("  * Precision: %.1f%%\n", precision);
    printf("  * F1-Score:  %.1f%%\n\n", f1);
    
    // Cleanup
    remove("/tmp/test_nbvi_buffer.bin");
    
    // Record for summary table
    record_result("mvs_nbvi", recall, fp_rate, precision, f1);
    
    TEST_ASSERT_TRUE_MESSAGE(recall >= recall_target, "NBVI Recall too low");
    TEST_ASSERT_TRUE_MESSAGE(fp_rate <= fp_target, "NBVI FP Rate too high");
}

// ============================================================================
// Test 3: ML Detection
// ============================================================================
// Tests ML neural network detector with fixed subcarriers.

void test_ml_detection(void) {
    float fp_target = get_ml_fp_rate_target();
    float recall_target = get_ml_recall_target();
    bool cv_norm = needs_cv_normalization();
    const int pkt_size = csi_test_data::packet_size();
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: ML Detection (Neural Network)\n");
    printf("  Chip: %s, CV Norm: %s\n", 
           csi_test_data::chip_name(csi_test_data::current_chip()),
           cv_norm ? "ON" : "OFF");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    MLDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, ML_DEFAULT_THRESHOLD);
    detector.set_cv_normalization(cv_norm);
    
    printf("ML subcarriers: [11, 14, 17, ..., 49] (fixed)\n");
    printf("Threshold: %.1f\n\n", detector.get_threshold());
    
    // Warmup = window_size: detector needs full buffer before producing valid predictions
    const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
    
    // Process baseline (skip first warmup packets - buffer not ready)
    int baseline_motion = 0;
    for (int i = 0; i < num_baseline; i++) {
        detector.process_packet((const int8_t*)baseline_packets[i], pkt_size,
                               ML_SUBCARRIERS, 12);
        detector.update_state();
        // Only count after warmup (when buffer is full)
        if (i >= warmup && detector.get_state() == MotionState::MOTION) {
            baseline_motion++;
        }
    }
    
    // Process movement (skip first warmup packets - transition period)
    int movement_motion = 0;
    int movement_idle = 0;
    
    for (int i = 0; i < num_movement; i++) {
        detector.process_packet((const int8_t*)movement_packets[i], pkt_size,
                               ML_SUBCARRIERS, 12);
        detector.update_state();
        if (i >= warmup) {
            if (detector.get_state() == MotionState::MOTION) {
                movement_motion++;
            } else {
                movement_idle++;
            }
        }
    }
    
    int baseline_eval = num_baseline - warmup;
    int movement_eval = num_movement - warmup;
    float recall = (float)movement_motion / movement_eval * 100.0f;
    float fp_rate = (float)baseline_motion / baseline_eval * 100.0f;
    float precision = (movement_motion + baseline_motion > 0) ?
        (float)movement_motion / (movement_motion + baseline_motion) * 100.0f : 0.0f;
    float f1 = (precision + recall > 0) ?
        2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
    
    printf("Results:\n");
    printf("  * Recall:    %.1f%% (target: >%.0f%%)\n", recall, recall_target);
    printf("  * FP Rate:   %.1f%% (target: <%.0f%%)\n", fp_rate, fp_target);
    printf("  * Precision: %.1f%%\n", precision);
    printf("  * F1-Score:  %.1f%%\n\n", f1);
    
    // Record for summary table
    record_result("ml", recall, fp_rate, precision, f1);
    
    TEST_ASSERT_TRUE_MESSAGE(recall > recall_target, "ML Recall too low");
    TEST_ASSERT_TRUE_MESSAGE(fp_rate < fp_target, "ML FP Rate too high");
}

// ============================================================================
// Test Runner
// ============================================================================

int run_tests_for_chip(csi_test_data::ChipType chip) {
    printf("\n========================================\n");
    printf("Running tests with %s 64 SC dataset (HT20)\n", csi_test_data::chip_name(chip));
    printf("========================================\n");
    
    const char* skip_reason = csi_test_data::chip_skip_reason(chip);
    if (skip_reason != nullptr) {
        printf("SKIPPED: %s\n", skip_reason);
        return 0;
    }
    
    if (!csi_test_data::switch_dataset(chip)) {
        printf("ERROR: Failed to load %s dataset\n", csi_test_data::chip_name(chip));
        return 1;
    }
    
    UNITY_BEGIN();
    RUN_TEST(test_mvs_optimal_subcarriers);   // Best case reference
    RUN_TEST(test_mvs_nbvi_calibration);      // Production case
    RUN_TEST(test_ml_detection);              // ML neural network
    return UNITY_END();
}

int process(void) {
    int failures = 0;
    for (auto chip : csi_test_data::get_available_chips()) {
        failures += run_tests_for_chip(chip);
    }
    
    // Print summary table at the end
    print_summary_table();
    
    return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
