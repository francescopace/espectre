/*
 * ESPectre - Motion Detection Integration Tests
 * 
 * Integration tests for MVS and ML motion detection algorithms.
 * Tests motion detection performance with real CSI data.
 * 
 * Test Categories:
 *   1. test_mvs_default_subcarriers - MVS with default (offline-tuned) subcarriers (production baseline)
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
#include "espectre.h"
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
    PerformanceResult mvs_default;
    PerformanceResult mvs_nbvi;
    PerformanceResult ml;
};

static ChipResults g_results[5];  // C3, C5, C6, ESP32, S3
static int g_results_count = 0;

// Forward declarations for target getters used in summary output.
inline float get_default_fp_rate_target();
inline float get_default_recall_target();
inline float get_fp_rate_target();
inline float get_nbvi_recall_target();
inline float get_ml_fp_rate_target();
inline float get_ml_recall_target();

static void record_result(const char* algorithm, float recall, float fp_rate, float precision, float f1) {
    if (g_results_count == 0 || strcmp(g_results[g_results_count - 1].chip_name, 
            csi_test_data::chip_name(csi_test_data::current_chip())) != 0) {
        // New chip
        g_results[g_results_count].chip_name = csi_test_data::chip_name(csi_test_data::current_chip());
        g_results[g_results_count].mvs_default = {0, 0, 0, 0, false};
        g_results[g_results_count].mvs_nbvi = {0, 0, 0, 0, false};
        g_results[g_results_count].ml = {0, 0, 0, 0, false};
        g_results_count++;
    }
    
    ChipResults& current = g_results[g_results_count - 1];
    if (strcmp(algorithm, "mvs_default") == 0) {
        current.mvs_default = {recall, fp_rate, precision, f1, true};
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
    printf("| Chip   | MVS Default             | MVS + NBVI              | ML                      |\n");
    printf("|--------|-------------------------|-------------------------|-------------------------|\n");
    
    for (int i = 0; i < g_results_count; i++) {
        const ChipResults& r = g_results[i];
        
        char mvs_default_str[32] = "N/A";
        char mvs_nbvi_str[32] = "N/A";
        char ml_str[32] = "N/A";
        
        if (r.mvs_default.valid) {
            snprintf(mvs_default_str, sizeof(mvs_default_str), "%.1f%% R, %.1f%% FP",
                     r.mvs_default.recall, r.mvs_default.fp_rate);
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
               r.chip_name, mvs_default_str, mvs_nbvi_str, ml_str);
    }
    
    printf("\n");
    printf("Legend: R = Recall, FP = False Positive Rate\n");
    printf("Targets: MVS default >%.0f%% R, <%.1f%% FP | NBVI >=%.0f%% R, <=%.1f%% FP | ML >%.0f%% R, <%.1f%% FP\n",
           get_default_recall_target(), get_default_fp_rate_target(),
           get_nbvi_recall_target(), get_fp_rate_target(),
           get_ml_recall_target(), get_ml_fp_rate_target());
    printf("================================================================================\n");
    
    // Detailed table for PERFORMANCE.md
    printf("\n");
    printf("                         DETAILED METRICS (for PERFORMANCE.md)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("| Chip   | Algorithm   | Recall  | Precision | FP Rate | F1-Score |\n");
    printf("|--------|-------------|---------|-----------|---------|----------|\n");
    
    for (int i = 0; i < g_results_count; i++) {
        const ChipResults& r = g_results[i];
        
        if (r.mvs_default.valid) {
            printf("| %-6s | MVS Default | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   r.chip_name, r.mvs_default.recall, r.mvs_default.precision,
                   r.mvs_default.fp_rate, r.mvs_default.f1);
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
// Chip-Specific Configuration
// ============================================================================

inline bool is_esp32_chip() {
    return csi_test_data::current_chip() == csi_test_data::ChipType::ESP32;
}

// Determine whether CV normalization (std/mean) is needed for the current dataset.
// Uses 'gain_locked' metadata from the NPZ file when available; falls back to
// chip-based heuristics for older files that predate the field.
inline bool needs_cv_normalization() {
    if (csi_test_data::baseline_gain_locked_known()) {
        return !csi_test_data::baseline_gain_locked();
    }
    // Fallback: ESP32 has no hardware gain lock;
    return is_esp32_chip();
}

inline const char* get_pairing_mode() {
    return csi_test_data::is_temporally_paired() ? "paired" : "single-dataset fallback";
}

// Unified parameters for all chips (use production defaults)
inline uint16_t get_window_size() { return DETECTOR_DEFAULT_WINDOW_SIZE; }
inline bool get_enable_hampel() { return true; }

// MVS targets
// Default-band baseline test uses the same strict targets as NBVI/ML.
inline float get_default_fp_rate_target() { return 5.0f; }
inline float get_default_recall_target() { return 95.0f; }
// NBVI targets
inline float get_fp_rate_target() { return 5.0f; }
inline float get_nbvi_recall_target() { return 95.0f; }

// ML targets
inline float get_ml_fp_rate_target() { return 5.0f; }
inline float get_ml_recall_target() { return 95.0f; }

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// Test 1: MVS with Default Subcarriers (Production Baseline)
// ============================================================================
// Uses default offline-tuned subcarriers to validate production-baseline behavior.
// This serves as a fixed reference to measure NBVI impact.

void test_mvs_default_subcarriers(void) {
    float fp_target = get_default_fp_rate_target();
    float recall_target = get_default_recall_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    bool cv_norm = needs_cv_normalization();
    const int pkt_size = csi_test_data::packet_size();
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: MVS with Default Subcarriers (Production Baseline)\n");
    printf("  Chip: %s, Window: %d, CV Norm: %s\n", 
           csi_test_data::chip_name(csi_test_data::current_chip()), 
           window_size, cv_norm ? "ON" : "OFF");
    double pair_delta_sec = 0.0;
    if (csi_test_data::current_pair_delta_seconds(pair_delta_sec)) {
        printf("  Pair mode: %s (delta: %.1fs)\n", get_pairing_mode(), pair_delta_sec);
    } else {
        printf("  Pair mode: %s (delta: N/A)\n", get_pairing_mode());
    }
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Use default subcarriers for this chip.
    const uint8_t* default_band = DEFAULT_SUBCARRIERS;
    const uint8_t default_size = 12;
    printf("Default subcarriers: [");
    for (int i = 0; i < default_size; i++) {
        printf("%d", default_band[i]);
        if (i < default_size - 1) printf(", ");
    }
    printf("]\n\n");
    
    // Calculate adaptive threshold from baseline using selected band.
    MVSDetector cal_detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    cal_detector.configure_lowpass(false);
    cal_detector.configure_hampel(enable_hampel);
    cal_detector.set_cv_normalization(cv_norm);

    std::vector<float> mv_values;
    int calibration_packets = std::min(num_baseline, static_cast<int>(CALIBRATION_DEFAULT_BUFFER_SIZE));
    for (int i = 0; i < calibration_packets; i++) {
        cal_detector.process_packet((const int8_t*)baseline_packets[i], pkt_size,
                          default_band, default_size);
        cal_detector.update_state();
        if (cal_detector.is_ready()) {
            mv_values.push_back(cal_detector.get_motion_metric());
        }
    }

    float adaptive_threshold;
    uint8_t percentile;
    calculate_adaptive_threshold(mv_values, ThresholdMode::AUTO, adaptive_threshold, percentile);
    printf("Adaptive threshold: %.6f (P%d x %.1f, from %zu MV values)\n\n",
           adaptive_threshold, percentile, DEFAULT_ADAPTIVE_FACTOR, mv_values.size());
    
    // Create detector for evaluation
    MVSDetector detector(window_size, adaptive_threshold);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel);
    detector.set_cv_normalization(cv_norm);
    
    // Process baseline
    int baseline_motion = 0;
    for (int p = 0; p < num_baseline; p++) {
        detector.process_packet((const int8_t*)baseline_packets[p], pkt_size,
                          default_band, default_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            baseline_motion++;
        }
    }
    
    // Process movement
    int movement_motion = 0;
    for (int p = 0; p < num_movement; p++) {
        detector.process_packet((const int8_t*)movement_packets[p], pkt_size,
                          default_band, default_size);
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
    record_result("mvs_default", recall, fp_rate, precision, f1);
    
    TEST_ASSERT_TRUE_MESSAGE(recall > recall_target, "Recall too low");
    TEST_ASSERT_TRUE_MESSAGE(fp_rate < fp_target, "FP Rate too high");
}

// ============================================================================
// Test 2: MVS with NBVI Calibration (Production Case)
// ============================================================================
// Uses NBVI auto-calibration as in production.
// NBVI calibration runs for all chips.
// When CV normalization is needed (e.g., no gain lock), calibrator and detector
// use CV turbulence (std/mean) instead of raw std.

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
    double pair_delta_sec = 0.0;
    if (csi_test_data::current_pair_delta_seconds(pair_delta_sec)) {
        printf("  Pair mode: %s (delta: %.1fs)\n", get_pairing_mode(), pair_delta_sec);
    } else {
        printf("  Pair mode: %s (delta: N/A)\n", get_pairing_mode());
    }
    printf("═══════════════════════════════════════════════════════\n\n");
    
    MVSDetector detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel);
    detector.set_cv_normalization(cv_norm);

    CSIManager csi_manager;
    csi_manager.init(&detector, DEFAULT_SUBCARRIERS, 100, GainLockMode::DISABLED, &g_wifi_mock);

    const char* buffer_path = "/tmp/test_nbvi_buffer.bin";
    NBVICalibrator nbvi;
    nbvi.init(&csi_manager, buffer_path);
    nbvi.set_mvs_window_size(window_size);
    nbvi.set_cv_normalization(cv_norm);
    nbvi.configure_hampel(enable_hampel);

    uint16_t buffer_size = std::min(static_cast<int>(nbvi.get_buffer_size()), num_baseline);
    nbvi.set_buffer_size(buffer_size);

    bool calibration_success = false;
    uint8_t percentile_tmp = 95;
    float calibrated_threshold = 1.0f;
    uint8_t calibrated_band[12] = {0};
    uint8_t calibrated_size = 0;

    esp_err_t err = nbvi.start_calibration(DEFAULT_SUBCARRIERS, 12,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& mv_values, bool success) {
            if (success && size > 0) {
                memcpy(calibrated_band, band, size);
                calibrated_size = size;
                calculate_adaptive_threshold(mv_values, ThresholdMode::AUTO, calibrated_threshold, percentile_tmp);
            }
            calibration_success = success;
        });

    TEST_ASSERT_EQUAL_MESSAGE(ESP_OK, err, "NBVI calibration start failed");

    for (int i = 0; i < buffer_size && i < num_baseline; i++) {
        nbvi.add_packet(baseline_packets[i], pkt_size);
    }

    for (int wait = 0; wait < 500 && nbvi.is_calibrating(); wait++) {
        vTaskDelay(1);
    }

    TEST_ASSERT_TRUE_MESSAGE(calibration_success, "NBVI calibration failed");
    TEST_ASSERT_EQUAL_MESSAGE(12, calibrated_size, "NBVI band size mismatch");

    printf("  Selected band: [");
    for (int i = 0; i < calibrated_size; i++) {
        printf("%d", calibrated_band[i]);
        if (i < calibrated_size - 1) printf(", ");
    }
    printf("]\n");
    printf("  Adaptive threshold: %.6f\n\n", calibrated_threshold);

    detector.set_threshold(calibrated_threshold);
    detector.clear_buffer();

    int baseline_motion = 0;
    for (int i = 0; i < num_baseline; i++) {
        detector.process_packet((const int8_t*)baseline_packets[i], pkt_size, calibrated_band, calibrated_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) baseline_motion++;
    }

    int movement_motion = 0;
    for (int i = 0; i < num_movement; i++) {
        detector.process_packet((const int8_t*)movement_packets[i], pkt_size, calibrated_band, calibrated_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) movement_motion++;
    }

    float recall = (float)movement_motion / num_movement * 100.0f;
    float fp_rate = (float)baseline_motion / num_baseline * 100.0f;
    float precision = (movement_motion + baseline_motion > 0) ?
        (float)movement_motion / (movement_motion + baseline_motion) * 100.0f : 0.0f;
    float f1 = (precision + recall > 0) ?
        2.0f * (precision / 100.0f) * (recall / 100.0f) /
        ((precision + recall) / 100.0f) * 100.0f : 0.0f;

    printf("Results:\n");
    printf("  * Recall:    %.1f%% (target: >%.0f%%)\n", recall, recall_target);
    printf("  * FP Rate:   %.1f%% (target: <%.0f%%)\n", fp_rate, fp_target);
    printf("  * Precision: %.1f%%\n", precision);
    printf("  * F1-Score:  %.1f%%\n\n", f1);

    record_result("mvs_nbvi", recall, fp_rate, precision, f1);

    remove(buffer_path);

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
    detector.configure_hampel(get_enable_hampel());
    detector.set_cv_normalization(cv_norm);
    
    printf("ML subcarriers: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d] (fixed)\n",
           DEFAULT_SUBCARRIERS[0], DEFAULT_SUBCARRIERS[1], DEFAULT_SUBCARRIERS[2], DEFAULT_SUBCARRIERS[3],
           DEFAULT_SUBCARRIERS[4], DEFAULT_SUBCARRIERS[5], DEFAULT_SUBCARRIERS[6], DEFAULT_SUBCARRIERS[7],
           DEFAULT_SUBCARRIERS[8], DEFAULT_SUBCARRIERS[9], DEFAULT_SUBCARRIERS[10], DEFAULT_SUBCARRIERS[11]);
    printf("Threshold: %.1f\n\n", detector.get_threshold());
    
    // Warmup = window_size: detector needs full buffer before producing valid predictions
    const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
    
    // Process baseline (skip first warmup packets - buffer not ready)
    int baseline_motion = 0;
    for (int i = 0; i < num_baseline; i++) {
        detector.process_packet((const int8_t*)baseline_packets[i], pkt_size,
                               DEFAULT_SUBCARRIERS, 12);
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
                               DEFAULT_SUBCARRIERS, 12);
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
    RUN_TEST(test_mvs_default_subcarriers);   // Production baseline reference
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
