/*
 * ESPectre - Motion Detection Integration Tests
 * 
 * Integration tests for MVS and ML motion detection algorithms.
 * Tests motion detection performance with real CSI data.
 * 
 * Test Categories:
 *   1. test_mvs_fixed_subcarriers - MVS with fixed production subcarriers
 *   2. test_ml_detection - ML neural network detection
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_harness.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>

// Include headers from lib/espectre
#include "utils.h"
#include "filters.h"
#include "mvs_detector.h"
#include "ml_detector.h"
#include "threshold.h"
#include "esphome/core/log.h"
#include "esp_system.h"

using namespace esphome::espectre;

// Include CSI data loader (loads from NPZ files)
#include "csi_test_data.h"

// Compatibility macros for existing test code
#define static_presence_packets csi_test_data::static_presence_packets()
#define motion_packets csi_test_data::motion_packets()
#define num_static_presence csi_test_data::num_static_presence()
#define num_motion csi_test_data::num_motion()

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

struct DatasetResults {
    std::string dataset_name;
    const char* chip_name;
    PerformanceResult mvs;
    PerformanceResult ml;
};

static std::vector<DatasetResults> g_results;

// Forward declarations for target getters used in summary output.
inline float get_mvs_fp_rate_target();
inline float get_mvs_recall_target();
inline float get_ml_fp_rate_target();
inline float get_ml_recall_target();

static void record_result(const char* algorithm, float recall, float fp_rate, float precision, float f1) {
    const char* current_label = csi_test_data::current_pair_label();
    if (g_results.empty() || g_results.back().dataset_name != current_label) {
        DatasetResults row{};
        row.dataset_name = current_label;
        row.chip_name = csi_test_data::chip_name(csi_test_data::current_chip());
        row.mvs = {0, 0, 0, 0, false};
        row.ml = {0, 0, 0, 0, false};
        g_results.push_back(row);
    }
    
    DatasetResults& current = g_results.back();
    if (strcmp(algorithm, "mvs") == 0) {
        current.mvs = {recall, fp_rate, precision, f1, true};
    } else if (strcmp(algorithm, "ml") == 0) {
        current.ml = {recall, fp_rate, precision, f1, true};
    }
}

static PerformanceResult mean_result_for_chip(const char* chip_name, bool ml) {
    PerformanceResult mean{0, 0, 0, 0, false};
    int count = 0;
    for (const auto& r : g_results) {
        if (strcmp(r.chip_name, chip_name) != 0) {
            continue;
        }
        const PerformanceResult& value = ml ? r.ml : r.mvs;
        if (!value.valid) {
            continue;
        }
        mean.recall += value.recall;
        mean.fp_rate += value.fp_rate;
        mean.precision += value.precision;
        mean.f1 += value.f1;
        count++;
    }
    if (count == 0) {
        return mean;
    }
    mean.recall /= count;
    mean.fp_rate /= count;
    mean.precision /= count;
    mean.f1 /= count;
    mean.valid = true;
    return mean;
}

static int dataset_count_for_chip(const char* chip_name) {
    int count = 0;
    for (const auto& r : g_results) {
        if (strcmp(r.chip_name, chip_name) == 0) {
            count++;
        }
    }
    return count;
}

static void print_summary_table() {
    printf("\n");
    printf("================================================================================\n");
    printf("                      PERFORMANCE SUMMARY TABLE (C++)\n");
    printf("================================================================================\n");
    printf("\n");
    printf("| Chip   | Datasets | MVS                     | ML                      |\n");
    printf("|--------|----------|-------------------------|-------------------------|\n");

    for (auto chip : csi_test_data::get_supported_chips()) {
        const char* chip_name = csi_test_data::chip_name(chip);
        const int dataset_count = dataset_count_for_chip(chip_name);
        if (dataset_count == 0) {
            continue;
        }

        char mvs_str[32] = "N/A";
        char ml_str[32] = "N/A";
        const PerformanceResult mvs = mean_result_for_chip(chip_name, false);
        const PerformanceResult ml = mean_result_for_chip(chip_name, true);
        
        if (mvs.valid) {
            snprintf(mvs_str, sizeof(mvs_str), "%.1f%% R, %.1f%% FP",
                     mvs.recall, mvs.fp_rate);
        }
        if (ml.valid) {
            snprintf(ml_str, sizeof(ml_str), "%.1f%% R, %.1f%% FP",
                     ml.recall, ml.fp_rate);
        }
        
        printf("| %-6s | %8d | %-23s | %-23s |\n",
               chip_name, dataset_count, mvs_str, ml_str);
    }
    
    printf("\n");
    printf("Legend: R = Recall, FP = False Positive Rate\n");
    printf("Targets: MVS >%.0f%% R, <%.1f%% FP | ML >%.0f%% R, <%.1f%% FP\n",
           get_mvs_recall_target(), get_mvs_fp_rate_target(),
           get_ml_recall_target(), get_ml_fp_rate_target());
    printf("================================================================================\n");
    
    // Detailed table for PERFORMANCE.md
    printf("\n");
    printf("                         DETAILED METRICS (for PERFORMANCE.md)\n");
    printf("--------------------------------------------------------------------------------\n");
    printf("| Dataset                                         | Chip   | Algorithm   | Recall  | Precision | FP Rate | F1-Score |\n");
    printf("|-------------------------------------------------|--------|-------------|---------|-----------|---------|----------|\n");

    for (const auto& r : g_results) {
        std::string dataset_name = r.dataset_name;
        const size_t slash_pos = dataset_name.find_last_of('/');
        if (slash_pos != std::string::npos) {
            dataset_name = dataset_name.substr(slash_pos + 1);
        }
        
        if (r.mvs.valid) {
            printf("| %-47.47s | %-6s | MVS         | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   dataset_name.c_str(), r.chip_name, r.mvs.recall, r.mvs.precision,
                   r.mvs.fp_rate, r.mvs.f1);
        }
        if (r.ml.valid) {
            printf("| %-47.47s | %-6s | ML          | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   dataset_name.c_str(), r.chip_name, r.ml.recall, r.ml.precision,
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

// Unified parameters for all chips (use production defaults)
inline uint16_t get_window_size() { return DETECTOR_DEFAULT_WINDOW_SIZE; }
inline bool get_enable_hampel() { return true; }

// MVS targets
inline float get_mvs_fp_rate_target() { return 5.0f; }
inline float get_mvs_recall_target() { return 95.0f; }

// ML targets
inline float get_ml_fp_rate_target() { return 5.0f; }
inline float get_ml_recall_target() { return 95.0f; }

void setUp(void) {}
void tearDown(void) {}

// ============================================================================
// Test 1: MVS with Fixed Subcarriers (Production Runtime)
// ============================================================================
// Uses the shared fixed subcarriers and a static-presence-derived adaptive threshold.

void test_mvs_fixed_subcarriers(void) {
    float fp_target = get_mvs_fp_rate_target();
    float recall_target = get_mvs_recall_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    const int pkt_size = csi_test_data::packet_size();
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: MVS with Fixed Subcarriers (Production Runtime)\n");
    printf("  Chip: %s, Window: %d\n",
           csi_test_data::chip_name(csi_test_data::current_chip()), 
           window_size);
    printf("  Pair: %s\n", csi_test_data::current_pair_label());
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
    
    // Calculate adaptive threshold from static presence using the selected band.
    MVSDetector cal_detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    cal_detector.configure_lowpass(false);
    cal_detector.configure_hampel(enable_hampel);

    std::vector<float> mv_values;
    int calibration_packets = std::min(num_static_presence, static_cast<int>(CALIBRATION_DEFAULT_BUFFER_SIZE));
    for (int i = 0; i < calibration_packets; i++) {
        cal_detector.process_packet((const int8_t*)static_presence_packets[i], pkt_size,
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
    
    // Process static presence
    int static_presence_motion = 0;
    for (int p = 0; p < num_static_presence; p++) {
        detector.process_packet((const int8_t*)static_presence_packets[p], pkt_size,
                          default_band, default_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            static_presence_motion++;
        }
    }
    
    // Process motion
    int motion_detected = 0;
    for (int p = 0; p < num_motion; p++) {
        detector.process_packet((const int8_t*)motion_packets[p], pkt_size,
                          default_band, default_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            motion_detected++;
        }
    }
    
    // Calculate metrics
    float recall = (float)motion_detected / num_motion * 100.0f;
    float fp_rate = (float)static_presence_motion / num_static_presence * 100.0f;
    float precision = (motion_detected + static_presence_motion > 0) ?
        (float)motion_detected / (motion_detected + static_presence_motion) * 100.0f : 0.0f;
    float f1 = (precision + recall > 0) ?
        2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
    
    printf("Results:\n");
    printf("  * Recall:    %.1f%% (target: >%.0f%%)\n", recall, recall_target);
    printf("  * FP Rate:   %.1f%% (target: <%.0f%%)\n", fp_rate, fp_target);
    printf("  * Precision: %.1f%%\n", precision);
    printf("  * F1-Score:  %.1f%%\n\n", f1);
    
    // Record for summary table
    record_result("mvs", recall, fp_rate, precision, f1);
    
    TEST_ASSERT_TRUE_MESSAGE(recall > recall_target, "Recall too low");
    TEST_ASSERT_TRUE_MESSAGE(fp_rate < fp_target, "FP Rate too high");
}

// ============================================================================
// Test 2: ML Detection
// ============================================================================
// Tests ML neural network detector with fixed subcarriers.

void test_ml_detection(void) {
    float fp_target = get_ml_fp_rate_target();
    float recall_target = get_ml_recall_target();
    const int pkt_size = csi_test_data::packet_size();
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: ML Detection (Neural Network)\n");
    printf("  Chip: %s\n", csi_test_data::chip_name(csi_test_data::current_chip()));
    printf("═══════════════════════════════════════════════════════\n\n");
    
    MLDetector detector(DETECTOR_DEFAULT_WINDOW_SIZE, ML_DEFAULT_THRESHOLD);
    detector.configure_hampel(get_enable_hampel());
    
    printf("ML subcarriers: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d] (fixed)\n",
           DEFAULT_SUBCARRIERS[0], DEFAULT_SUBCARRIERS[1], DEFAULT_SUBCARRIERS[2], DEFAULT_SUBCARRIERS[3],
           DEFAULT_SUBCARRIERS[4], DEFAULT_SUBCARRIERS[5], DEFAULT_SUBCARRIERS[6], DEFAULT_SUBCARRIERS[7],
           DEFAULT_SUBCARRIERS[8], DEFAULT_SUBCARRIERS[9], DEFAULT_SUBCARRIERS[10], DEFAULT_SUBCARRIERS[11]);
    printf("Threshold: %.1f\n\n", detector.get_threshold());
    
    // Warmup = window_size: detector needs full buffer before producing valid predictions
    const int warmup = DETECTOR_DEFAULT_WINDOW_SIZE;
    
    // Process static presence (skip first warmup packets - buffer not ready)
    int static_presence_motion = 0;
    for (int i = 0; i < num_static_presence; i++) {
        detector.process_packet((const int8_t*)static_presence_packets[i], pkt_size,
                               DEFAULT_SUBCARRIERS, 12);
        detector.update_state();
        // Only count after warmup (when buffer is full)
        if (i >= warmup && detector.get_state() == MotionState::MOTION) {
            static_presence_motion++;
        }
    }
    
    // Process motion (skip first warmup packets - transition period)
    int motion_detected = 0;
    int motion_idle = 0;
    
    for (int i = 0; i < num_motion; i++) {
        detector.process_packet((const int8_t*)motion_packets[i], pkt_size,
                               DEFAULT_SUBCARRIERS, 12);
        detector.update_state();
        if (i >= warmup) {
            if (detector.get_state() == MotionState::MOTION) {
                motion_detected++;
            } else {
                motion_idle++;
            }
        }
    }
    
    int static_presence_eval = num_static_presence - warmup;
    int motion_eval = num_motion - warmup;
    float recall = (float)motion_detected / motion_eval * 100.0f;
    float fp_rate = (float)static_presence_motion / static_presence_eval * 100.0f;
    float precision = (motion_detected + static_presence_motion > 0) ?
        (float)motion_detected / (motion_detected + static_presence_motion) * 100.0f : 0.0f;
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

int run_tests_for_pair(int pair_index) {
    const csi_test_data::ChipType chip = csi_test_data::pair_chip(pair_index);
    printf("\n========================================\n");
    printf("Running tests with %s 64 SC dataset pair (HT20)\n", csi_test_data::chip_name(chip));
    printf("Pair: %s\n", csi_test_data::pair_label(pair_index));
    printf("========================================\n");
    
    const char* skip_reason = csi_test_data::chip_skip_reason(chip);
    if (skip_reason != nullptr) {
        printf("SKIPPED: %s\n", skip_reason);
        return 0;
    }
    
    if (!csi_test_data::switch_dataset_pair(pair_index)) {
        printf("ERROR: Failed to load %s dataset pair\n", csi_test_data::chip_name(chip));
        return 1;
    }
    
    UNITY_BEGIN();
    RUN_TEST(test_mvs_fixed_subcarriers);   // Production runtime path
    RUN_TEST(test_ml_detection);              // ML neural network
    return UNITY_END();
}

int process(void) {
    int failures = 0;
    const int pair_count = csi_test_data::get_available_pair_count();
    if (pair_count <= 0) {
        printf("ERROR: No complete 64 SC static-presence/motion dataset pairs available\n");
        return 1;
    }

    for (int pair_index = 0; pair_index < pair_count; pair_index++) {
        failures += run_tests_for_pair(pair_index);
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
