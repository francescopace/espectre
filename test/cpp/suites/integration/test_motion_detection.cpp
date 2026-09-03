/*
 * ESPectre - Motion Detection Integration Tests
 *
 * Integration tests for Lightweight and High Accuracy motion detection algorithms.
 * Tests motion detection performance with real CSI data.
 *
 * Test Categories:
 *   1. test_classic_fixed_subcarriers - Lightweight with fixed production subcarriers
 *   2. test_ml_detection - ML neural network detection
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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
#include "lightweight_detector.h"
#include "filters.h"
#include "high_accuracy_detector.h"
#include "runtime_sensing_schema.h"
#include "threshold.h"
#include "esphome/core/log.h"
#include "esp_system.h"
#include "csi_replay_timing.h"
#include "csi_replay_metrics.h"
#include "csi_replay_summary.h"

using namespace espectre;
namespace replay = espectre::test::replay;
namespace replay_summary = espectre::test::summary;

// Include CSI data loader (loads from NPZ files)
#include "csi_test_data.h"

// Compatibility macros for existing test code
#define static_presence_packets csi_test_data::static_presence_packets()
#define motion_packets csi_test_data::motion_packets()
#define num_static_presence csi_test_data::num_static_presence()
#define num_motion csi_test_data::num_motion()


static replay::ReplayPacketMetadata static_presence_metadata() {
    return {
        csi_test_data::static_presence_stream_seq_num(),
        csi_test_data::static_presence_device_ticks_us(),
        csi_test_data::static_presence_wifi_rx_ts_us(),
        csi_test_data::static_presence_csi_target_pps(),
    };
}

static replay::ReplayPacketMetadata motion_metadata() {
    return {
        csi_test_data::motion_stream_seq_num(),
        csi_test_data::motion_device_ticks_us(),
        csi_test_data::motion_wifi_rx_ts_us(),
        csi_test_data::motion_csi_target_pps(),
    };
}

// ============================================================================
// Performance Results Storage (for summary table)
// ============================================================================

struct PerformanceResult {
    float recall;
    float min_recall;
    float fp_rate;
    float max_fp_rate;
    float precision;
    float f1;
    int effective_alarms;
    bool valid;
};

struct DatasetResults {
    std::string dataset_name;
    const char* chip_name;
    bool synthetic;
    bool report_reserved;
    PerformanceResult classic;
    PerformanceResult ml;
};

static std::vector<DatasetResults> g_results;
static std::string g_missing_pair_reason;

// Forward declarations for target getters used in summary output.
inline float get_classic_fp_rate_target();
inline float get_classic_recall_target();
inline float get_ml_fp_rate_target();
inline float get_ml_recall_target();


static void record_result(const char* algorithm, float recall, float fp_rate, float precision, float f1,
                          int effective_alarms) {
    const char* current_label = csi_test_data::current_pair_label();
    if (g_results.empty() || g_results.back().dataset_name != current_label) {
        DatasetResults row{};
        row.dataset_name = current_label;
        row.chip_name = csi_test_data::chip_name(csi_test_data::current_chip());
        row.synthetic = csi_test_data::current_pair_is_synthetic();
        row.report_reserved = csi_test_data::current_pair_is_report_reserved();
        row.classic = {0, 0, 0, 0, 0, 0, 0, false};
        row.ml = {0, 0, 0, 0, 0, 0, 0, false};
        g_results.push_back(row);
    }
    
    DatasetResults& current = g_results.back();
    if (strcmp(algorithm, "classic") == 0) {
        current.classic = {recall, recall, fp_rate, fp_rate, precision, f1, effective_alarms, true};
    } else if (strcmp(algorithm, "ml") == 0) {
        current.ml = {recall, recall, fp_rate, fp_rate, precision, f1, effective_alarms, true};
    }
}

static PerformanceResult mean_result_for_chip(const char* chip_name, const char* algorithm,
                                              bool synthetic, bool report_only = false) {
    PerformanceResult mean{0, 0, 0, 0, 0, 0, 0, false};
    int count = 0;
    for (const auto& r : g_results) {
        if (strcmp(r.chip_name, chip_name) != 0 || r.synthetic != synthetic ||
            (report_only && !r.report_reserved)) {
            continue;
        }
        const PerformanceResult& value =
            (strcmp(algorithm, "ml") == 0) ? r.ml
            : (strcmp(algorithm, "classic") == 0) ? r.classic
                                                   : r.classic;
        if (!value.valid) {
            continue;
        }
        mean.recall += value.recall;
        mean.min_recall = count == 0 ? value.recall : std::min(mean.min_recall, value.recall);
        mean.fp_rate += value.fp_rate;
        mean.max_fp_rate = count == 0 ? value.fp_rate : std::max(mean.max_fp_rate, value.fp_rate);
        mean.precision += value.precision;
        mean.f1 += value.f1;
        mean.effective_alarms += value.effective_alarms;
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

static int dataset_count_for_chip(const char* chip_name, bool synthetic) {
    int count = 0;
    for (const auto& r : g_results) {
        if (strcmp(r.chip_name, chip_name) == 0 && r.synthetic == synthetic) {
            count++;
        }
    }
    return count;
}

static int valid_result_count_for_chip(const char* chip_name, const char* algorithm,
                                       bool synthetic, bool report_only = false) {
    int count = 0;
    for (const auto& r : g_results) {
        if (strcmp(r.chip_name, chip_name) != 0 || r.synthetic != synthetic ||
            (report_only && !r.report_reserved)) {
            continue;
        }
        const PerformanceResult& value =
            (strcmp(algorithm, "ml") == 0) ? r.ml
            : (strcmp(algorithm, "classic") == 0) ? r.classic
                                                  : r.classic;
        if (value.valid) {
            count++;
        }
    }
    return count;
}

static void assert_metrics_are_valid(float recall, float fp_rate, float precision, float f1) {
    TEST_ASSERT_TRUE(recall >= 0.0f && recall <= 100.0f);
    TEST_ASSERT_TRUE(fp_rate >= 0.0f && fp_rate <= 100.0f);
    TEST_ASSERT_TRUE(precision >= 0.0f && precision <= 100.0f);
    TEST_ASSERT_TRUE(f1 >= 0.0f && f1 <= 100.0f);
}

static void print_summary_table() {
    std::vector<replay_summary::DualDetectorSummaryRow> rows;

    for (auto chip : csi_test_data::get_supported_chips()) {
        const char* chip_name = csi_test_data::chip_name(chip);
        const int dataset_count = dataset_count_for_chip(chip_name, false);
        if (dataset_count == 0) {
            continue;
        }

        const PerformanceResult classic = mean_result_for_chip(chip_name, "classic", false);
        const PerformanceResult ml = mean_result_for_chip(chip_name, "ml", false);
        rows.push_back({
            chip_name,
            dataset_count,
            {classic.valid, classic.recall, classic.fp_rate},
            {ml.valid, ml.recall, ml.fp_rate},
        });
    }

    char targets_line[128];
    snprintf(targets_line, sizeof(targets_line),
             "Targets: Lightweight >%.0f%% R, <%.1f%% FP | ML >%.0f%% R, <%.1f%% FP",
             get_classic_recall_target(), get_classic_fp_rate_target(),
             get_ml_recall_target(), get_ml_fp_rate_target());
    replay_summary::print_dual_detector_summary_table(
        "                      PERFORMANCE SUMMARY TABLE (C++)",
        rows,
        "Legend: R = Recall, FP = False Positive Rate",
        targets_line);
    
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
        
        if (r.classic.valid) {
            printf("| %-47.47s | %-6s | CLASSIC     | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   dataset_name.c_str(), r.chip_name, r.classic.recall, r.classic.precision,
                   r.classic.fp_rate, r.classic.f1);
        }
        if (r.ml.valid) {
            printf("| %-47.47s | %-6s | ML          | %6.1f%% | %8.1f%% | %6.1f%% | %7.1f%% |\n",
                   dataset_name.c_str(), r.chip_name, r.ml.recall, r.ml.precision,
                   r.ml.fp_rate, r.ml.f1);
        }
    }
    
    printf("--------------------------------------------------------------------------------\n");
}

static void write_algorithm_json(FILE* handle, const char* algorithm, bool synthetic) {
    bool first_chip = true;
    for (auto chip : csi_test_data::get_supported_chips()) {
        const char* chip_name = csi_test_data::chip_name(chip);
        const PerformanceResult metrics = mean_result_for_chip(
            chip_name, algorithm, synthetic, true);
        const int count = valid_result_count_for_chip(
            chip_name, algorithm, synthetic, true);
        if (!metrics.valid || count == 0) {
            continue;
        }
        if (!first_chip) {
            fprintf(handle, ",");
        }
        first_chip = false;
        fprintf(
            handle,
            "\"%s\":{\"count\":%d,\"recall\":%.6f,\"min_recall\":%.6f,\"precision\":%.6f,"
            "\"fp_rate\":%.6f,\"max_fp_rate\":%.6f,\"f1\":%.6f,"
            "\"effective_alarms\":%d}",
            chip_name,
            count,
            metrics.recall,
            metrics.min_recall,
            metrics.precision,
            metrics.fp_rate,
            metrics.max_fp_rate,
            metrics.f1,
            metrics.effective_alarms);
    }
}

static void write_parity_payload_if_requested() {
    const char* output_dir = getenv("ESPECTRE_PARITY_OUTPUT_DIR");
    if (output_dir == nullptr || output_dir[0] == '\0') {
        return;
    }

    std::string path = std::string(output_dir) + "/test_motion_detection.json";
    FILE* handle = fopen(path.c_str(), "w");
    if (handle == nullptr) {
        printf("WARNING: failed to open parity output path: %s\n", path.c_str());
        return;
    }

    fprintf(handle, "{");
    fprintf(handle, "\"suite\":\"test_motion_detection\",");
    fprintf(handle, "\"paired\":{");
    fprintf(handle, "\"classic\":{");
    write_algorithm_json(handle, "classic", false);
    fprintf(handle, "},");
    fprintf(handle, "\"ml\":{");
    write_algorithm_json(handle, "ml", false);
    fprintf(handle, "}");
    fprintf(handle, "},");
    fprintf(handle, "\"paired_synthetic\":{");
    fprintf(handle, "\"classic\":{");
    write_algorithm_json(handle, "classic", true);
    fprintf(handle, "},");
    fprintf(handle, "\"ml\":{");
    write_algorithm_json(handle, "ml", true);
    fprintf(handle, "}");
    fprintf(handle, "}");
    fprintf(handle, "}\n");
    fclose(handle);
    printf("Wrote parity metrics to %s\n", path.c_str());
}

// ============================================================================
// Chip-Specific Configuration
// ============================================================================

inline bool is_esp32_chip() {
    return csi_test_data::current_chip() == csi_test_data::ChipType::ESP32;
}

// Unified parameters for all chips (use production defaults)
inline uint16_t get_window_size() {
    return replay::detector_window_packets(
        static_presence_metadata(), num_static_presence);
}
inline bool get_enable_hampel() { return true; }

// Lightweight targets
inline float get_classic_fp_rate_target() { return 3.0f; }
inline float get_classic_recall_target() { return 95.0f; }
inline float get_ml_fp_rate_target() { return 5.0f; }
inline float get_ml_recall_target() { return 95.0f; }

void setUp(void) {}
void tearDown(void) {}

void test_supported_chip_matrix_includes_s2(void) {
    csi_test_data::ChipType parsed_chip = csi_test_data::ChipType::C3;
    TEST_ASSERT_TRUE(csi_test_data::chip_from_string("S2", parsed_chip));
    TEST_ASSERT_EQUAL_INT(
        static_cast<int>(csi_test_data::ChipType::S2),
        static_cast<int>(parsed_chip));
    TEST_ASSERT_EQUAL_STRING("S2", csi_test_data::chip_name(parsed_chip));

    const std::vector<csi_test_data::ChipType> supported_chips =
        csi_test_data::get_supported_chips();
    TEST_ASSERT_TRUE(
        std::find(supported_chips.begin(), supported_chips.end(),
                  csi_test_data::ChipType::S2) != supported_chips.end());
}

// ============================================================================
// Test 1: Lightweight with Fixed Subcarriers (Production Runtime)
// ============================================================================
// Uses the same startup-calibration flow as the runtime: build the threshold
// from the Lightweight probability metric, apply its session adaptation, then warm-clear
// before evaluation.

void test_classic_fixed_subcarriers(void) {
    if (!g_missing_pair_reason.empty()) {
        TEST_IGNORE_MESSAGE(g_missing_pair_reason.c_str());
    }

    float fp_target = get_classic_fp_rate_target();
    float recall_target = get_classic_recall_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    const int pkt_size = csi_test_data::packet_size();

    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: Lightweight with Fixed Subcarriers (Production Runtime)\n");
    printf("  Chip: %s, Window: %d\n",
           csi_test_data::chip_name(csi_test_data::current_chip()),
           window_size);
    printf("  Pair: %s\n", csi_test_data::current_pair_label());
    printf("═══════════════════════════════════════════════════════\n\n");

    const uint8_t* default_band = DEFAULT_SUBCARRIERS;
    const uint8_t default_size = 12;

    LightweightDetector detector(window_size, LIGHTWEIGHT_DEFAULT_THRESHOLD);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel);

    int calibration_packets = std::min(
        num_static_presence,
        static_cast<int>(replay::calibration_packet_count(
            static_presence_metadata(), num_static_presence)));
    float adaptive_threshold = LIGHTWEIGHT_DEFAULT_THRESHOLD;
    const bool calibrated = replay::calibrate_lightweight_detector(
        detector,
        calibration_packets,
        static_presence_packets,
        num_static_presence,
        csi_test_data::static_presence_rssi_dbm(),
        static_presence_metadata(),
        pkt_size,
        default_band,
        default_size,
        adaptive_threshold);
    const float auto_factor = detector.get_startup_threshold_factor();

    printf("Adaptive threshold: %.6f (%s x %.1f)\n", adaptive_threshold,
           calibrated ? "shared calibration" : "default threshold", auto_factor);
    printf("Fusion: l1_delta + turb_autocorr (double Hampel)\n\n");

    const replay::ReplayMetrics metrics = replay::evaluate_detector(
        detector,
        static_presence_packets,
        num_static_presence,
        csi_test_data::static_presence_rssi_dbm(),
        static_presence_metadata(),
        motion_packets,
        num_motion,
        csi_test_data::motion_rssi_dbm(),
        motion_metadata(),
        pkt_size,
        default_band,
        default_size);

    printf("Results:\n");
    printf("  * Recall:    %.1f%% (target: >%.0f%%)\n", metrics.recall, recall_target);
    printf("  * FP Rate:   %.1f%% (target: <%.1f%%)\n", metrics.fp_rate, fp_target);
    printf("  * Precision: %.1f%%\n", metrics.precision);
    printf("  * F1-Score:  %.1f%%\n", metrics.f1);
    printf("  * Effective Alarms: %d\n\n", metrics.effective_alarms);

    record_result("classic", metrics.recall, metrics.fp_rate, metrics.precision, metrics.f1,
                  metrics.effective_alarms);

    assert_metrics_are_valid(metrics.recall, metrics.fp_rate, metrics.precision, metrics.f1);
}

// ============================================================================
// Test 2: ML Detection
// ============================================================================
// Tests ML neural network detector with fixed subcarriers.

void test_ml_detection(void) {
    if (!g_missing_pair_reason.empty()) {
        TEST_IGNORE_MESSAGE(g_missing_pair_reason.c_str());
    }

    float fp_target = get_ml_fp_rate_target();
    float recall_target = get_ml_recall_target();
    const int pkt_size = csi_test_data::packet_size();
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TEST: ML Detection (Neural Network)\n");
    printf("  Chip: %s\n", csi_test_data::chip_name(csi_test_data::current_chip()));
    printf("═══════════════════════════════════════════════════════\n\n");
    
    HighAccuracyDetector detector(get_window_size(), HIGH_ACCURACY_DEFAULT_THRESHOLD);
    detector.configure_hampel(get_enable_hampel());
    
    printf("ML subcarriers: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d] (fixed)\n",
           DEFAULT_SUBCARRIERS[0], DEFAULT_SUBCARRIERS[1], DEFAULT_SUBCARRIERS[2], DEFAULT_SUBCARRIERS[3],
           DEFAULT_SUBCARRIERS[4], DEFAULT_SUBCARRIERS[5], DEFAULT_SUBCARRIERS[6], DEFAULT_SUBCARRIERS[7],
           DEFAULT_SUBCARRIERS[8], DEFAULT_SUBCARRIERS[9], DEFAULT_SUBCARRIERS[10], DEFAULT_SUBCARRIERS[11]);
    printf("Threshold: %.1f\n\n", detector.get_threshold());
    
    const replay::ReplayMetrics metrics = replay::evaluate_detector(
        detector,
        static_presence_packets,
        num_static_presence,
        nullptr,
        static_presence_metadata(),
        motion_packets,
        num_motion,
        nullptr,
        motion_metadata(),
        pkt_size,
        DEFAULT_SUBCARRIERS,
        12);
    
    printf("Results:\n");
    printf("  * Recall:    %.1f%% (target: >%.0f%%)\n", metrics.recall, recall_target);
    printf("  * FP Rate:   %.1f%% (target: <%.0f%%)\n", metrics.fp_rate, fp_target);
    printf("  * Precision: %.1f%%\n", metrics.precision);
    printf("  * F1-Score:  %.1f%%\n", metrics.f1);
    printf("  * Effective Alarms: %d\n\n", metrics.effective_alarms);
    
    // Record for summary table
    record_result("ml", metrics.recall, metrics.fp_rate, metrics.precision, metrics.f1,
                  metrics.effective_alarms);
    
    assert_metrics_are_valid(metrics.recall, metrics.fp_rate, metrics.precision, metrics.f1);
}

// ============================================================================
// Test Runner
// ============================================================================

int run_tests_for_pair(int pair_index) {
    const csi_test_data::ChipType chip = csi_test_data::pair_chip(pair_index);
    g_missing_pair_reason.clear();
    printf("\n========================================\n");
    printf("Running tests with %s 64 SC dataset pair (HT20)\n", csi_test_data::chip_name(chip));
    printf("Pair: %s\n", csi_test_data::pair_label(pair_index));
    printf("========================================\n");
    
    if (!csi_test_data::switch_dataset_pair(pair_index)) {
        printf("ERROR: Failed to load %s dataset pair\n", csi_test_data::chip_name(chip));
        return 1;
    }
    
    UNITY_BEGIN();
    RUN_TEST(test_classic_fixed_subcarriers); // Production runtime path
    RUN_TEST(test_ml_detection);              // ML neural network
    return UNITY_END();
}

int run_supported_chip_matrix_test() {
    UNITY_BEGIN();
    RUN_TEST(test_supported_chip_matrix_includes_s2);
    return UNITY_END();
}

int run_skipped_tests_for_chip(csi_test_data::ChipType chip) {
    g_missing_pair_reason =
        std::string("No complete 64 SC static-presence/motion dataset pair available for chip ") +
        csi_test_data::chip_name(chip);
    printf("\n========================================\n");
    printf("Running tests with %s (dataset pending)\n", csi_test_data::chip_name(chip));
    printf("========================================\n");

    UNITY_BEGIN();
    RUN_TEST(test_classic_fixed_subcarriers);
    RUN_TEST(test_ml_detection);
    const int result = UNITY_END();
    g_missing_pair_reason.clear();
    return result;
}

int process(void) {
    int failures = run_supported_chip_matrix_test();
    g_results.clear();
    const int pair_count = csi_test_data::get_available_pair_count();
    if (pair_count <= 0) {
        printf("ERROR: No complete 64 SC static-presence/motion dataset pairs available\n");
        return 1;
    }

    for (int pair_index = 0; pair_index < pair_count; pair_index++) {
        failures += run_tests_for_pair(pair_index);
    }

    const std::vector<csi_test_data::ChipType> available_chips =
        csi_test_data::get_available_chips();
    for (csi_test_data::ChipType chip : csi_test_data::get_supported_chips()) {
        if (std::find(available_chips.begin(), available_chips.end(), chip) ==
            available_chips.end()) {
            failures += run_skipped_tests_for_chip(chip);
        }
    }
    
    // Print summary table at the end
    print_summary_table();
    write_parity_payload_if_requested();
    
    return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
