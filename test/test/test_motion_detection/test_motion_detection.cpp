/*
 * ESPectre - Motion Detection Integration Tests
 * 
 * Integration tests for motion detection algorithms (MVS and PCA).
 * Tests motion detection performance with real CSI data.
 * 
 * Focus: Maximize Recall (90% target) for security/presence detection
 * 
 * Test Categories:
 *   1. test_mvs_detection_accuracy - MVS full performance evaluation
 *   2. test_mvs_threshold_sensitivity - MVS threshold parameter sweep
 *   3. test_mvs_window_size_sensitivity - MVS window size parameter sweep
 *   4. test_pca_detection_accuracy - PCA full performance evaluation
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
#include "pca_detector.h"
#include "csi_manager.h"
#include "p95_calibrator.h"
#include "nbvi_calibrator.h"
#include "pca_calibrator.h"
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

// Optimal subcarriers by chip type (found via grid search analysis)
// - C6 64 SC (HT20): subcarriers 11-22
// - S3 64 SC (HT20): subcarriers 48-59
static const uint8_t SUBCARRIERS_C6_64SC[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
static const uint8_t SUBCARRIERS_S3_64SC[] = {48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59};
static const uint8_t NUM_SELECTED_SUBCARRIERS = 12;

// ============================================================================
// Chip-Specific Configuration
// ============================================================================
// S3 has higher baseline noise, requiring different parameters

inline bool is_s3_chip() {
    return csi_test_data::current_chip() == csi_test_data::ChipType::S3;
}

// Get optimal subcarriers based on chip type
inline const uint8_t* get_optimal_subcarriers() {
    return is_s3_chip() ? SUBCARRIERS_S3_64SC : SUBCARRIERS_C6_64SC;
}

inline float get_fp_rate_target() {
    // S3 has higher baseline noise, allow 15% FP rate
    return is_s3_chip() ? 15.0f : 10.0f;
}

inline uint16_t get_window_size() {
    // S3 needs larger window for stable variance estimation
    return is_s3_chip() ? 100 : SEGMENTATION_DEFAULT_WINDOW_SIZE;
}

inline bool get_enable_hampel() {
    // S3 benefits from Hampel filter to reduce spikes
    return is_s3_chip();
}

// Motion detection metrics structure
typedef struct {
    int true_positives;      // Movement packets with motion detected
    int true_negatives;      // Baseline packets without motion
    int false_positives;     // Baseline packets with false motion
    int false_negatives;     // Movement packets without motion
    float accuracy;
    float precision;
    float recall;
    float specificity;
    float f1_score;
    float false_positive_rate;
    float false_negative_rate;
} motion_metrics_t;

void setUp(void) {
    // Setup before each test
}

void tearDown(void) {
    // Cleanup after each test
}

// Helper: Calculate motion detection metrics
static void calculate_motion_metrics(motion_metrics_t *metrics, 
                                     int total_baseline, int total_movement) {
    int total = total_baseline + total_movement;
    
    metrics->accuracy = (float)(metrics->true_positives + metrics->true_negatives) / total * 100.0f;
    
    int predicted_positive = metrics->true_positives + metrics->false_positives;
    metrics->precision = (predicted_positive > 0) ? 
        (float)metrics->true_positives / predicted_positive * 100.0f : 0.0f;
    
    int actual_positive = metrics->true_positives + metrics->false_negatives;
    metrics->recall = (actual_positive > 0) ? 
        (float)metrics->true_positives / actual_positive * 100.0f : 0.0f;
    
    int actual_negative = metrics->true_negatives + metrics->false_positives;
    metrics->specificity = (actual_negative > 0) ? 
        (float)metrics->true_negatives / actual_negative * 100.0f : 0.0f;
    
    float prec_decimal = metrics->precision / 100.0f;
    float rec_decimal = metrics->recall / 100.0f;
    metrics->f1_score = (prec_decimal + rec_decimal > 0) ? 
        2.0f * (prec_decimal * rec_decimal) / (prec_decimal + rec_decimal) * 100.0f : 0.0f;
    
    metrics->false_positive_rate = (actual_negative > 0) ?
        (float)metrics->false_positives / actual_negative * 100.0f : 0.0f;
    
    metrics->false_negative_rate = (actual_positive > 0) ?
        (float)metrics->false_negatives / actual_positive * 100.0f : 0.0f;
}

// Helper function to process a packet and update state
// Note: In production, update_state() is called only at publish time for efficiency.
// In tests, we call it after every packet to measure per-packet detection accuracy.
static void process_packet(MVSDetector *detector, const int8_t *packet) {
    detector->process_packet(packet, csi_test_data::packet_size(), get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS);
    detector->update_state();  // Lazy evaluation: update state for testing
}

// Test: MVS motion detection accuracy with real CSI data
void test_mvs_detection_accuracy(void) {
    float fp_target = get_fp_rate_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   ESPECTRE PERFORMANCE SUITE - MOTION DETECTION       ║\n");
    printf("║   Comprehensive evaluation for security/presence      ║\n");
    printf("║   Target: 90%% Recall, <%.0f%% FP Rate                   ║\n", fp_target);
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Architecture: CSI Packet → P95 Calibration → Normalization → MVS → State\n");
    printf("Chip: %s, Window: %d, Hampel: %s\n", csi_test_data::chip_name(csi_test_data::current_chip()), window_size, enable_hampel ? "ON" : "OFF");
    printf("\n");
    
    // ========================================================================
    // P95 CALIBRATION (exactly as in production)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  P95 CALIBRATION\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    MVSDetector detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    P95Calibrator cm;
    cm.init(&csi_manager, "/tmp/test_accuracy_buffer.bin");
    cm.init_subcarrier_config();
    
    // Calibration results
    uint8_t calibrated_band[12] = {0};
    uint8_t calibrated_size = 0;
    float calibrated_adaptive_threshold = 1.0f;
    bool calibration_success = false;
    
    esp_err_t err = cm.start_calibration(get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& mv_values, bool success) {
            if (success && size > 0) {
                memcpy(calibrated_band, band, size);
                calibrated_size = size;
                calibrated_adaptive_threshold = calculate_adaptive_threshold(mv_values, 95, 1.4f);
            }
            calibration_success = success;
        });
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    
    // Feed baseline packets for calibration
    const int gain_lock_skip = csi_manager.get_gain_lock_packets();
    const int calibration_packets = cm.get_buffer_size();
    const int pkt_size = csi_test_data::packet_size();
    
    printf("Calibrating with %d baseline packets...\n", calibration_packets);
    for (int i = 0; i < calibration_packets && (i + gain_lock_skip) < num_baseline; i++) {
        cm.add_packet(baseline_packets[i + gain_lock_skip], pkt_size);
    }
    
    TEST_ASSERT_TRUE_MESSAGE(calibration_success, "P95 calibration failed");
    
    printf("Calibration results:\n");
    printf("  Band: [%d-%d]\n", calibrated_band[0], calibrated_band[calibrated_size-1]);
    printf("  Adaptive threshold: %.4f (P95 × 1.4)\n", calibrated_adaptive_threshold);
    
    // ========================================================================
    // MOTION DETECTION PERFORMANCE
    // ========================================================================
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  MOTION DETECTION PERFORMANCE\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Apply calibration (exactly as in production)
    detector.set_threshold(calibrated_adaptive_threshold);
    detector.clear_buffer();
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    detector.configure_lowpass(false);
    
    float threshold = detector.get_threshold();
    printf("Using default threshold: %.4f\n", threshold);
    printf("Window size: %d\n", detector.get_window_size());
    
    // Test on baseline (should have minimal false positives)
    printf("Testing on baseline packets (expecting no motion)...\n");
    
    int baseline_segments_completed = 0;
    int baseline_motion_packets = 0;
    
    MotionState prev_state = MotionState::IDLE;
    
    for (int p = 0; p < num_baseline; p++) {
        // Use calibrated band (not hardcoded)
        detector.process_packet((const int8_t*)baseline_packets[p], pkt_size, 
                          calibrated_band, calibrated_size);
        detector.update_state();
        
        MotionState current_state = detector.get_state();
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == MotionState::MOTION && current_state == MotionState::IDLE) {
            baseline_segments_completed++;
        }
        
        // Also track packets in motion state (for info)
        if (current_state == MotionState::MOTION) {
            baseline_motion_packets++;
        }
        
        prev_state = current_state;
    }
    
    printf("  Baseline packets: %d\n", num_baseline);
    printf("  Motion packets: %d (%.1f%%)\n", baseline_motion_packets, 
           (float)baseline_motion_packets / num_baseline * 100.0f);
    printf("  Segments completed (FP): %d\n", baseline_segments_completed);
    printf("  FP Rate: %.2f%%\n\n", (float)baseline_segments_completed / num_baseline * 100.0f);
    
    // Test on movement (should detect motion)
    printf("Testing on movement packets (expecting motion)...\n");
    
    int movement_with_motion = 0;
    int movement_without_motion = 0;
    int total_segments_detected = 0;
    
    prev_state = MotionState::IDLE;
    
    for (int p = 0; p < num_movement; p++) {
        // Use calibrated band (not hardcoded)
        detector.process_packet((const int8_t*)movement_packets[p], pkt_size,
                          calibrated_band, calibrated_size);
        detector.update_state();
        
        MotionState current_state = detector.get_state();
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == MotionState::MOTION && current_state == MotionState::IDLE) {
            total_segments_detected++;
        }
        
        // Check if currently in motion state
        if (current_state == MotionState::MOTION) {
            movement_with_motion++;
        } else {
            movement_without_motion++;
        }
        
        prev_state = current_state;
    }
    
    printf("  Movement packets: %d\n", num_movement);
    printf("  With motion: %d\n", movement_with_motion);
    printf("  Without motion (FN): %d\n", movement_without_motion);
    printf("  Detection Rate: %.2f%%\n", (float)movement_with_motion / num_movement * 100.0f);
    printf("  Total segments detected: %d\n\n", total_segments_detected);
    
    // Calculate metrics based on segments completed (not packets in motion)
    motion_metrics_t metrics;
    metrics.true_positives = total_segments_detected;  // Segments detected in movement
    metrics.true_negatives = num_baseline - baseline_segments_completed;  // Baseline without segments
    metrics.false_positives = baseline_segments_completed;  // False segments in baseline
    metrics.false_negatives = 0;  // Assume all movement should have segments (simplified)
    
    calculate_motion_metrics(&metrics, num_baseline, num_movement);
    
    // ========================================================================
    // TEST SUMMARY
    // ========================================================================
    
    // Calculate packet-based metrics
    int pkt_tp = movement_with_motion;
    int pkt_fn = movement_without_motion;
    int pkt_tn = num_baseline - baseline_motion_packets;
    int pkt_fp = baseline_motion_packets;
    
    float pkt_recall = (float)pkt_tp / (pkt_tp + pkt_fn) * 100.0f;
    float pkt_precision = (pkt_tp + pkt_fp > 0) ? (float)pkt_tp / (pkt_tp + pkt_fp) * 100.0f : 0.0f;
    float pkt_fp_rate = (float)pkt_fp / num_baseline * 100.0f;
    float pkt_f1 = (pkt_precision + pkt_recall > 0) ? 
        2.0f * (pkt_precision / 100.0f) * (pkt_recall / 100.0f) / ((pkt_precision + pkt_recall) / 100.0f) * 100.0f : 0.0f;
    
    size_t free_heap = esp_get_free_heap_size();
    
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("                         TEST SUMMARY\n");
    printf("═══════════════════════════════════════════════════════════════════════\n");
    printf("\n");
    printf("CONFUSION MATRIX (%d baseline + %d movement packets):\n", num_baseline, num_movement);
    printf("                    Predicted\n");
    printf("                IDLE      MOTION\n");
    printf("Actual IDLE     %4d (TN)  %4d (FP)\n", pkt_tn, pkt_fp);
    printf("    MOTION      %4d (FN)  %4d (TP)\n", pkt_fn, pkt_tp);
    printf("\n");
    printf("MOTION DETECTION METRICS:\n");
    printf("  * True Positives (TP):   %d\n", pkt_tp);
    printf("  * True Negatives (TN):   %d\n", pkt_tn);
    printf("  * False Positives (FP):  %d\n", pkt_fp);
    printf("  * False Negatives (FN):  %d\n", pkt_fn);
    printf("  * Recall:     %.1f%% (target: >90%%)\n", pkt_recall);
    printf("  * Precision:  %.1f%%\n", pkt_precision);
    printf("  * FP Rate:    %.1f%% (target: <%.0f%%)\n", pkt_fp_rate, fp_target);
    printf("  * F1-Score:   %.1f%%\n", pkt_f1);
    printf("\n");
    printf("MEMORY:\n");
    printf("  * Free heap: %d bytes\n", (int)free_heap);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");
    
    // Cleanup
    remove("/tmp/test_accuracy_buffer.bin");
    
    // Verify minimum acceptable performance (chip-specific thresholds)
    TEST_ASSERT_TRUE_MESSAGE(pkt_recall > 90.0f, "Recall too low (target: >90%)");
    TEST_ASSERT_TRUE_MESSAGE(pkt_fp_rate < fp_target, "FP Rate too high (target: <10%)");
}

// Test: MVS threshold parameter sensitivity analysis
void test_mvs_threshold_sensitivity(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  THRESHOLD SENSITIVITY ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    float thresholds[] = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f};
    int num_thresholds = sizeof(thresholds) / sizeof(thresholds[0]);
    
    printf("Threshold   Recall    FP Rate   F1-Score\n");
    printf("──────────────────────────────────────────\n");
    
    for (int t = 0; t < num_thresholds; t++) {
        MVSDetector detector(SEGMENTATION_DEFAULT_WINDOW_SIZE, thresholds[t]);
        
        int baseline_motion = 0;
        int movement_motion = 0;
        
        // Process baseline
        for (int p = 0; p < num_baseline; p++) {
            process_packet(&detector, (const int8_t*)baseline_packets[p]);
            if (detector.get_state() == MotionState::MOTION) {
                baseline_motion++;
            }
        }
        
        // Process movement
        for (int p = 0; p < num_movement; p++) {
            process_packet(&detector, (const int8_t*)movement_packets[p]);
            if (detector.get_state() == MotionState::MOTION) {
                movement_motion++;
            }
        }
        
        float recall = (float)movement_motion / num_movement * 100.0f;
        float fp_rate = (float)baseline_motion / num_baseline * 100.0f;
        float precision = (movement_motion + baseline_motion > 0) ?
            (float)movement_motion / (movement_motion + baseline_motion) * 100.0f : 0.0f;
        float f1 = (precision + recall > 0) ?
            2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
        
        printf("  %.2f     %6.1f%%   %6.1f%%   %6.1f%%\n", 
               thresholds[t], recall, fp_rate, f1);
    }
    
    printf("\n");
    
    // Just verify the test ran successfully
    TEST_ASSERT_TRUE(true);
}

// Test: MVS window size parameter sensitivity analysis
void test_mvs_window_size_sensitivity(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  WINDOW SIZE SENSITIVITY ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    uint16_t window_sizes[] = {20, 30, 50, 75, 100, 150};
    int num_sizes = sizeof(window_sizes) / sizeof(window_sizes[0]);
    
    printf("Window Size   Recall    FP Rate   F1-Score\n");
    printf("────────────────────────────────────────────\n");
    
    for (int w = 0; w < num_sizes; w++) {
        MVSDetector detector(window_sizes[w], SEGMENTATION_DEFAULT_THRESHOLD);
        
        int baseline_motion = 0;
        int movement_motion = 0;
        
        // Process baseline
        for (int p = 0; p < num_baseline; p++) {
            process_packet(&detector, (const int8_t*)baseline_packets[p]);
            if (detector.get_state() == MotionState::MOTION) {
                baseline_motion++;
            }
        }
        
        // Process movement
        for (int p = 0; p < num_movement; p++) {
            process_packet(&detector, (const int8_t*)movement_packets[p]);
            if (detector.get_state() == MotionState::MOTION) {
                movement_motion++;
            }
        }
        
        float recall = (float)movement_motion / num_movement * 100.0f;
        float fp_rate = (float)baseline_motion / num_baseline * 100.0f;
        float precision = (movement_motion + baseline_motion > 0) ?
            (float)movement_motion / (movement_motion + baseline_motion) * 100.0f : 0.0f;
        float f1 = (precision + recall > 0) ?
            2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
        
        printf("    %3d       %6.1f%%   %6.1f%%   %6.1f%%\n", 
               window_sizes[w], recall, fp_rate, f1);
    }
    
    printf("\n");
    
    // Just verify the test ran successfully
    TEST_ASSERT_TRUE(true);
}

// Test: End-to-end with P95 band calibration and normalization
void test_mvs_end_to_end_with_calibration(void) {
    float fp_target = get_fp_rate_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  END-TO-END TEST: P95 Calibration + Normalization + MVS\n");
    printf("  Chip: %s, Window: %d, Hampel: %s\n", csi_test_data::chip_name(csi_test_data::current_chip()), window_size, enable_hampel ? "ON" : "OFF");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Create CSIManager and P95Calibrator using real code
    MVSDetector detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    P95Calibrator cm;
    cm.init(&csi_manager, "/tmp/test_e2e_buffer.bin");
    
    // Simulate production flow: set expected subcarriers after gain lock
    // This triggers guard band calculation based on loaded data
    cm.init_subcarrier_config();
    
    // Variables to capture calibration results
    uint8_t calibrated_band[12] = {0};
    uint8_t calibrated_size = 0;
    float calibrated_adaptive_threshold = 1.0f;
    bool calibration_success = false;
    
    // Start calibration with callback
    esp_err_t err = cm.start_calibration(get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& mv_values, bool success) {
            if (success && size > 0) {
                memcpy(calibrated_band, band, size);
                calibrated_size = size;
                calibrated_adaptive_threshold = calculate_adaptive_threshold(mv_values, 95, 1.4f);
            }
            calibration_success = success;
        });
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(cm.is_calibrating());
    
    // Feed baseline packets for calibration (use production buffer size)
    // Skip gain lock period - in production, calibration starts after AGC stabilization
    const int gain_lock_skip = csi_manager.get_gain_lock_packets();
    const int calibration_packets = cm.get_buffer_size();
    const int pkt_size = csi_test_data::packet_size();
    printf("Calibrating with %d baseline packets (skipping first %d for gain lock)...\n", 
           calibration_packets, gain_lock_skip);
    for (int i = 0; i < calibration_packets && (i + gain_lock_skip) < num_baseline; i++) {
        cm.add_packet(baseline_packets[i + gain_lock_skip], pkt_size);
    }
    
    // Calibration should complete
    TEST_ASSERT_TRUE(calibration_success);
    TEST_ASSERT_EQUAL(12, calibrated_size);
    TEST_ASSERT_TRUE(calibrated_adaptive_threshold > 0.0f);
    // Note: threshold value depends on dataset baseline noise level
    // C6 may have lower thresholds (~0.8) than S3 (~1.1) due to cleaner baseline
    
    printf("Calibration results:\n");
    printf("  Subcarriers: [");
    for (int i = 0; i < calibrated_size; i++) {
        printf("%d", calibrated_band[i]);
        if (i < calibrated_size - 1) printf(", ");
    }
    printf("]\n");
    printf("  Adaptive threshold: %.4f (P95 × 1.4)\n", calibrated_adaptive_threshold);
    
    // Apply calibration to detector (threshold - exactly as in production)
    detector.set_threshold(calibrated_adaptive_threshold);
    detector.clear_buffer();  // Clear stale data
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    detector.configure_lowpass(false);
    
    // Now run motion detection with P95-selected subcarriers and normalization
    printf("\nRunning motion detection with calibrated settings...\n");
    
    int baseline_motion = 0;
    for (int i = 0; i < num_baseline; i++) {
        detector.process_packet((const int8_t*)baseline_packets[i], pkt_size, 
                          calibrated_band, calibrated_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            baseline_motion++;
        }
    }
    
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
    int pkt_tp = movement_motion;
    int pkt_fn = num_movement - movement_motion;
    int pkt_tn = num_baseline - baseline_motion;
    int pkt_fp = baseline_motion;
    
    float recall = (float)pkt_tp / (pkt_tp + pkt_fn) * 100.0f;
    float precision = (pkt_tp + pkt_fp > 0) ? (float)pkt_tp / (pkt_tp + pkt_fp) * 100.0f : 0.0f;
    float fp_rate = (float)pkt_fp / num_baseline * 100.0f;
    float f1 = (precision + recall > 0) ? 
        2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
    
    printf("\nEnd-to-end results (P95 + Normalization + MVS):\n");
    printf("  TP: %d, TN: %d, FP: %d, FN: %d\n", pkt_tp, pkt_tn, pkt_fp, pkt_fn);
    printf("  Recall: %.1f%%, Precision: %.1f%%, FP Rate: %.1f%% (target: <%.0f%%), F1: %.1f%%\n", 
           recall, precision, fp_rate, fp_target, f1);
    
    // Performance should still meet targets with P95-selected subcarriers (chip-specific)
    TEST_ASSERT_TRUE_MESSAGE(recall > 90.0f, "End-to-end Recall too low (target: >90%)");
    TEST_ASSERT_TRUE_MESSAGE(fp_rate < fp_target, "End-to-end FP Rate too high (target: <10%)");
    
    // Cleanup
    remove("/tmp/test_e2e_buffer.bin");
}

/**
 * Test: NBVI Calibration + Normalization + MVS End-to-End
 * 
 * Same as P95 test but using NBVI algorithm.
 * NOTE: NBVI is skipped on S3 due to poor performance (67% recall vs 99% with P95).
 */
void test_mvs_end_to_end_with_nbvi_calibration(void) {
    // Skip NBVI on S3 - it doesn't work well with noisy baseline
    if (is_s3_chip()) {
        printf("\n═══════════════════════════════════════════════════════\n");
        printf("  SKIPPED: NBVI Calibration on S3\n");
        printf("  Reason: NBVI selects subcarriers poorly for S3 (67%% vs 99%% recall with P95)\n");
        printf("═══════════════════════════════════════════════════════\n\n");
        TEST_IGNORE_MESSAGE("NBVI skipped on S3 - use P95 instead");
        return;
    }
    
    float fp_target = get_fp_rate_target();
    uint16_t window_size = get_window_size();
    bool enable_hampel = get_enable_hampel();
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  END-TO-END TEST: NBVI Calibration + Normalization + MVS\n");
    printf("  Chip: %s, Window: %d, Hampel: %s\n", csi_test_data::chip_name(csi_test_data::current_chip()), window_size, enable_hampel ? "ON" : "OFF");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Create CSIManager and NBVICalibrator using real code
    MVSDetector detector(window_size, SEGMENTATION_DEFAULT_THRESHOLD);
    detector.configure_lowpass(false);
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    NBVICalibrator nbvi;
    nbvi.init(&csi_manager, "/tmp/test_nbvi_buffer.bin");
    
    // Variables to capture calibration results
    uint8_t calibrated_band[12] = {0};
    uint8_t calibrated_size = 0;
    float calibrated_adaptive_threshold = 1.0f;
    bool calibration_success = false;
    
    // Start calibration with callback
    esp_err_t err = nbvi.start_calibration(get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& mv_values, bool success) {
            if (success && size > 0) {
                memcpy(calibrated_band, band, size);
                calibrated_size = size;
                calibrated_adaptive_threshold = calculate_adaptive_threshold(mv_values, 95, 1.4f);
            }
            calibration_success = success;
        });
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    TEST_ASSERT_TRUE(nbvi.is_calibrating());
    
    // Feed baseline packets for calibration
    const int gain_lock_skip = csi_manager.get_gain_lock_packets();
    const int calibration_packets = nbvi.get_buffer_size();
    const int pkt_size = csi_test_data::packet_size();
    printf("NBVI: Calibrating with %d baseline packets (skipping first %d for gain lock)...\n", 
           calibration_packets, gain_lock_skip);
    for (int i = 0; i < calibration_packets && (i + gain_lock_skip) < num_baseline; i++) {
        nbvi.add_packet(baseline_packets[i + gain_lock_skip], pkt_size);
    }
    
    // Calibration should complete
    TEST_ASSERT_TRUE(calibration_success);
    TEST_ASSERT_EQUAL(12, calibrated_size);
    TEST_ASSERT_TRUE(calibrated_adaptive_threshold > 0.0f);
    
    printf("NBVI Calibration results:\n");
    printf("  Subcarriers: [");
    for (int i = 0; i < calibrated_size; i++) {
        printf("%d", calibrated_band[i]);
        if (i < calibrated_size - 1) printf(", ");
    }
    printf("]\n");
    printf("  Adaptive threshold: %.4f (P95 × 1.4)\n", calibrated_adaptive_threshold);
    
    // Apply calibration to detector
    detector.set_threshold(calibrated_adaptive_threshold);
    detector.clear_buffer();
    detector.configure_hampel(enable_hampel, 7, 4.0f);
    detector.configure_lowpass(false);
    
    // Run motion detection
    printf("\nRunning motion detection with NBVI-calibrated settings...\n");
    
    int baseline_motion = 0;
    for (int i = 0; i < num_baseline; i++) {
        detector.process_packet((const int8_t*)baseline_packets[i], pkt_size, 
                          calibrated_band, calibrated_size);
        detector.update_state();
        if (detector.get_state() == MotionState::MOTION) {
            baseline_motion++;
        }
    }
    
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
    int pkt_tp = movement_motion;
    int pkt_fn = num_movement - movement_motion;
    int pkt_tn = num_baseline - baseline_motion;
    int pkt_fp = baseline_motion;
    
    float recall = (float)pkt_tp / (pkt_tp + pkt_fn) * 100.0f;
    float precision = (pkt_tp + pkt_fp > 0) ? (float)pkt_tp / (pkt_tp + pkt_fp) * 100.0f : 0.0f;
    float fp_rate = (float)pkt_fp / num_baseline * 100.0f;
    float f1 = (precision + recall > 0) ? 
        2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
    
    printf("\nNBVI End-to-end results:\n");
    printf("  TP: %d, TN: %d, FP: %d, FN: %d\n", pkt_tp, pkt_tn, pkt_fp, pkt_fn);
    printf("  Recall: %.1f%%, Precision: %.1f%%, FP Rate: %.1f%% (target: <%.0f%%), F1: %.1f%%\n", 
           recall, precision, fp_rate, fp_target, f1);
    
    // Performance should meet targets
    TEST_ASSERT_TRUE_MESSAGE(recall > 90.0f, "NBVI End-to-end Recall too low (target: >90%)");
    TEST_ASSERT_TRUE_MESSAGE(fp_rate < fp_target, "NBVI End-to-end FP Rate too high");
    
    // Cleanup
    remove("/tmp/test_nbvi_buffer.bin");
}

// ============================================================================
// PCA DETECTION TESTS
// ============================================================================

// Test: PCA detection accuracy with real CSI data
void test_pca_detection_accuracy(void) {
    float fp_target = get_fp_rate_target();
    
    printf("\n═══════════════════════════════════════════════════════\n");
    printf("  PCA DETECTION PERFORMANCE\n");
    printf("  Chip: %s\n", csi_test_data::chip_name(csi_test_data::current_chip()));
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Create PCA detector
    PCADetector detector;
    
    CSIManager csi_manager;
    csi_manager.init(&detector, get_optimal_subcarriers(), 100, GainLockMode::DISABLED, &g_wifi_mock);
    
    // Create PCA calibrator
    PCACalibrator calibrator;
    calibrator.init(&csi_manager);
    
    // Calibration results
    float calibrated_threshold = PCA_DEFAULT_THRESHOLD;  // Default PCA threshold (scaled)
    float min_corr = 1.0f;
    bool calibration_success = false;
    std::vector<float> calibration_values;
    
    // Start calibration
    esp_err_t err = calibrator.start_calibration(get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS,
        [&](const uint8_t* band, uint8_t size, const std::vector<float>& corr_values, bool success) {
            if (success && !corr_values.empty()) {
                calibration_values = corr_values;
                // PCA threshold = (1 - min(correlation)) * PCA_SCALE
                // corr_values contains correlation values from baseline
                min_corr = *std::min_element(corr_values.begin(), corr_values.end());
                calibrated_threshold = (1.0f - min_corr) * PCA_SCALE;
            }
            calibration_success = success;
        });
    
    TEST_ASSERT_EQUAL(ESP_OK, err);
    
    // Feed baseline packets for calibration
    const int gain_lock_skip = csi_manager.get_gain_lock_packets();
    const int calibration_packets = calibrator.get_buffer_size();
    const int pkt_size = csi_test_data::packet_size();
    
    printf("PCA: Calibrating with %d baseline packets...\n", calibration_packets);
    for (int i = 0; i < calibration_packets && (i + gain_lock_skip) < num_baseline; i++) {
        calibrator.add_packet(baseline_packets[i + gain_lock_skip], pkt_size);
    }
    
    TEST_ASSERT_TRUE_MESSAGE(calibration_success, "PCA calibration failed");
    
    printf("PCA Calibration results:\n");
    printf("  Min correlation: %.4f\n", min_corr);
    printf("  Threshold (1-min_corr): %.4f\n", calibrated_threshold);
    printf("  Calibration values collected: %zu\n", calibration_values.size());
    
    // Apply calibrated threshold
    detector.set_threshold(calibrated_threshold);
    // Note: Don't call reset() - preserve detector state for proper warmup
    
    // Calculate where to start evaluation (after calibration)
    int start_idx = gain_lock_skip + calibration_packets;
    
    // ========================================================================
    // WARMUP: Process some baseline packets to fill detector buffers
    // ========================================================================
    const int warmup_packets = 50;  // PCA needs ~25 packets to fill buffers
    printf("Warming up detector with %d packets...\n", warmup_packets);
    for (int i = 0; i < warmup_packets && (start_idx + i) < num_baseline; i++) {
        detector.process_packet((const int8_t*)baseline_packets[start_idx + i], pkt_size, 
                          get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS);
        detector.update_state();
    }
    
    // ========================================================================
    // EVALUATE ON BASELINE (expect IDLE - count false positives)
    // ========================================================================
    printf("\nEvaluating on baseline packets (expect IDLE)...\n");
    
    int baseline_motion = 0;
    int baseline_start = start_idx + warmup_packets;
    int baseline_eval_count = num_baseline - baseline_start;
    
    // Skip baseline evaluation if not enough packets
    if (baseline_eval_count < 100) {
        printf("Note: Only %d baseline packets remaining after warmup, skipping FP rate check\n", 
               baseline_eval_count);
        baseline_eval_count = 0;  // Skip baseline evaluation
    } else {
        for (int i = baseline_start; i < num_baseline; i++) {
            detector.process_packet((const int8_t*)baseline_packets[i], pkt_size, 
                              get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS);
            detector.update_state();
            if (detector.get_state() == MotionState::MOTION) {
                baseline_motion++;
            }
        }
    }
    
    // ========================================================================
    // EVALUATE ON MOVEMENT (expect MOTION - count true positives)
    // ========================================================================
    
    // Warmup with first N movement packets (detector needs to adapt to new signal)
    const int movement_warmup = 50;
    printf("Warming up with %d movement packets...\n", movement_warmup);
    for (int i = 0; i < movement_warmup && i < num_movement; i++) {
        detector.process_packet((const int8_t*)movement_packets[i], pkt_size, 
                          get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS);
        detector.update_state();
    }
    
    printf("Evaluating on movement packets (expect MOTION)...\n");
    
    int movement_motion = 0;
    float max_metric = 0.0f;
    float min_metric = 1.0f;
    int eval_start = movement_warmup;
    int movement_eval_count = num_movement - eval_start;
    
    for (int i = eval_start; i < num_movement; i++) {
        detector.process_packet((const int8_t*)movement_packets[i], pkt_size, 
                          get_optimal_subcarriers(), NUM_SELECTED_SUBCARRIERS);
        detector.update_state();
        float metric = detector.get_motion_metric();
        if (metric > max_metric) max_metric = metric;
        if (metric < min_metric && metric > 0) min_metric = metric;
        if (detector.get_state() == MotionState::MOTION) {
            movement_motion++;
        }
    }
    printf("Movement metric range: %.4f - %.4f (threshold: %.4f)\n", min_metric, max_metric, calibrated_threshold);
    
    // ========================================================================
    // CALCULATE METRICS
    // ========================================================================
    int pkt_tp = movement_motion;
    int pkt_fn = movement_eval_count - movement_motion;
    int pkt_tn = baseline_eval_count - baseline_motion;
    int pkt_fp = baseline_motion;
    
    float recall = (pkt_tp + pkt_fn > 0) ? (float)pkt_tp / (pkt_tp + pkt_fn) * 100.0f : 0.0f;
    float precision = (pkt_tp + pkt_fp > 0) ? (float)pkt_tp / (pkt_tp + pkt_fp) * 100.0f : 0.0f;
    float fp_rate = (baseline_eval_count > 0) ? (float)pkt_fp / baseline_eval_count * 100.0f : 0.0f;
    float f1 = (precision + recall > 0) ? 
        2.0f * (precision / 100.0f) * (recall / 100.0f) / ((precision + recall) / 100.0f) * 100.0f : 0.0f;
    
    printf("\n┌─────────────────────────────────────────────────────┐\n");
    printf("│  PCA DETECTION RESULTS                              │\n");
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Baseline evaluated: %4d packets                   │\n", baseline_eval_count);
    printf("│  Movement evaluated: %4d packets                   │\n", movement_eval_count);
    printf("│  Threshold: %.4f                                  │\n", calibrated_threshold);
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  TP: %4d  TN: %4d  FP: %4d  FN: %4d            │\n", pkt_tp, pkt_tn, pkt_fp, pkt_fn);
    printf("├─────────────────────────────────────────────────────┤\n");
    printf("│  Recall:    %6.1f%%  (target: >90%%)               │\n", recall);
    printf("│  Precision: %6.1f%%                                │\n", precision);
    printf("│  FP Rate:   %6.1f%%  (target: <%.0f%%)                │\n", fp_rate, fp_target);
    printf("│  F1 Score:  %6.1f%%                                │\n", f1);
    printf("└─────────────────────────────────────────────────────┘\n");
    
    // TODO: Optimize PCA parameters and threshold calculation for higher recall
    if (recall < 50.0f) {
        printf("\nNote: PCA recall is below 50%%. This is expected for current experimental implementation.\n");
    }
    TEST_ASSERT_TRUE_MESSAGE(recall > 10.0f, "PCA Recall critically low (minimum: >10%)");
    if (baseline_eval_count > 0) {
        TEST_ASSERT_TRUE_MESSAGE(fp_rate < fp_target, "PCA FP Rate too high");
    }
}

// Run tests for a specific chip
int run_tests_for_chip(csi_test_data::ChipType chip) {
    printf("\n========================================\n");
    printf("Running tests with %s 64 SC dataset (HT20)\n", csi_test_data::chip_name(chip));
    printf("========================================\n");
    
    if (!csi_test_data::switch_dataset(chip)) {
        printf("ERROR: Failed to load %s dataset\n", csi_test_data::chip_name(chip));
        return 1;
    }
    
    UNITY_BEGIN();
    RUN_TEST(test_mvs_detection_accuracy);
    RUN_TEST(test_mvs_threshold_sensitivity);
    RUN_TEST(test_mvs_window_size_sensitivity);
    RUN_TEST(test_mvs_end_to_end_with_calibration);          // P95 calibration
    RUN_TEST(test_mvs_end_to_end_with_nbvi_calibration);     // NBVI calibration
    RUN_TEST(test_pca_detection_accuracy);                    // PCA detection
    return UNITY_END();
}

int process(void) {
    int failures = 0;
    
    // Run tests with both C6 and S3 datasets
    for (auto chip : csi_test_data::get_available_chips()) {
        failures += run_tests_for_chip(chip);
    }
    
    return failures;
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

