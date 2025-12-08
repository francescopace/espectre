/*
 * ESPectre - Motion Detection Integration Tests
 * 
 * Integration tests for the Moving Variance Segmentation (MVS) algorithm.
 * Tests motion detection performance with real CSI data.
 * 
 * Focus: Maximize Recall (90% target) for security/presence detection
 * 
 * Test Categories:
 *   1. test_mvs_detection_accuracy - Full performance evaluation with confusion matrix
 *   2. test_mvs_threshold_sensitivity - Threshold parameter sweep
 *   3. test_mvs_window_size_sensitivity - Window size parameter sweep
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <unity.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>

// Include headers from lib/espectre
#include "csi_processor.h"
#include "esphome/core/log.h"
#include "esp_system.h"

using namespace esphome::espectre;

// Include CSI data
#include "real_csi_data_esp32.h"
#include "real_csi_arrays.inc"

static const char *TAG = "test_motion_detection";

// Default subcarrier selection for all tests (optimized based on PCA analysis)
static const uint8_t SELECTED_SUBCARRIERS[] = {11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22};
static const uint8_t NUM_SUBCARRIERS = sizeof(SELECTED_SUBCARRIERS) / sizeof(SELECTED_SUBCARRIERS[0]);

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

// Helper function to process a packet
static void process_packet(csi_processor_context_t *ctx, const int8_t *packet) {
    csi_process_packet(ctx, packet, 128, SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
}

// Test: MVS motion detection accuracy with real CSI data
void test_mvs_detection_accuracy(void) {
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   ESPECTRE PERFORMANCE SUITE - MOTION DETECTION       ║\n");
    printf("║   Comprehensive evaluation for security/presence      ║\n");
    printf("║   Target: 90%% Recall, <10%% FP Rate                   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("Architecture: CSI Packet → Motion Detection (MVS) → State\n");
    printf("Accuracy based on: Motion detection performance\n");
    printf("\n");
    
    // ========================================================================
    // MOTION DETECTION PERFORMANCE
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  MOTION DETECTION PERFORMANCE\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    csi_processor_context_t ctx;
    TEST_ASSERT_TRUE(csi_processor_init(&ctx, SEGMENTATION_DEFAULT_WINDOW_SIZE, SEGMENTATION_DEFAULT_THRESHOLD));
    
    float threshold = csi_processor_get_threshold(&ctx);
    printf("Using default threshold: %.4f\n", threshold);
    printf("Window size: %d\n", csi_processor_get_window_size(&ctx));
    
    // Test on baseline (should have minimal false positives)
    printf("Testing on baseline packets (expecting no motion)...\n");
    
    int baseline_segments_completed = 0;
    int baseline_motion_packets = 0;
    
    csi_motion_state_t prev_state = CSI_STATE_IDLE;
    
    for (int p = 0; p < num_baseline; p++) {
        process_packet(&ctx, (const int8_t*)baseline_packets[p]);
        
        csi_motion_state_t current_state = csi_processor_get_state(&ctx);
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == CSI_STATE_MOTION && current_state == CSI_STATE_IDLE) {
            baseline_segments_completed++;
        }
        
        // Also track packets in motion state (for info)
        if (current_state == CSI_STATE_MOTION) {
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
    
    prev_state = CSI_STATE_IDLE;
    
    for (int p = 0; p < num_movement; p++) {
        process_packet(&ctx, (const int8_t*)movement_packets[p]);
        
        csi_motion_state_t current_state = csi_processor_get_state(&ctx);
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == CSI_STATE_MOTION && current_state == CSI_STATE_IDLE) {
            total_segments_detected++;
        }
        
        // Check if currently in motion state
        if (current_state == CSI_STATE_MOTION) {
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
    printf("  * FP Rate:    %.1f%% (target: <10%%)\n", pkt_fp_rate);
    printf("  * F1-Score:   %.1f%%\n", pkt_f1);
    printf("\n");
    printf("MEMORY:\n");
    printf("  * Free heap: %d bytes\n", (int)free_heap);
    printf("\n");
    printf("═══════════════════════════════════════════════════════════════════════\n\n");
    
    // Cleanup
    csi_processor_cleanup(&ctx);
    
    // Verify minimum acceptable performance based on motion packets
    TEST_ASSERT_GREATER_THAN(950, movement_with_motion);  // >95% recall
    TEST_ASSERT_LESS_THAN(1.0f, pkt_fp_rate);             // <1% FP rate
}

// Test: MVS threshold parameter sensitivity analysis
void test_mvs_threshold_sensitivity(void) {
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  THRESHOLD SENSITIVITY ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    float thresholds[] = {0.5f, 0.75f, 1.0f, 1.5f, 2.0f, 3.0f};
    int num_thresholds = sizeof(thresholds) / sizeof(thresholds[0]);
    
    printf("Threshold   Recall    FP Rate   F1-Score\n");
    printf("──────────────────────────────────────────\n");
    
    for (int t = 0; t < num_thresholds; t++) {
        csi_processor_context_t ctx;
        TEST_ASSERT_TRUE(csi_processor_init(&ctx, SEGMENTATION_DEFAULT_WINDOW_SIZE, thresholds[t]));
        
        int baseline_motion = 0;
        int movement_motion = 0;
        
        // Process baseline
        for (int p = 0; p < num_baseline; p++) {
            process_packet(&ctx, (const int8_t*)baseline_packets[p]);
            if (csi_processor_get_state(&ctx) == CSI_STATE_MOTION) {
                baseline_motion++;
            }
        }
        
        // Process movement
        for (int p = 0; p < num_movement; p++) {
            process_packet(&ctx, (const int8_t*)movement_packets[p]);
            if (csi_processor_get_state(&ctx) == CSI_STATE_MOTION) {
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
        
        csi_processor_cleanup(&ctx);
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
    
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    uint16_t window_sizes[] = {20, 30, 50, 75, 100, 150};
    int num_sizes = sizeof(window_sizes) / sizeof(window_sizes[0]);
    
    printf("Window Size   Recall    FP Rate   F1-Score\n");
    printf("────────────────────────────────────────────\n");
    
    for (int w = 0; w < num_sizes; w++) {
        csi_processor_context_t ctx;
        TEST_ASSERT_TRUE(csi_processor_init(&ctx, window_sizes[w], SEGMENTATION_DEFAULT_THRESHOLD));
        
        int baseline_motion = 0;
        int movement_motion = 0;
        
        // Process baseline
        for (int p = 0; p < num_baseline; p++) {
            process_packet(&ctx, (const int8_t*)baseline_packets[p]);
            if (csi_processor_get_state(&ctx) == CSI_STATE_MOTION) {
                baseline_motion++;
            }
        }
        
        // Process movement
        for (int p = 0; p < num_movement; p++) {
            process_packet(&ctx, (const int8_t*)movement_packets[p]);
            if (csi_processor_get_state(&ctx) == CSI_STATE_MOTION) {
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
        
        csi_processor_cleanup(&ctx);
    }
    
    printf("\n");
    
    // Just verify the test ran successfully
    TEST_ASSERT_TRUE(true);
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_mvs_detection_accuracy);
    RUN_TEST(test_mvs_threshold_sensitivity);
    RUN_TEST(test_mvs_window_size_sensitivity);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif

