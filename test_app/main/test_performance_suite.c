/*
 * ESPectre - Performance Suite Test
 * 
 * Comprehensive performance evaluation based on segmentation.
 * Tests the Moving Variance Segmentation (MVS) algorithm for motion detection.
 * 
 * Focus: Maximize Recall (90% target) for security/presence detection
 * 
 * New Architecture:
 *   CSI Packet → Segmentation (always) → IF MOTION && features_enabled:
 *                                           → Extract Features + Publish
 *                                        ELSE:
 *                                           → Publish without features
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "csi_processor.h"
#include "segmentation.h"
#include "real_csi_data_esp32.h"
#include "espectre.h"
#include "esp_system.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Include CSI data arrays
#include "real_csi_arrays.inc"

// Default subcarrier selection for all tests (optimized based on PCA analysis)
static const uint8_t SELECTED_SUBCARRIERS[] = DEFAULT_SUBCARRIERS;
static const uint8_t NUM_SUBCARRIERS = sizeof(SELECTED_SUBCARRIERS) / sizeof(SELECTED_SUBCARRIERS[0]);

#define NUM_FEATURES 10

// Feature names for display (secondary - for feature ranking)
static const char* feature_names[] = {
    "variance", "skewness", "kurtosis", "entropy", "iqr",
    "spatial_variance", "spatial_correlation", "spatial_gradient",
    "temporal_delta_mean", "temporal_delta_variance"
};

// Helper: extract all features for ranking
static const uint8_t test_all_features[NUM_FEATURES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// Segmentation metrics structure
typedef struct {
    int true_positives;      // Movement packets with segments detected
    int true_negatives;      // Baseline packets without segments
    int false_positives;     // Baseline packets with false segments
    int false_negatives;     // Movement packets without segments
    float accuracy;
    float precision;
    float recall;
    float specificity;
    float f1_score;
    float false_positive_rate;
    float false_negative_rate;
} segmentation_metrics_t;

// Feature ranking structure (secondary)
typedef struct {
    int feature_idx;
    const char* name;
    float threshold;
    float recall;
    float fp_rate;
    float f1_score;
} feature_ranking_t;

// Helper: Calculate segmentation metrics
static void calculate_segmentation_metrics(segmentation_metrics_t *metrics, 
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

// Helper: Find optimal threshold using Otsu's method (for feature ranking)
static float find_optimal_threshold_otsu(const float *baseline_values, int baseline_count,
                                         const float *movement_values, int movement_count) {
    int total_count = baseline_count + movement_count;
    float *all_values = malloc(total_count * sizeof(float));
    if (!all_values) return 0.0f;
    
    memcpy(all_values, baseline_values, baseline_count * sizeof(float));
    memcpy(all_values + baseline_count, movement_values, movement_count * sizeof(float));
    
    float min_val = all_values[0];
    float max_val = all_values[0];
    for (int i = 1; i < total_count; i++) {
        if (all_values[i] < min_val) min_val = all_values[i];
        if (all_values[i] > max_val) max_val = all_values[i];
    }
    
    float best_threshold = (min_val + max_val) / 2.0f;
    float best_variance = 0.0f;
    int num_steps = 100;
    float step = (max_val - min_val) / num_steps;
    
    for (int i = 1; i < num_steps; i++) {
        float threshold = min_val + i * step;
        
        int class0_count = 0;
        int class1_count = 0;
        float class0_sum = 0.0f;
        float class1_sum = 0.0f;
        
        for (int j = 0; j < total_count; j++) {
            if (all_values[j] < threshold) {
                class0_count++;
                class0_sum += all_values[j];
            } else {
                class1_count++;
                class1_sum += all_values[j];
            }
        }
        
        if (class0_count == 0 || class1_count == 0) continue;
        
        float class0_mean = class0_sum / class0_count;
        float class1_mean = class1_sum / class1_count;
        
        float w0 = (float)class0_count / total_count;
        float w1 = (float)class1_count / total_count;
        float between_variance = w0 * w1 * (class0_mean - class1_mean) * (class0_mean - class1_mean);
        
        if (between_variance > best_variance) {
            best_variance = between_variance;
            best_threshold = threshold;
        }
    }
    
    free(all_values);
    return best_threshold;
}

TEST_CASE_ESP(performance_suite_comprehensive, "[performance][security]")
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   ESPECTRE PERFORMANCE SUITE - SEGMENTATION-BASED     ║\n");
    printf("║   Comprehensive evaluation for security/presence      ║\n");
    printf("║   Target: 90%% Recall, <10%% FP Rate                   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("New Architecture: Segmentation → [IF MOTION] → Features\n");
    printf("Accuracy based on: Segmentation performance\n");
    printf("\n");
    
    // ========================================================================
    // PART 1: SEGMENTATION PERFORMANCE (PRIMARY)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  PART 1: SEGMENTATION PERFORMANCE (PRIMARY)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Set global subcarrier selection for CSI processing
    csi_set_subcarrier_selection(SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
    
    segmentation_context_t seg_ctx;
    segmentation_init(&seg_ctx);
    
    float threshold = segmentation_get_threshold(&seg_ctx);
    printf("Using default threshold: %.4f\n", threshold);
    printf("Window size: %d\n", segmentation_get_window_size(&seg_ctx));
    
    // Test on baseline (should have minimal false positives)
    printf("Testing on baseline packets (expecting no segments)...\n");
    
    int baseline_segments_completed = 0;
    int baseline_motion_packets = 0;
    
    segmentation_state_t prev_state = SEG_STATE_IDLE;
    
    for (int p = 0; p < num_baseline; p++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)baseline_packets[p], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        segmentation_add_turbulence(&seg_ctx, turbulence);
        
        segmentation_state_t current_state = segmentation_get_state(&seg_ctx);
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == SEG_STATE_MOTION && current_state == SEG_STATE_IDLE) {
            baseline_segments_completed++;
        }
        
        // Also track packets in motion state (for info)
        if (current_state == SEG_STATE_MOTION) {
            baseline_motion_packets++;
        }
        
        prev_state = current_state;
    }
    
    printf("  Baseline packets: %d\n", num_baseline);
    printf("  Motion packets: %d (%.1f%%)\n", baseline_motion_packets, 
           (float)baseline_motion_packets / num_baseline * 100.0f);
    printf("  Segments completed (FP): %d\n", baseline_segments_completed);
    printf("  FP Rate: %.2f%%\n\n", (float)baseline_segments_completed / num_baseline * 100.0f);
    
    // Test on movement (should detect segments)
    printf("Testing on movement packets (expecting segments)...\n");
    
    int movement_with_segments = 0;
    int movement_without_segments = 0;
    int total_segments_detected = 0;
    
    prev_state = SEG_STATE_IDLE;
    
    for (int p = 0; p < num_movement; p++) {
        float turbulence = csi_calculate_spatial_turbulence(
            (const int8_t*)movement_packets[p], 128,
            SELECTED_SUBCARRIERS, NUM_SUBCARRIERS);
        
        segmentation_add_turbulence(&seg_ctx, turbulence);
        
        segmentation_state_t current_state = segmentation_get_state(&seg_ctx);
        
        // Count transitions from MOTION to IDLE as completed segments
        if (prev_state == SEG_STATE_MOTION && current_state == SEG_STATE_IDLE) {
            total_segments_detected++;
        }
        
        // Check if currently in motion state
        if (current_state == SEG_STATE_MOTION) {
            movement_with_segments++;
        } else {
            movement_without_segments++;
        }
        
        prev_state = current_state;
    }
    
    printf("  Movement packets: %d\n", num_movement);
    printf("  With segments: %d\n", movement_with_segments);
    printf("  Without segments (FN): %d\n", movement_without_segments);
    printf("  Detection Rate: %.2f%%\n", (float)movement_with_segments / num_movement * 100.0f);
    printf("  Total segments detected: %d\n\n", total_segments_detected);
    
    // Calculate metrics based on segments completed (not packets in motion)
    segmentation_metrics_t metrics;
    metrics.true_positives = total_segments_detected;  // Segments detected in movement
    metrics.true_negatives = num_baseline - baseline_segments_completed;  // Baseline without segments
    metrics.false_positives = baseline_segments_completed;  // False segments in baseline
    metrics.false_negatives = 0;  // Assume all movement should have segments (simplified)
    
    calculate_segmentation_metrics(&metrics, num_baseline, num_movement);
    
    
    // ========================================================================
    // PART 2: FEATURE RANKING (SECONDARY - for features_enabled mode)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  PART 2: FEATURE RANKING (SECONDARY)\n");
    printf("  Note: Features extracted only when segmentation\n");
    printf("        detects motion (features_enabled=true)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Allocate storage for feature values
    float **baseline_features = malloc(NUM_FEATURES * sizeof(float*));
    float **movement_features = malloc(NUM_FEATURES * sizeof(float*));
    
    if (!baseline_features || !movement_features) {
        printf("⚠️  Skipping feature ranking (memory allocation failed)\n\n");
        goto test_summary;
    }
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        baseline_features[f] = malloc(num_baseline * sizeof(float));
        movement_features[f] = malloc(num_movement * sizeof(float));
        
        if (!baseline_features[f] || !movement_features[f]) {
            printf("⚠️  Skipping feature ranking (memory allocation failed)\n\n");
            goto feature_cleanup;
        }
    }
    
    printf("Extracting features for ranking...\n");
    
    // Extract baseline features
    for (int p = 0; p < num_baseline; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        baseline_features[0][p] = features.variance;
        baseline_features[1][p] = features.skewness;
        baseline_features[2][p] = features.kurtosis;
        baseline_features[3][p] = features.entropy;
        baseline_features[4][p] = features.iqr;
        baseline_features[5][p] = features.spatial_variance;
        baseline_features[6][p] = features.spatial_correlation;
        baseline_features[7][p] = features.spatial_gradient;
        baseline_features[8][p] = features.temporal_delta_mean;
        baseline_features[9][p] = features.temporal_delta_variance;
    }
    
    // Extract movement features
    for (int p = 0; p < num_movement; p++) {
        csi_features_t features;
        csi_extract_features(movement_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        movement_features[0][p] = features.variance;
        movement_features[1][p] = features.skewness;
        movement_features[2][p] = features.kurtosis;
        movement_features[3][p] = features.entropy;
        movement_features[4][p] = features.iqr;
        movement_features[5][p] = features.spatial_variance;
        movement_features[6][p] = features.spatial_correlation;
        movement_features[7][p] = features.spatial_gradient;
        movement_features[8][p] = features.temporal_delta_mean;
        movement_features[9][p] = features.temporal_delta_variance;
    }
    
    // Rank features
    feature_ranking_t *rankings = malloc(NUM_FEATURES * sizeof(feature_ranking_t));
    if (!rankings) {
        printf("⚠️  Skipping feature ranking (memory allocation failed)\n\n");
        goto feature_cleanup;
    }
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        rankings[f].feature_idx = f;
        rankings[f].name = feature_names[f];
        rankings[f].threshold = find_optimal_threshold_otsu(
            baseline_features[f], num_baseline,
            movement_features[f], num_movement
        );
        
        int tp = 0, fp = 0;
        for (int p = 0; p < num_baseline; p++) {
            if (baseline_features[f][p] >= rankings[f].threshold) fp++;
        }
        for (int p = 0; p < num_movement; p++) {
            if (movement_features[f][p] >= rankings[f].threshold) tp++;
        }
        
        rankings[f].recall = (float)tp / num_movement * 100.0f;
        rankings[f].fp_rate = (float)fp / num_baseline * 100.0f;
        
        float prec = (tp + fp > 0) ? (float)tp / (tp + fp) : 0.0f;
        float rec = rankings[f].recall / 100.0f;
        rankings[f].f1_score = (prec + rec > 0) ? 2.0f * prec * rec / (prec + rec) * 100.0f : 0.0f;
    }
    
    // Sort by recall
    for (int i = 0; i < NUM_FEATURES - 1; i++) {
        for (int j = i + 1; j < NUM_FEATURES; j++) {
            if (rankings[j].recall > rankings[i].recall) {
                feature_ranking_t temp = rankings[i];
                rankings[i] = rankings[j];
                rankings[j] = temp;
            }
        }
    }
    
    printf("\nTop 5 Features (for features_enabled mode):\n");
    printf("Rank  Feature                   Recall    FP Rate   F1-Score\n");
    printf("────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < 5 && i < NUM_FEATURES; i++) {
        printf("%2d    %-22s  %6.2f%%   %6.2f%%   %6.2f%%\n",
               i + 1, rankings[i].name,
               rankings[i].recall,
               rankings[i].fp_rate,
               rankings[i].f1_score);
    }
    printf("\n");
    
    free(rankings);
    
feature_cleanup:
    for (int f = 0; f < NUM_FEATURES; f++) {
        free(baseline_features[f]);
        free(movement_features[f]);
    }
    free(baseline_features);
    free(movement_features);
    
test_summary:
    // ========================================================================
    // FINAL TEST SUMMARY (printed before assertions so always visible)
    // ========================================================================
    
    // Calculate packet-based metrics
    int pkt_tp = movement_with_segments;
    int pkt_fn = movement_without_segments;
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
    printf("SEGMENTATION METRICS:\n");
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
    
    // Verify minimum acceptable performance based on motion packets
    TEST_ASSERT_GREATER_THAN(950, movement_with_segments);  // >95% recall
    TEST_ASSERT_LESS_THAN(1.0f, pkt_fp_rate);               // <1% FP rate
}
