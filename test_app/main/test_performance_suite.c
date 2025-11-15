/*
 * ESPectre - Performance Suite Test
 * 
 * Comprehensive performance evaluation for Home Assistant integration.
 * Consolidates feature classification and detection approach comparison.
 * 
 * Focus: Maximize Recall (90% target) for security/presence detection
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "csi_processor.h"
#include "real_csi_data.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Include CSI data arrays
#include "real_csi_arrays.inc"

#define NUM_FEATURES 10
#define WARMUP_PACKETS 10

// Feature names for display
static const char* feature_names[] = {
    "variance", "skewness", "kurtosis", "entropy", "iqr",
    "spatial_variance", "spatial_correlation", "spatial_gradient",
    "temporal_delta_mean", "temporal_delta_variance"
};

// Helper: extract all features for testing
static const uint8_t test_all_features[NUM_FEATURES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// Classification metrics structure
typedef struct {
    int feature_idx;
    const char* name;
    float threshold;
    int true_positives;
    int true_negatives;
    int false_positives;
    int false_negatives;
    float accuracy;
    float precision;
    float recall;
    float specificity;
    float f1_score;
    float false_positive_rate;
    float false_negative_rate;
} classification_metrics_t;

// Helper: Find optimal threshold using Otsu's method
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

// Helper: Calculate classification metrics
static void calculate_metrics(classification_metrics_t *metrics, int total_baseline, int total_movement) {
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

TEST_CASE_ESP(performance_suite_comprehensive, "[performance][security]")
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   ESPECTRE PERFORMANCE SUITE - HOME ASSISTANT         ║\n");
    printf("║   Comprehensive evaluation for security/presence      ║\n");
    printf("║   Target: 90%% Recall, <10%% FP Rate                   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Allocate storage for feature values
    float **baseline_features = malloc(NUM_FEATURES * sizeof(float*));
    float **movement_features = malloc(NUM_FEATURES * sizeof(float*));
    
    if (!baseline_features || !movement_features) {
        printf("ERROR: Failed to allocate feature arrays\n");
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        baseline_features[f] = malloc(num_baseline * sizeof(float));
        movement_features[f] = malloc(num_movement * sizeof(float));
        
        if (!baseline_features[f] || !movement_features[f]) {
            printf("ERROR: Failed to allocate feature storage\n");
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }
    }
    
    printf("Extracting features from %d baseline packets...\n", num_baseline);
    
    int baseline_valid_count = 0;
    for (int p = 0; p < num_baseline; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        baseline_features[0][baseline_valid_count] = features.variance;
        baseline_features[1][baseline_valid_count] = features.skewness;
        baseline_features[2][baseline_valid_count] = features.kurtosis;
        baseline_features[3][baseline_valid_count] = features.entropy;
        baseline_features[4][baseline_valid_count] = features.iqr;
        baseline_features[5][baseline_valid_count] = features.spatial_variance;
        baseline_features[6][baseline_valid_count] = features.spatial_correlation;
        baseline_features[7][baseline_valid_count] = features.spatial_gradient;
        baseline_features[8][baseline_valid_count] = features.temporal_delta_mean;
        baseline_features[9][baseline_valid_count] = features.temporal_delta_variance;
        baseline_valid_count++;
    }
    
    printf("  Valid baseline samples: %d (skipped %d warm-up)\n", baseline_valid_count, WARMUP_PACKETS);
    
    printf("Extracting features from %d movement packets...\n", num_movement);
    
    int movement_valid_count = 0;
    for (int p = 0; p < num_movement; p++) {
        csi_features_t features;
        csi_extract_features(movement_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        movement_features[0][movement_valid_count] = features.variance;
        movement_features[1][movement_valid_count] = features.skewness;
        movement_features[2][movement_valid_count] = features.kurtosis;
        movement_features[3][movement_valid_count] = features.entropy;
        movement_features[4][movement_valid_count] = features.iqr;
        movement_features[5][movement_valid_count] = features.spatial_variance;
        movement_features[6][movement_valid_count] = features.spatial_correlation;
        movement_features[7][movement_valid_count] = features.spatial_gradient;
        movement_features[8][movement_valid_count] = features.temporal_delta_mean;
        movement_features[9][movement_valid_count] = features.temporal_delta_variance;
        movement_valid_count++;
    }
    
    printf("  Valid movement samples: %d (skipped %d warm-up)\n\n", movement_valid_count, WARMUP_PACKETS);
    
    // Allocate metrics array
    classification_metrics_t *metrics = malloc(NUM_FEATURES * sizeof(classification_metrics_t));
    if (!metrics) {
        printf("ERROR: Failed to allocate metrics array\n");
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  INDIVIDUAL FEATURE PERFORMANCE\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Evaluate each feature
    for (int f = 0; f < NUM_FEATURES; f++) {
        metrics[f].feature_idx = f;
        metrics[f].name = feature_names[f];
        metrics[f].true_positives = 0;
        metrics[f].true_negatives = 0;
        metrics[f].false_positives = 0;
        metrics[f].false_negatives = 0;
        
        metrics[f].threshold = find_optimal_threshold_otsu(
            baseline_features[f], baseline_valid_count,
            movement_features[f], movement_valid_count
        );
        
        for (int p = 0; p < baseline_valid_count; p++) {
            if (baseline_features[f][p] >= metrics[f].threshold) {
                metrics[f].false_positives++;
            } else {
                metrics[f].true_negatives++;
            }
        }
        
        for (int p = 0; p < movement_valid_count; p++) {
            if (movement_features[f][p] >= metrics[f].threshold) {
                metrics[f].true_positives++;
            } else {
                metrics[f].false_negatives++;
            }
        }
        
        calculate_metrics(&metrics[f], baseline_valid_count, movement_valid_count);
    }
    
    // Sort by recall (descending) - PRIORITY FOR SECURITY
    classification_metrics_t *sorted_by_recall = malloc(NUM_FEATURES * sizeof(classification_metrics_t));
    memcpy(sorted_by_recall, metrics, NUM_FEATURES * sizeof(classification_metrics_t));
    
    for (int i = 0; i < NUM_FEATURES - 1; i++) {
        for (int j = i + 1; j < NUM_FEATURES; j++) {
            if (sorted_by_recall[j].recall > sorted_by_recall[i].recall) {
                classification_metrics_t temp = sorted_by_recall[i];
                sorted_by_recall[i] = sorted_by_recall[j];
                sorted_by_recall[j] = temp;
            }
        }
    }
    
    printf("Rank  Feature                   Recall    FN Rate   FP Rate   F1-Score\n");
    printf("────────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < NUM_FEATURES; i++) {
        const char* status = sorted_by_recall[i].recall >= 90.0f ? "✅" : 
                            sorted_by_recall[i].recall >= 80.0f ? "⚠️ " : "❌";
        printf("%s %2d  %-22s  %6.2f%%   %6.2f%%   %6.2f%%   %6.2f%%\n",
               status, i + 1, sorted_by_recall[i].name,
               sorted_by_recall[i].recall,
               sorted_by_recall[i].false_negative_rate,
               sorted_by_recall[i].false_positive_rate,
               sorted_by_recall[i].f1_score);
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  CONFUSION MATRIX - BEST RECALL FEATURE\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    classification_metrics_t *best = &sorted_by_recall[0];
    printf("Feature: %s (Recall: %.2f%%)\n\n", best->name, best->recall);
    printf("                    Predicted\n");
    printf("                IDLE      DETECTED\n");
    printf("Actual IDLE     %-8d  %-8d  (FP Rate: %.2f%%)\n",
           best->true_negatives, best->false_positives, best->false_positive_rate);
    printf("    DETECTED    %-8d  %-8d  (FN Rate: %.2f%%)\n",
           best->false_negatives, best->true_positives, best->false_negative_rate);
    printf("\n");
    printf("Metrics:\n");
    printf("  Accuracy:    %.2f%%\n", best->accuracy);
    printf("  Precision:   %.2f%%\n", best->precision);
    printf("  Recall:      %.2f%% %s\n", best->recall, best->recall >= 90.0f ? "✅ TARGET MET" : "❌ BELOW TARGET");
    printf("  F1-Score:    %.2f%%\n", best->f1_score);
    printf("  Specificity: %.2f%%\n", best->specificity);
    printf("\n");
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TOP 5 FEATURES FOR HOME ASSISTANT SECURITY\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    for (int i = 0; i < 5 && i < NUM_FEATURES; i++) {
        printf("%d. %s\n", i + 1, sorted_by_recall[i].name);
        printf("   Recall: %.2f%% | FP Rate: %.2f%% | F1: %.2f%%\n",
               sorted_by_recall[i].recall,
               sorted_by_recall[i].false_positive_rate,
               sorted_by_recall[i].f1_score);
        printf("   Threshold: %.4f\n", sorted_by_recall[i].threshold);
        if (i < 4) printf("\n");
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  RECOMMENDATIONS FOR HOME ASSISTANT\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    if (best->recall >= 90.0f) {
        printf("✅ EXCELLENT: Best feature meets 90%% recall target!\n");
        printf("   Recommended feature: %s\n", best->name);
        printf("   Expected false alarms: ~%.1f per hour (at 15 pps)\n",
               best->false_positive_rate / 100.0f * 15.0f * 3600.0f / baseline_valid_count);
    } else {
        printf("⚠️  WARNING: No single feature meets 90%% recall target\n");
        printf("   Best recall: %.2f%% (%s)\n", best->recall, best->name);
        printf("   Recommendation: Combine multiple features\n");
        printf("   Suggested combination:\n");
        for (int i = 0; i < 3 && i < NUM_FEATURES; i++) {
            printf("     - %s (%.2f%% recall)\n", sorted_by_recall[i].name, sorted_by_recall[i].recall);
        }
    }
    
    printf("\n");
    printf("💡 Next Steps:\n");
    printf("   1. Run threshold_optimization test for fine-tuning\n");
    printf("   2. Test temporal_robustness for real-world scenarios\n");
    printf("   3. Validate with home_assistant_integration test\n");
    printf("\n");
    
    // JSON output for Python analysis
    printf("═══════════════════════════════════════════════════════\n");
    printf("  JSON OUTPUT (for Python analysis)\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("{\n");
    printf("  \"test_name\": \"performance_suite_comprehensive\",\n");
    printf("  \"baseline_samples\": %d,\n", baseline_valid_count);
    printf("  \"movement_samples\": %d,\n", movement_valid_count);
    printf("  \"best_feature\": {\n");
    printf("    \"name\": \"%s\",\n", best->name);
    printf("    \"recall\": %.4f,\n", best->recall / 100.0f);
    printf("    \"precision\": %.4f,\n", best->precision / 100.0f);
    printf("    \"f1_score\": %.4f,\n", best->f1_score / 100.0f);
    printf("    \"fp_rate\": %.4f,\n", best->false_positive_rate / 100.0f);
    printf("    \"fn_rate\": %.4f,\n", best->false_negative_rate / 100.0f);
    printf("    \"threshold\": %.4f\n", best->threshold);
    printf("  },\n");
    printf("  \"target_met\": %s\n", best->recall >= 90.0f ? "true" : "false");
    printf("}\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Verify at least one feature has reasonable recall
    TEST_ASSERT_TRUE(best->recall > 50.0f);
    
    // Cleanup
    for (int f = 0; f < NUM_FEATURES; f++) {
        free(baseline_features[f]);
        free(movement_features[f]);
    }
    free(baseline_features);
    free(movement_features);
    free(metrics);
    free(sorted_by_recall);
}
