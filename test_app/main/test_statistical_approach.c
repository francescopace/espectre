/*
 * ESPectre - Statistical Approach 3-Way Comparison Test
 * 
 * Compares 3 statistical approaches with real CSI data:
 * 1. Variance-based (moving variance of amplitude)
 * 2. Skewness-based (moving skewness of amplitude)
 * 3. Abs-Skewness-based (moving |skewness| of amplitude)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "real_csi_data.h"
#include <math.h>
#include <string.h>
#include <stdio.h>

// ============================================================================
// CONFIGURATION
// ============================================================================

#define MOVING_AVG_WINDOW 5      // Moving average filter window
#define METRIC_WINDOW 20         // Window for variance/skewness calculation
#define SENSITIVITY_K 1.5f       // Threshold multiplier (reduced from 2.0)

// ============================================================================
// DATA STRUCTURES
// ============================================================================

typedef struct {
    float buffer[MOVING_AVG_WINDOW];
    int index;
    int count;
} moving_avg_t;

typedef struct {
    float buffer[METRIC_WINDOW];
    int index;
    int count;
} metric_buffer_t;

typedef struct {
    float sum;
    float sum_sq;
    int count;
} stats_t;

typedef struct {
    float baseline_mean;
    float movement_mean;
    float separation;
    float threshold;
    float accuracy;
    float false_positive_rate;
    float false_negative_rate;
} approach_results_t;

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static void moving_avg_init(moving_avg_t *ma) {
    memset(ma, 0, sizeof(moving_avg_t));
}

static float moving_avg_add(moving_avg_t *ma, float value) {
    ma->buffer[ma->index] = value;
    ma->index = (ma->index + 1) % MOVING_AVG_WINDOW;
    if (ma->count < MOVING_AVG_WINDOW) ma->count++;
    
    float sum = 0.0f;
    for (int i = 0; i < ma->count; i++) {
        sum += ma->buffer[i];
    }
    return sum / ma->count;
}

static void metric_buffer_init(metric_buffer_t *mb) {
    memset(mb, 0, sizeof(metric_buffer_t));
}

static void stats_init(stats_t *stats) {
    memset(stats, 0, sizeof(stats_t));
}

static void stats_add(stats_t *stats, float value) {
    stats->sum += value;
    stats->sum_sq += value * value;
    stats->count++;
}

static float stats_mean(const stats_t *stats) {
    return (stats->count == 0) ? 0.0f : stats->sum / stats->count;
}

static float stats_stddev(const stats_t *stats) {
    if (stats->count < 2) return 0.0f;
    float mean = stats_mean(stats);
    float variance = (stats->sum_sq / stats->count) - (mean * mean);
    return sqrtf(fmaxf(0.0f, variance));
}

static float calculate_avg_amplitude(const int8_t *csi_data, size_t len) {
    if (len < 2) return 0.0f;
    
    float sum = 0.0f;
    int num_subcarriers = len / 2;
    
    for (int i = 0; i < num_subcarriers; i++) {
        float I = (float)csi_data[2 * i];
        float Q = (float)csi_data[2 * i + 1];
        sum += sqrtf(I * I + Q * Q);
    }
    
    return sum / num_subcarriers;
}

// ============================================================================
// METRIC CALCULATION FUNCTIONS
// ============================================================================

static float calculate_variance(metric_buffer_t *mb, float value) {
    mb->buffer[mb->index] = value;
    mb->index = (mb->index + 1) % METRIC_WINDOW;
    if (mb->count < METRIC_WINDOW) mb->count++;
    
    if (mb->count < 2) return 0.0f;
    
    float mean = 0.0f;
    for (int i = 0; i < mb->count; i++) {
        mean += mb->buffer[i];
    }
    mean /= mb->count;
    
    float variance = 0.0f;
    for (int i = 0; i < mb->count; i++) {
        float diff = mb->buffer[i] - mean;
        variance += diff * diff;
    }
    return variance / mb->count;
}

static float calculate_skewness(metric_buffer_t *mb, float value) {
    mb->buffer[mb->index] = value;
    mb->index = (mb->index + 1) % METRIC_WINDOW;
    if (mb->count < METRIC_WINDOW) mb->count++;
    
    if (mb->count < 3) return 0.0f;
    
    float mean = 0.0f;
    for (int i = 0; i < mb->count; i++) {
        mean += mb->buffer[i];
    }
    mean /= mb->count;
    
    float m2 = 0.0f, m3 = 0.0f;
    for (int i = 0; i < mb->count; i++) {
        float diff = mb->buffer[i] - mean;
        float diff2 = diff * diff;
        m2 += diff2;
        m3 += diff2 * diff;
    }
    
    m2 /= mb->count;
    m3 /= mb->count;
    
    float stddev = sqrtf(m2);
    if (stddev < 1e-6f) return 0.0f;
    
    return m3 / (stddev * stddev * stddev);
}

// ============================================================================
// APPROACH TESTING FUNCTIONS
// ============================================================================

static approach_results_t test_approach_variance(const int8_t **baseline_pkts, 
                                                  const int8_t **movement_pkts) {
    approach_results_t results = {0};
    
    // Process baseline
    moving_avg_t ma_b;
    metric_buffer_t mb_b;
    stats_t stats_b;
    moving_avg_init(&ma_b);
    metric_buffer_init(&mb_b);
    stats_init(&stats_b);
    
    float baseline_values[50];
    int baseline_count = 0;
    
    for (int p = 0; p < 50; p++) {
        float amp = calculate_avg_amplitude(baseline_pkts[p], 128);
        float filtered = moving_avg_add(&ma_b, amp);
        float metric = calculate_variance(&mb_b, filtered);
        
        if (mb_b.count >= METRIC_WINDOW) {
            baseline_values[baseline_count++] = metric;
            stats_add(&stats_b, metric);
        }
    }
    
    results.baseline_mean = stats_mean(&stats_b);
    float baseline_std = stats_stddev(&stats_b);
    
    // Process movement
    moving_avg_t ma_m;
    metric_buffer_t mb_m;
    stats_t stats_m;
    moving_avg_init(&ma_m);
    metric_buffer_init(&mb_m);
    stats_init(&stats_m);
    
    float movement_values[50];
    int movement_count = 0;
    
    for (int p = 0; p < 50; p++) {
        float amp = calculate_avg_amplitude(movement_pkts[p], 128);
        float filtered = moving_avg_add(&ma_m, amp);
        float metric = calculate_variance(&mb_m, filtered);
        
        if (mb_m.count >= METRIC_WINDOW) {
            movement_values[movement_count++] = metric;
            stats_add(&stats_m, metric);
        }
    }
    
    results.movement_mean = stats_mean(&stats_m);
    
    // Calculate threshold and test
    results.threshold = results.baseline_mean + (SENSITIVITY_K * baseline_std);
    results.separation = (results.baseline_mean > 1e-6f) ? 
                        results.movement_mean / results.baseline_mean : 0.0f;
    
    int fp = 0, fn = 0;
    for (int i = 0; i < baseline_count; i++) {
        if (baseline_values[i] > results.threshold) fp++;
    }
    for (int i = 0; i < movement_count; i++) {
        if (movement_values[i] <= results.threshold) fn++;
    }
    
    int total = baseline_count + movement_count;
    int correct = (baseline_count - fp) + (movement_count - fn);
    results.accuracy = (float)correct / total * 100.0f;
    results.false_positive_rate = (float)fp / baseline_count * 100.0f;
    results.false_negative_rate = (float)fn / movement_count * 100.0f;
    
    return results;
}

static approach_results_t test_approach_skewness(const int8_t **baseline_pkts,
                                                  const int8_t **movement_pkts) {
    approach_results_t results = {0};
    
    // Process baseline
    moving_avg_t ma_b;
    metric_buffer_t mb_b;
    stats_t stats_b;
    moving_avg_init(&ma_b);
    metric_buffer_init(&mb_b);
    stats_init(&stats_b);
    
    float baseline_values[50];
    int baseline_count = 0;
    
    for (int p = 0; p < 50; p++) {
        float amp = calculate_avg_amplitude(baseline_pkts[p], 128);
        float filtered = moving_avg_add(&ma_b, amp);
        float metric = calculate_skewness(&mb_b, filtered);
        
        if (mb_b.count >= METRIC_WINDOW) {
            baseline_values[baseline_count++] = metric;
            stats_add(&stats_b, metric);
        }
    }
    
    results.baseline_mean = stats_mean(&stats_b);
    float baseline_std = stats_stddev(&stats_b);
    
    // Process movement
    moving_avg_t ma_m;
    metric_buffer_t mb_m;
    stats_t stats_m;
    moving_avg_init(&ma_m);
    metric_buffer_init(&mb_m);
    stats_init(&stats_m);
    
    float movement_values[50];
    int movement_count = 0;
    
    for (int p = 0; p < 50; p++) {
        float amp = calculate_avg_amplitude(movement_pkts[p], 128);
        float filtered = moving_avg_add(&ma_m, amp);
        float metric = calculate_skewness(&mb_m, filtered);
        
        if (mb_m.count >= METRIC_WINDOW) {
            movement_values[movement_count++] = metric;
            stats_add(&stats_m, metric);
        }
    }
    
    results.movement_mean = stats_mean(&stats_m);
    
    // Threshold: baseline - k×σ (skewness DECREASES with movement)
    results.threshold = results.baseline_mean - (SENSITIVITY_K * baseline_std);
    results.separation = fabsf(results.baseline_mean - results.movement_mean);
    
    int fp = 0, fn = 0;
    // Baseline should be ABOVE threshold
    for (int i = 0; i < baseline_count; i++) {
        if (baseline_values[i] < results.threshold) fp++;
    }
    // Movement should be BELOW threshold
    for (int i = 0; i < movement_count; i++) {
        if (movement_values[i] >= results.threshold) fn++;
    }
    
    int total = baseline_count + movement_count;
    int correct = (baseline_count - fp) + (movement_count - fn);
    results.accuracy = (float)correct / total * 100.0f;
    results.false_positive_rate = (float)fp / baseline_count * 100.0f;
    results.false_negative_rate = (float)fn / movement_count * 100.0f;
    
    return results;
}

static approach_results_t test_approach_abs_skewness(const int8_t **baseline_pkts,
                                                      const int8_t **movement_pkts) {
    approach_results_t results = {0};
    
    // Process baseline
    moving_avg_t ma_b;
    metric_buffer_t mb_b;
    stats_t stats_b;
    moving_avg_init(&ma_b);
    metric_buffer_init(&mb_b);
    stats_init(&stats_b);
    
    float baseline_values[50];
    int baseline_count = 0;
    
    for (int p = 0; p < 50; p++) {
        float amp = calculate_avg_amplitude(baseline_pkts[p], 128);
        float filtered = moving_avg_add(&ma_b, amp);
        float skewness = calculate_skewness(&mb_b, filtered);
        float metric = fabsf(skewness);  // Take absolute value!
        
        if (mb_b.count >= METRIC_WINDOW) {
            baseline_values[baseline_count++] = metric;
            stats_add(&stats_b, metric);
        }
    }
    
    results.baseline_mean = stats_mean(&stats_b);
    float baseline_std = stats_stddev(&stats_b);
    
    // Process movement
    moving_avg_t ma_m;
    metric_buffer_t mb_m;
    stats_t stats_m;
    moving_avg_init(&ma_m);
    metric_buffer_init(&mb_m);
    stats_init(&stats_m);
    
    float movement_values[50];
    int movement_count = 0;
    
    for (int p = 0; p < 50; p++) {
        float amp = calculate_avg_amplitude(movement_pkts[p], 128);
        float filtered = moving_avg_add(&ma_m, amp);
        float skewness = calculate_skewness(&mb_m, filtered);
        float metric = fabsf(skewness);  // Take absolute value!
        
        if (mb_m.count >= METRIC_WINDOW) {
            movement_values[movement_count++] = metric;
            stats_add(&stats_m, metric);
        }
    }
    
    results.movement_mean = stats_mean(&stats_m);
    
    // Threshold: baseline + k×σ (|skewness| INCREASES with movement)
    results.threshold = results.baseline_mean + (SENSITIVITY_K * baseline_std);
    results.separation = (results.baseline_mean > 1e-6f) ?
                        results.movement_mean / results.baseline_mean : 0.0f;
    
    int fp = 0, fn = 0;
    // Baseline should be BELOW threshold
    for (int i = 0; i < baseline_count; i++) {
        if (baseline_values[i] > results.threshold) fp++;
    }
    // Movement should be ABOVE threshold
    for (int i = 0; i < movement_count; i++) {
        if (movement_values[i] <= results.threshold) fn++;
    }
    
    int total = baseline_count + movement_count;
    int correct = (baseline_count - fp) + (movement_count - fn);
    results.accuracy = (float)correct / total * 100.0f;
    results.false_positive_rate = (float)fp / baseline_count * 100.0f;
    results.false_negative_rate = (float)fn / movement_count * 100.0f;
    
    return results;
}

// ============================================================================
// MAIN TEST
// ============================================================================

TEST_CASE_ESP(statistical_approach_with_real_csi_data, "[statistical][real]")
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════╗\n");
    printf("║      3-WAY STATISTICAL APPROACH COMPARISON TEST          ║\n");
    printf("╚═══════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Packet arrays
    const int8_t *baseline_packets[] = {
        real_baseline_0, real_baseline_1, real_baseline_2, real_baseline_3,
        real_baseline_4, real_baseline_5, real_baseline_6, real_baseline_7,
        real_baseline_8, real_baseline_9, real_baseline_10, real_baseline_11,
        real_baseline_12, real_baseline_13, real_baseline_14, real_baseline_15,
        real_baseline_16, real_baseline_17, real_baseline_18, real_baseline_19,
        real_baseline_20, real_baseline_21, real_baseline_22, real_baseline_23,
        real_baseline_24, real_baseline_25, real_baseline_26, real_baseline_27,
        real_baseline_28, real_baseline_29, real_baseline_30, real_baseline_31,
        real_baseline_32, real_baseline_33, real_baseline_34, real_baseline_35,
        real_baseline_36, real_baseline_37, real_baseline_38, real_baseline_39,
        real_baseline_40, real_baseline_41, real_baseline_42, real_baseline_43,
        real_baseline_44, real_baseline_45, real_baseline_46, real_baseline_47,
        real_baseline_48, real_baseline_49
    };
    
    const int8_t *movement_packets[] = {
        real_movement_0, real_movement_1, real_movement_2, real_movement_3,
        real_movement_4, real_movement_5, real_movement_6, real_movement_7,
        real_movement_8, real_movement_9, real_movement_10, real_movement_11,
        real_movement_12, real_movement_13, real_movement_14, real_movement_15,
        real_movement_16, real_movement_17, real_movement_18, real_movement_19,
        real_movement_20, real_movement_21, real_movement_22, real_movement_23,
        real_movement_24, real_movement_25, real_movement_26, real_movement_27,
        real_movement_28, real_movement_29, real_movement_30, real_movement_31,
        real_movement_32, real_movement_33, real_movement_34, real_movement_35,
        real_movement_36, real_movement_37, real_movement_38, real_movement_39,
        real_movement_40, real_movement_41, real_movement_42, real_movement_43,
        real_movement_44, real_movement_45, real_movement_46, real_movement_47,
        real_movement_48, real_movement_49
    };
    
    // ========================================================================
    // TEST ALL 3 APPROACHES
    // ========================================================================
    
    printf("🔬 Testing APPROACH 1: Variance-based...\n");
    approach_results_t var_results = test_approach_variance(baseline_packets, movement_packets);
    
    printf("🔬 Testing APPROACH 2: Skewness-based...\n");
    approach_results_t skew_results = test_approach_skewness(baseline_packets, movement_packets);
    
    printf("🔬 Testing APPROACH 3: Abs-Skewness-based...\n");
    approach_results_t abs_skew_results = test_approach_abs_skewness(baseline_packets, movement_packets);
    
    // ========================================================================
    // COMPARISON TABLE
    // ========================================================================
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════════════╗\n");
    printf("║                    3-WAY STATISTICAL COMPARISON                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    printf("║                                                                           ║\n");
    printf("║  📊 APPROACH 1: VARIANCE-BASED                                            ║\n");
    printf("║  ────────────────────────────────────────────────────────────────────     ║\n");
    printf("║  Baseline mean:       %.6f                                            ║\n", var_results.baseline_mean);
    printf("║  Movement mean:       %.6f                                            ║\n", var_results.movement_mean);
    printf("║  Separation:          %.2fx                                               ║\n", var_results.separation);
    printf("║  Threshold:           %.6f                                            ║\n", var_results.threshold);
    printf("║  Accuracy:            %.1f%%                                              ║\n", var_results.accuracy);
    printf("║  False positives:     %.1f%%                                              ║\n", var_results.false_positive_rate);
    printf("║  False negatives:     %.1f%%                                              ║\n", var_results.false_negative_rate);
    printf("║                                                                           ║\n");
    printf("║  📊 APPROACH 2: SKEWNESS-BASED                                            ║\n");
    printf("║  ────────────────────────────────────────────────────────────────────     ║\n");
    printf("║  Baseline mean:       %.6f                                            ║\n", skew_results.baseline_mean);
    printf("║  Movement mean:       %.6f                                            ║\n", skew_results.movement_mean);
    printf("║  Separation:          %.3f (absolute diff)                                ║\n", skew_results.separation);
    printf("║  Threshold:           %.6f                                            ║\n", skew_results.threshold);
    printf("║  Accuracy:            %.1f%%                                              ║\n", skew_results.accuracy);
    printf("║  False positives:     %.1f%%                                              ║\n", skew_results.false_positive_rate);
    printf("║  False negatives:     %.1f%%                                              ║\n", skew_results.false_negative_rate);
    printf("║                                                                           ║\n");
    printf("║  📊 APPROACH 3: ABS-SKEWNESS-BASED                                        ║\n");
    printf("║  ────────────────────────────────────────────────────────────────────     ║\n");
    printf("║  Baseline mean:       %.6f                                            ║\n", abs_skew_results.baseline_mean);
    printf("║  Movement mean:       %.6f                                            ║\n", abs_skew_results.movement_mean);
    printf("║  Separation:          %.2fx                                               ║\n", abs_skew_results.separation);
    printf("║  Threshold:           %.6f                                            ║\n", abs_skew_results.threshold);
    printf("║  Accuracy:            %.1f%%                                              ║\n", abs_skew_results.accuracy);
    printf("║  False positives:     %.1f%%                                              ║\n", abs_skew_results.false_positive_rate);
    printf("║  False negatives:     %.1f%%                                              ║\n", abs_skew_results.false_negative_rate);
    printf("║                                                                           ║\n");
    printf("╠═══════════════════════════════════════════════════════════════════════════╣\n");
    
    // Determine winner
    float best_accuracy = fmaxf(var_results.accuracy, fmaxf(skew_results.accuracy, abs_skew_results.accuracy));
    const char *winner = "Unknown";
    if (fabsf(var_results.accuracy - best_accuracy) < 0.1f) winner = "Variance";
    else if (fabsf(skew_results.accuracy - best_accuracy) < 0.1f) winner = "Skewness";
    else if (fabsf(abs_skew_results.accuracy - best_accuracy) < 0.1f) winner = "Abs-Skewness";
    
    printf("║  🏆 WINNER: %s (%.1f%% accuracy)                                    ║\n", 
           winner, best_accuracy);
    printf("╚═══════════════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // ========================================================================
    // ASSERTIONS - Use best approach
    // ========================================================================
    
    // At least one approach should have > 70% accuracy
    TEST_ASSERT_TRUE(best_accuracy > 70.0f);
    
    printf("✅ Test completed! Best accuracy: %.1f%%\n\n", best_accuracy);
}
