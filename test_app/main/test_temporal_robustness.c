/*
 * ESPectre - Temporal Robustness Test
 * 
 * Tests system behavior in real-world temporal scenarios:
 * - Baseline → Movement transitions (intrusion detection)
 * - Movement → Baseline transitions (person leaving)
 * - Prolonged baseline (empty room stability)
 * - Detection latency measurement
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

static const char* feature_names[] = {
    "variance", "skewness", "kurtosis", "entropy", "iqr",
    "spatial_variance", "spatial_correlation", "spatial_gradient",
    "temporal_delta_mean", "temporal_delta_variance"
};

static const uint8_t test_all_features[NUM_FEATURES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// Scenario result structure
typedef struct {
    const char* name;
    int total_packets;
    int detections;
    int false_positives;
    int false_negatives;
    float detection_rate;
    float false_alarm_rate;
    int first_detection_packet;  // Latency measurement
    float avg_score;
} scenario_result_t;

// Helper: Calculate feature score (simple weighted sum of top features)
static float calculate_detection_score(const csi_features_t *features) {
    // Use top 3 features from performance analysis
    // temporal_delta_mean, spatial_gradient, entropy
    float score = 0.0f;
    
    // Normalize and weight features (empirical weights)
    score += features->temporal_delta_mean * 0.4f;
    score += features->spatial_gradient * 0.3f;
    score += features->entropy * 0.3f;
    
    return score;
}

TEST_CASE_ESP(temporal_robustness_scenarios, "[temporal][robustness]")
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   TEMPORAL ROBUSTNESS TEST                            ║\n");
    printf("║   Real-world scenario evaluation                      ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Allocate results for 4 scenarios
    scenario_result_t scenarios[4];
    memset(scenarios, 0, sizeof(scenarios));
    
    scenarios[0].name = "Prolonged Baseline (Empty Room)";
    scenarios[1].name = "Baseline → Movement (Intrusion)";
    scenarios[2].name = "Movement → Baseline (Person Leaves)";
    scenarios[3].name = "Continuous Movement";
    
    // Determine threshold from baseline data
    printf("Calculating detection threshold from baseline data...\n");
    
    float *baseline_scores = malloc(num_baseline * sizeof(float));
    if (!baseline_scores) {
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    int baseline_valid_count = 0;
    for (int p = 0; p < num_baseline; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        float score = calculate_detection_score(&features);
        baseline_scores[baseline_valid_count++] = score;
    }
    
    // Calculate mean and std of baseline
    float baseline_mean = 0.0f;
    for (int i = 0; i < baseline_valid_count; i++) {
        baseline_mean += baseline_scores[i];
    }
    baseline_mean /= baseline_valid_count;
    
    float baseline_std = 0.0f;
    for (int i = 0; i < baseline_valid_count; i++) {
        float diff = baseline_scores[i] - baseline_mean;
        baseline_std += diff * diff;
    }
    baseline_std = sqrtf(baseline_std / baseline_valid_count);
    
    // Threshold: mean + 2*std (captures ~95% of baseline)
    float threshold = baseline_mean + 2.0f * baseline_std;
    
    printf("  Baseline mean: %.4f\n", baseline_mean);
    printf("  Baseline std:  %.4f\n", baseline_std);
    printf("  Threshold:     %.4f (mean + 2*std)\n\n", threshold);
    
    free(baseline_scores);
    
    // ========================================================================
    // SCENARIO 1: Prolonged Baseline (Empty Room Stability)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SCENARIO 1: Prolonged Baseline (Empty Room)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Testing stability with %d baseline packets...\n", num_baseline);
    
    csi_reset_temporal_buffer();
    
    int fp_count = 0;
    float total_score = 0.0f;
    
    for (int p = 0; p < num_baseline; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        float score = calculate_detection_score(&features);
        total_score += score;
        
        if (score >= threshold) {
            fp_count++;
        }
    }
    
    scenarios[0].total_packets = baseline_valid_count;
    scenarios[0].detections = fp_count;
    scenarios[0].false_positives = fp_count;
    scenarios[0].false_negatives = 0;
    scenarios[0].detection_rate = 0.0f;  // Should be 0 for baseline
    scenarios[0].false_alarm_rate = (float)fp_count / baseline_valid_count * 100.0f;
    scenarios[0].first_detection_packet = -1;
    scenarios[0].avg_score = total_score / baseline_valid_count;
    
    printf("Results:\n");
    printf("  Total packets:     %d\n", scenarios[0].total_packets);
    printf("  False alarms:      %d\n", scenarios[0].false_positives);
    printf("  False alarm rate:  %.2f%% %s\n", 
           scenarios[0].false_alarm_rate,
           scenarios[0].false_alarm_rate <= 10.0f ? "✅ GOOD" : "⚠️  HIGH");
    printf("  Average score:     %.4f (threshold: %.4f)\n", scenarios[0].avg_score, threshold);
    
    float fp_per_hour = scenarios[0].false_alarm_rate / 100.0f * 15.0f * 3600.0f;
    printf("  Expected FP/hour:  ~%.1f %s\n", fp_per_hour,
           (fp_per_hour >= 1.0f && fp_per_hour <= 5.0f) ? "✅ TARGET" : "⚠️ ");
    printf("\n");
    
    // ========================================================================
    // SCENARIO 2: Baseline → Movement (Intrusion Detection)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SCENARIO 2: Baseline → Movement (Intrusion)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Simulating intrusion: baseline → movement transition...\n");
    
    csi_reset_temporal_buffer();
    
    // Process first 100 baseline packets
    int baseline_phase = 100;
    if (baseline_phase > num_baseline) baseline_phase = num_baseline;
    
    for (int p = 0; p < baseline_phase; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, NUM_FEATURES);
    }
    
    printf("  Processed %d baseline packets (room empty)\n", baseline_phase);
    
    // Now process movement packets (intrusion)
    int movement_phase = 100;
    if (movement_phase > num_movement) movement_phase = num_movement;
    
    int detections = 0;
    int first_detection = -1;
    total_score = 0.0f;
    
    for (int p = 0; p < movement_phase; p++) {
        csi_features_t features;
        csi_extract_features(movement_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        float score = calculate_detection_score(&features);
        total_score += score;
        
        if (score >= threshold) {
            detections++;
            if (first_detection == -1) {
                first_detection = p;
            }
        }
    }
    
    int movement_valid = movement_phase - WARMUP_PACKETS;
    
    scenarios[1].total_packets = movement_valid;
    scenarios[1].detections = detections;
    scenarios[1].false_positives = 0;
    scenarios[1].false_negatives = movement_valid - detections;
    scenarios[1].detection_rate = (float)detections / movement_valid * 100.0f;
    scenarios[1].false_alarm_rate = 0.0f;
    scenarios[1].first_detection_packet = first_detection;
    scenarios[1].avg_score = total_score / movement_valid;
    
    printf("Results:\n");
    printf("  Movement packets:  %d\n", scenarios[1].total_packets);
    printf("  Detections:        %d\n", scenarios[1].detections);
    printf("  Detection rate:    %.2f%% %s\n", 
           scenarios[1].detection_rate,
           scenarios[1].detection_rate >= 90.0f ? "✅ TARGET MET" : "⚠️  BELOW TARGET");
    printf("  Missed detections: %d\n", scenarios[1].false_negatives);
    printf("  First detection:   packet #%d\n", first_detection);
    
    if (first_detection >= 0) {
        float latency_seconds = (float)first_detection / 15.0f;  // Assuming 15 pps
        printf("  Detection latency: %.2f seconds %s\n", latency_seconds,
               latency_seconds <= 2.0f ? "✅ FAST" : "⚠️  SLOW");
    }
    printf("  Average score:     %.4f (threshold: %.4f)\n", scenarios[1].avg_score, threshold);
    printf("\n");
    
    // ========================================================================
    // SCENARIO 3: Movement → Baseline (Person Leaves)
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SCENARIO 3: Movement → Baseline (Person Leaves)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Simulating person leaving: movement → baseline transition...\n");
    
    csi_reset_temporal_buffer();
    
    // Process first 100 movement packets
    movement_phase = 100;
    if (movement_phase > num_movement) movement_phase = num_movement;
    
    for (int p = 0; p < movement_phase; p++) {
        csi_features_t features;
        csi_extract_features(movement_packets[p], 128, &features, test_all_features, NUM_FEATURES);
    }
    
    printf("  Processed %d movement packets (person present)\n", movement_phase);
    
    // Now process baseline packets (person left)
    baseline_phase = 100;
    if (baseline_phase > num_baseline) baseline_phase = num_baseline;
    
    fp_count = 0;
    int last_detection = -1;
    total_score = 0.0f;
    
    for (int p = 0; p < baseline_phase; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        float score = calculate_detection_score(&features);
        total_score += score;
        
        if (score >= threshold) {
            fp_count++;
            last_detection = p;
        }
    }
    
    baseline_valid_count = baseline_phase - WARMUP_PACKETS;
    
    scenarios[2].total_packets = baseline_valid_count;
    scenarios[2].detections = fp_count;
    scenarios[2].false_positives = fp_count;
    scenarios[2].false_negatives = 0;
    scenarios[2].detection_rate = 0.0f;
    scenarios[2].false_alarm_rate = (float)fp_count / baseline_valid_count * 100.0f;
    scenarios[2].first_detection_packet = last_detection;
    scenarios[2].avg_score = total_score / baseline_valid_count;
    
    printf("Results:\n");
    printf("  Baseline packets:  %d\n", scenarios[2].total_packets);
    printf("  False alarms:      %d\n", scenarios[2].false_positives);
    printf("  False alarm rate:  %.2f%% %s\n", 
           scenarios[2].false_alarm_rate,
           scenarios[2].false_alarm_rate <= 10.0f ? "✅ GOOD" : "⚠️  HIGH");
    
    if (last_detection >= 0) {
        float persistence_seconds = (float)last_detection / 15.0f;
        printf("  Last false alarm:  packet #%d (%.2f seconds after transition)\n", 
               last_detection, persistence_seconds);
    } else {
        printf("  Last false alarm:  none ✅\n");
    }
    printf("  Average score:     %.4f (threshold: %.4f)\n", scenarios[2].avg_score, threshold);
    printf("\n");
    
    // ========================================================================
    // SCENARIO 4: Continuous Movement
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SCENARIO 4: Continuous Movement\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Testing continuous movement detection...\n");
    
    csi_reset_temporal_buffer();
    
    detections = 0;
    first_detection = -1;
    total_score = 0.0f;
    int movement_valid_count = 0;
    
    for (int p = 0; p < num_movement; p++) {
        csi_features_t features;
        csi_extract_features(movement_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        float score = calculate_detection_score(&features);
        total_score += score;
        movement_valid_count++;
        
        if (score >= threshold) {
            detections++;
            if (first_detection == -1) {
                first_detection = p;
            }
        }
    }
    
    scenarios[3].total_packets = movement_valid_count;
    scenarios[3].detections = detections;
    scenarios[3].false_positives = 0;
    scenarios[3].false_negatives = movement_valid_count - detections;
    scenarios[3].detection_rate = (float)detections / movement_valid_count * 100.0f;
    scenarios[3].false_alarm_rate = 0.0f;
    scenarios[3].first_detection_packet = first_detection;
    scenarios[3].avg_score = total_score / movement_valid_count;
    
    printf("Results:\n");
    printf("  Movement packets:  %d\n", scenarios[3].total_packets);
    printf("  Detections:        %d\n", scenarios[3].detections);
    printf("  Detection rate:    %.2f%% %s\n", 
           scenarios[3].detection_rate,
           scenarios[3].detection_rate >= 90.0f ? "✅ TARGET MET" : "⚠️  BELOW TARGET");
    printf("  Missed detections: %d\n", scenarios[3].false_negatives);
    printf("  Average score:     %.4f (threshold: %.4f)\n", scenarios[3].avg_score, threshold);
    printf("\n");
    
    // ========================================================================
    // SUMMARY
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SUMMARY - ALL SCENARIOS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Scenario                          Detection Rate  FP Rate   Status\n");
    printf("────────────────────────────────────────────────────────────────────\n");
    
    for (int i = 0; i < 4; i++) {
        const char* status;
        if (i == 0 || i == 2) {
            // Baseline scenarios - check FP rate
            status = scenarios[i].false_alarm_rate <= 10.0f ? "✅ PASS" : "⚠️  WARN";
        } else {
            // Movement scenarios - check detection rate
            status = scenarios[i].detection_rate >= 90.0f ? "✅ PASS" : "⚠️  WARN";
        }
        
        printf("%-32s  %6.2f%%        %6.2f%%   %s\n",
               scenarios[i].name,
               scenarios[i].detection_rate,
               scenarios[i].false_alarm_rate,
               status);
    }
    
    printf("\n");
    printf("💡 Recommendations:\n");
    
    // Check if any scenario failed
    bool all_pass = true;
    if (scenarios[0].false_alarm_rate > 10.0f || scenarios[2].false_alarm_rate > 10.0f) {
        printf("   ⚠️  High false alarm rate - consider increasing threshold\n");
        all_pass = false;
    }
    if (scenarios[1].detection_rate < 90.0f || scenarios[3].detection_rate < 90.0f) {
        printf("   ⚠️  Low detection rate - consider decreasing threshold or combining features\n");
        all_pass = false;
    }
    if (all_pass) {
        printf("   ✅ All scenarios passed! System is robust for Home Assistant integration.\n");
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  JSON OUTPUT (for Python analysis)\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("{\n");
    printf("  \"test_name\": \"temporal_robustness_scenarios\",\n");
    printf("  \"threshold\": %.4f,\n", threshold);
    printf("  \"scenarios\": [\n");
    
    for (int i = 0; i < 4; i++) {
        printf("    {\n");
        printf("      \"name\": \"%s\",\n", scenarios[i].name);
        printf("      \"detection_rate\": %.4f,\n", scenarios[i].detection_rate / 100.0f);
        printf("      \"false_alarm_rate\": %.4f,\n", scenarios[i].false_alarm_rate / 100.0f);
        printf("      \"avg_score\": %.4f\n", scenarios[i].avg_score);
        printf("    }%s\n", i < 3 ? "," : "");
    }
    
    printf("  ]\n");
    printf("}\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Verify at least movement scenarios have reasonable detection
    TEST_ASSERT_TRUE(scenarios[1].detection_rate > 50.0f);
    TEST_ASSERT_TRUE(scenarios[3].detection_rate > 50.0f);
}
