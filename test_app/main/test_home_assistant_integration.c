/*
 * ESPectre - Home Assistant Integration Test
 * 
 * End-to-end test simulating real Home Assistant usage for security/presence detection.
 * Validates complete system behavior including debouncing, persistence, and state transitions.
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
#define DEBOUNCE_COUNT 3  // Consecutive detections required
#define PERSISTENCE_PACKETS 15  // ~1 second at 15 pps

static const uint8_t test_all_features[NUM_FEATURES] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// State machine states (mimicking detection_engine.c)
typedef enum {
    STATE_IDLE,
    STATE_DETECTED
} detection_state_t;

// Detection statistics
typedef struct {
    int total_packets;
    int idle_packets;
    int detected_packets;
    int state_transitions;
    int false_positives;
    int false_negatives;
    float avg_detection_latency;
    float avg_persistence_time;
} detection_stats_t;

// Helper: Calculate detection score
static float calculate_detection_score(const csi_features_t *features) {
    // Weighted combination of top features
    float score = 0.0f;
    score += features->temporal_delta_mean * 0.4f;
    score += features->spatial_gradient * 0.3f;
    score += features->entropy * 0.3f;
    return score;
}

// Helper: State machine with debouncing and persistence
static detection_state_t update_state_machine(detection_state_t current_state,
                                             float score, float threshold,
                                             int *consecutive_detections,
                                             int *persistence_counter) {
    detection_state_t new_state = current_state;
    
    if (score >= threshold) {
        (*consecutive_detections)++;
        *persistence_counter = PERSISTENCE_PACKETS;  // Reset persistence
        
        if (current_state == STATE_IDLE && *consecutive_detections >= DEBOUNCE_COUNT) {
            new_state = STATE_DETECTED;
        }
    } else {
        *consecutive_detections = 0;
        
        if (current_state == STATE_DETECTED) {
            (*persistence_counter)--;
            if (*persistence_counter <= 0) {
                new_state = STATE_IDLE;
            }
        }
    }
    
    return new_state;
}

TEST_CASE_ESP(home_assistant_integration_e2e, "[integration][home_assistant]")
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   HOME ASSISTANT INTEGRATION TEST (END-TO-END)        ║\n");
    printf("║   Simulating real-world security/presence detection   ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Calculate threshold from baseline
    printf("Phase 1: Calibration (calculating threshold)...\n");
    
    float *baseline_scores = malloc(num_baseline * sizeof(float));
    if (!baseline_scores) {
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    csi_reset_temporal_buffer();
    
    int baseline_valid_count = 0;
    for (int p = 0; p < num_baseline; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, NUM_FEATURES);
        
        if (p < WARMUP_PACKETS) continue;
        
        float score = calculate_detection_score(&features);
        baseline_scores[baseline_valid_count++] = score;
    }
    
    // Calculate mean and std
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
    
    float threshold = baseline_mean + 2.0f * baseline_std;
    
    printf("  Baseline mean: %.4f\n", baseline_mean);
    printf("  Baseline std:  %.4f\n", baseline_std);
    printf("  Threshold:     %.4f\n\n", threshold);
    
    free(baseline_scores);
    
    // ========================================================================
    // Phase 2: Simulate realistic Home Assistant scenario
    // ========================================================================
    printf("Phase 2: Simulating realistic Home Assistant scenario...\n");
    printf("  Scenario: Empty room → Person enters → Moves around → Leaves\n\n");
    
    detection_stats_t stats;
    memset(&stats, 0, sizeof(stats));
    
    detection_state_t state = STATE_IDLE;
    int consecutive_detections = 0;
    int persistence_counter = 0;
    
    int packets_in_state = 0;
    int transition_count = 0;
    
    // Simulate sequence: 200 baseline → 300 movement → 200 baseline
    int sequence_phases[] = {200, 300, 200};
    const char* phase_names[] = {"Empty Room", "Person Present", "Person Left"};
    bool expected_detection[] = {false, true, false};
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  PROCESSING SEQUENCE\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    for (int phase = 0; phase < 3; phase++) {
        printf("Phase %d: %s (%d packets)\n", phase + 1, phase_names[phase], sequence_phases[phase]);
        
        csi_reset_temporal_buffer();
        
        int phase_detections = 0;
        int phase_idle = 0;
        int phase_transitions = 0;
        detection_state_t prev_state = state;
        
        for (int p = 0; p < sequence_phases[phase]; p++) {
            csi_features_t features;
            
            // Select appropriate packet source
            if (phase == 1) {
                // Movement phase
                int idx = p % num_movement;
                csi_extract_features(movement_packets[idx], 128, &features, test_all_features, NUM_FEATURES);
            } else {
                // Baseline phase
                int idx = p % num_baseline;
                csi_extract_features(baseline_packets[idx], 128, &features, test_all_features, NUM_FEATURES);
            }
            
            if (p < WARMUP_PACKETS) continue;
            
            float score = calculate_detection_score(&features);
            
            // Update state machine
            detection_state_t new_state = update_state_machine(state, score, threshold,
                                                              &consecutive_detections,
                                                              &persistence_counter);
            
            // Track state transitions
            if (new_state != state) {
                phase_transitions++;
                transition_count++;
                packets_in_state = 0;
            }
            
            state = new_state;
            packets_in_state++;
            
            // Count states
            if (state == STATE_DETECTED) {
                phase_detections++;
            } else {
                phase_idle++;
            }
            
            stats.total_packets++;
        }
        
        int phase_valid = sequence_phases[phase] - WARMUP_PACKETS;
        float detection_rate = (float)phase_detections / phase_valid * 100.0f;
        
        printf("  State transitions: %d\n", phase_transitions);
        printf("  Detected packets:  %d/%d (%.1f%%)\n", phase_detections, phase_valid, detection_rate);
        printf("  Idle packets:      %d/%d (%.1f%%)\n", phase_idle, phase_valid, 
               (float)phase_idle / phase_valid * 100.0f);
        
        // Evaluate correctness
        if (expected_detection[phase]) {
            // Should detect movement
            if (detection_rate >= 80.0f) {
                printf("  ✅ Correctly detected movement\n");
            } else {
                printf("  ⚠️  Low detection rate (expected >80%%)\n");
                stats.false_negatives += (phase_valid - phase_detections);
            }
        } else {
            // Should be idle
            float idle_rate = (float)phase_idle / phase_valid * 100.0f;
            if (idle_rate >= 90.0f) {
                printf("  ✅ Correctly remained idle\n");
            } else {
                printf("  ⚠️  High false alarm rate\n");
                stats.false_positives += phase_detections;
            }
        }
        
        printf("\n");
    }
    
    stats.state_transitions = transition_count;
    
    // ========================================================================
    // Phase 3: Calculate overall metrics
    // ========================================================================
    printf("═══════════════════════════════════════════════════════\n");
    printf("  OVERALL PERFORMANCE METRICS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Total packets processed: %d\n", stats.total_packets);
    printf("State transitions:       %d\n", stats.state_transitions);
    printf("\n");
    
    // Calculate rates
    int expected_baseline_packets = (sequence_phases[0] + sequence_phases[2] - 2 * WARMUP_PACKETS);
    int expected_movement_packets = sequence_phases[1] - WARMUP_PACKETS;
    
    float fp_rate = (float)stats.false_positives / expected_baseline_packets * 100.0f;
    float fn_rate = (float)stats.false_negatives / expected_movement_packets * 100.0f;
    
    printf("False Positives:  %d/%d (%.2f%%) %s\n", 
           stats.false_positives, expected_baseline_packets, fp_rate,
           fp_rate <= 10.0f ? "✅ GOOD" : "⚠️  HIGH");
    
    printf("False Negatives:  %d/%d (%.2f%%) %s\n", 
           stats.false_negatives, expected_movement_packets, fn_rate,
           fn_rate <= 10.0f ? "✅ GOOD" : "⚠️  HIGH");
    
    float recall = 100.0f - fn_rate;
    printf("\nRecall (Detection Rate): %.2f%% %s\n", recall,
           recall >= 90.0f ? "✅ TARGET MET" : "⚠️  BELOW TARGET");
    
    float specificity = 100.0f - fp_rate;
    printf("Specificity (Idle Accuracy): %.2f%%\n", specificity);
    
    // Calculate F1-score
    float precision = (expected_movement_packets - stats.false_negatives + stats.false_positives > 0) ?
        (float)(expected_movement_packets - stats.false_negatives) / 
        (expected_movement_packets - stats.false_negatives + stats.false_positives) * 100.0f : 0.0f;
    
    float f1_score = (precision + recall > 0) ? 
        2.0f * (precision * recall) / (precision + recall) : 0.0f;
    
    printf("Precision: %.2f%%\n", precision);
    printf("F1-Score:  %.2f%%\n\n", f1_score);
    
    // Calculate false alarms per hour
    float fp_per_hour = fp_rate / 100.0f * 15.0f * 3600.0f;
    printf("Expected false alarms: ~%.1f per hour (at 15 pps) %s\n", fp_per_hour,
           (fp_per_hour >= 1.0f && fp_per_hour <= 5.0f) ? "✅ TARGET" : 
           fp_per_hour < 1.0f ? "✅ EXCELLENT" : "⚠️  HIGH");
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  HOME ASSISTANT INTEGRATION ASSESSMENT\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    bool meets_recall = recall >= 90.0f;
    bool meets_fp = fp_per_hour >= 1.0f && fp_per_hour <= 5.0f;
    bool meets_transitions = stats.state_transitions >= 2 && stats.state_transitions <= 6;
    
    printf("✓ Recall Target (90%%):           %s (%.1f%%)\n", 
           meets_recall ? "✅ MET" : "❌ NOT MET", recall);
    printf("✓ FP Rate Target (1-5/hour):     %s (%.1f/hour)\n", 
           meets_fp ? "✅ MET" : "❌ NOT MET", fp_per_hour);
    printf("✓ State Stability:                %s (%d transitions)\n", 
           meets_transitions ? "✅ STABLE" : "⚠️  UNSTABLE", stats.state_transitions);
    
    printf("\n");
    
    if (meets_recall && meets_fp && meets_transitions) {
        printf("🎉 EXCELLENT! System is ready for Home Assistant integration.\n");
        printf("   All targets met for security/presence detection.\n");
    } else if (meets_recall && meets_transitions) {
        printf("✅ GOOD: System meets recall target with stable state transitions.\n");
        if (!meets_fp) {
            printf("   ⚠️  False positive rate is outside target range.\n");
            printf("   Consider adjusting threshold or debounce settings.\n");
        }
    } else {
        printf("⚠️  WARNING: System needs tuning for Home Assistant.\n");
        if (!meets_recall) {
            printf("   - Recall below 90%% target\n");
            printf("   - Recommendation: Lower threshold or combine more features\n");
        }
        if (!meets_transitions) {
            printf("   - Too many state transitions (unstable)\n");
            printf("   - Recommendation: Increase debounce count or persistence time\n");
        }
    }
    
    printf("\n");
    printf("💡 Configuration Recommendations:\n");
    printf("   Threshold:        %.4f\n", threshold);
    printf("   Debounce:         %d consecutive detections\n", DEBOUNCE_COUNT);
    printf("   Persistence:      %d packets (~%.1f seconds)\n", 
           PERSISTENCE_PACKETS, (float)PERSISTENCE_PACKETS / 15.0f);
    printf("   Top Features:     temporal_delta_mean, spatial_gradient, entropy\n");
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  JSON OUTPUT (for Python analysis)\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("{\n");
    printf("  \"test_name\": \"home_assistant_integration_e2e\",\n");
    printf("  \"threshold\": %.4f,\n", threshold);
    printf("  \"debounce_count\": %d,\n", DEBOUNCE_COUNT);
    printf("  \"persistence_packets\": %d,\n", PERSISTENCE_PACKETS);
    printf("  \"metrics\": {\n");
    printf("    \"recall\": %.4f,\n", recall / 100.0f);
    printf("    \"precision\": %.4f,\n", precision / 100.0f);
    printf("    \"f1_score\": %.4f,\n", f1_score / 100.0f);
    printf("    \"fp_rate\": %.4f,\n", fp_rate / 100.0f);
    printf("    \"fn_rate\": %.4f,\n", fn_rate / 100.0f);
    printf("    \"fp_per_hour\": %.2f,\n", fp_per_hour);
    printf("    \"state_transitions\": %d\n", stats.state_transitions);
    printf("  },\n");
    printf("  \"targets_met\": {\n");
    printf("    \"recall_90\": %s,\n", meets_recall ? "true" : "false");
    printf("    \"fp_1_5_per_hour\": %s,\n", meets_fp ? "true" : "false");
    printf("    \"stable_transitions\": %s\n", meets_transitions ? "true" : "false");
    printf("  },\n");
    printf("  \"ready_for_production\": %s\n", 
           (meets_recall && meets_fp && meets_transitions) ? "true" : "false");
    printf("}\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Verify minimum acceptable performance
    TEST_ASSERT_TRUE(recall > 70.0f);  // At least 70% recall
    TEST_ASSERT_TRUE(fp_rate < 20.0f);  // Less than 20% FP rate
}
