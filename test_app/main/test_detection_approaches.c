/*
 * ESPectre - Detection Approaches Comparison Test
 * 
 * Compares 4 different approaches for movement detection:
 * 1. Fisher criterion (current implementation)
 * 2. Modified Fisher (sqrt in denominator - less penalty for high variance)
 * 3. Simple ratio selection (best single feature by mean ratio)
 * 4. Temporal delta variance only (direct approach)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "calibration.h"
#include "csi_processor.h"
#include "real_csi_data.h"
#include "config_manager.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Include CSI data arrays (must be at file scope to avoid stack overflow)
#include "real_csi_arrays.inc"

// Helper: extract all features for testing
static const uint8_t test_all_features[10] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

// Feature names for display
static const char* feature_names[] = {
    "variance", "skewness", "kurtosis", "entropy", "iqr",
    "spatial_variance", "spatial_correlation", "spatial_gradient",
    "temporal_delta_mean", "temporal_delta_variance"
};

// Helper: Calculate mean of a feature across all packets
static float calculate_feature_mean(const float *values, size_t count) {
    if (count == 0) return 0.0f;
    
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += values[i];
    }
    return sum / count;
}

// Helper: Calculate variance of a feature across all packets
static float calculate_feature_variance(const float *values, size_t count, float mean) {
    if (count < 2) return 0.0f;
    
    float sum_sq = 0.0f;
    for (size_t i = 0; i < count; i++) {
        float diff = values[i] - mean;
        sum_sq += diff * diff;
    }
    return sum_sq / (count - 1);
}

TEST_CASE_ESP(compare_detection_approaches_on_real_data, "[detection][comparison]")
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║     DETECTION APPROACHES COMPARISON TEST              ║\n");
    printf("║     Testing different methods on real CSI data      ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    // Storage for all feature values
    #define NUM_FEATURES 10
    float *baseline_features[NUM_FEATURES];
    float *movement_features[NUM_FEATURES];
    
    // Allocate memory for features
    for (int f = 0; f < NUM_FEATURES; f++) {
        baseline_features[f] = malloc(num_baseline * sizeof(float));
        movement_features[f] = malloc(num_movement * sizeof(float));
    }
    
    // Reset temporal buffer before baseline phase
    csi_reset_temporal_buffer();
    
    // Extract features from all baseline packets
    printf("Extracting features from %d baseline packets...\n", num_baseline);
    for (int p = 0; p < num_baseline; p++) {
        csi_features_t features;
        csi_extract_features(baseline_packets[p], 128, &features, test_all_features, 10);
        
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
    
    // Reset temporal buffer before movement phase
    csi_reset_temporal_buffer();
    
    // Extract features from all movement packets
    printf("Extracting features from %d movement packets...\n", num_movement);
    for (int p = 0; p < num_movement; p++) {
        csi_features_t features;
        csi_extract_features(movement_packets[p], 128, &features, test_all_features, 10);
        
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
    
    // Calculate statistics for each feature
    float baseline_means[NUM_FEATURES];
    float baseline_vars[NUM_FEATURES];
    float movement_means[NUM_FEATURES];
    float movement_vars[NUM_FEATURES];
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        baseline_means[f] = calculate_feature_mean(baseline_features[f], num_baseline);
        baseline_vars[f] = calculate_feature_variance(baseline_features[f], num_baseline, baseline_means[f]);
        movement_means[f] = calculate_feature_mean(movement_features[f], num_movement);
        movement_vars[f] = calculate_feature_variance(movement_features[f], num_movement, movement_means[f]);
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  APPROACH 1: FISHER CRITERION (Current Implementation)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Calculate Fisher scores
    float fisher_scores[NUM_FEATURES];
    int best_fisher_idx = 0;
    float best_fisher_score = 0.0f;
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        float mean_diff = fabsf(movement_means[f] - baseline_means[f]);
        float var_sum = baseline_vars[f] + movement_vars[f];
        
        if (var_sum > 1e-6f) {
            fisher_scores[f] = (mean_diff * mean_diff) / var_sum;
        } else {
            fisher_scores[f] = 0.0f;
        }
        
        if (fisher_scores[f] > best_fisher_score) {
            best_fisher_score = fisher_scores[f];
            best_fisher_idx = f;
        }
        
        printf("  %s: Fisher=%.4f (μ_base=%.2f, μ_move=%.2f, σ²_base=%.2f, σ²_move=%.2f)\n",
               feature_names[f], fisher_scores[f],
               baseline_means[f], movement_means[f],
               baseline_vars[f], movement_vars[f]);
    }
    
    printf("\n✅ Best Fisher feature: %s (score=%.4f)\n", 
           feature_names[best_fisher_idx], best_fisher_score);
    printf("   Baseline: %.2f, Movement: %.2f\n", 
           baseline_means[best_fisher_idx], movement_means[best_fisher_idx]);
    float fisher_separation = movement_means[best_fisher_idx] / (baseline_means[best_fisher_idx] + 1e-6f);
    printf("   Separation ratio: %.2fx\n\n", fisher_separation);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  APPROACH 2: MODIFIED FISHER (Sqrt Denominator)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Calculate Modified Fisher scores (sqrt in denominator)
    float modified_fisher_scores[NUM_FEATURES];
    int best_mod_fisher_idx = 0;
    float best_mod_fisher_score = 0.0f;
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        float mean_diff = fabsf(movement_means[f] - baseline_means[f]);
        float var_sum = baseline_vars[f] + movement_vars[f];
        
        if (var_sum > 1e-6f) {
            // Modified Fisher: divide by sqrt instead of raw sum
            modified_fisher_scores[f] = (mean_diff * mean_diff) / sqrtf(var_sum);
        } else {
            modified_fisher_scores[f] = 0.0f;
        }
        
        if (modified_fisher_scores[f] > best_mod_fisher_score) {
            best_mod_fisher_score = modified_fisher_scores[f];
            best_mod_fisher_idx = f;
        }
        
        printf("  %s: ModFisher=%.4f (standard=%.4f)\n",
               feature_names[f], modified_fisher_scores[f], fisher_scores[f]);
    }
    
    printf("\n✅ Best Modified Fisher feature: %s (score=%.4f)\n", 
           feature_names[best_mod_fisher_idx], best_mod_fisher_score);
    printf("   Baseline: %.2f, Movement: %.2f\n", 
           baseline_means[best_mod_fisher_idx], movement_means[best_mod_fisher_idx]);
    float mod_fisher_separation = movement_means[best_mod_fisher_idx] / (baseline_means[best_mod_fisher_idx] + 1e-6f);
    printf("   Separation ratio: %.2fx\n\n", mod_fisher_separation);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  APPROACH 3: SIMPLE RATIO (Best Single Feature)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Calculate simple ratios
    float ratios[NUM_FEATURES];
    int best_ratio_idx = 0;
    float best_ratio = 0.0f;
    
    for (int f = 0; f < NUM_FEATURES; f++) {
        // Use absolute values and add epsilon to avoid division by zero
        float base_abs = fabsf(baseline_means[f]) + 1e-6f;
        float move_abs = fabsf(movement_means[f]) + 1e-6f;
        
        // Calculate ratio (always > 0)
        if (move_abs > base_abs) {
            ratios[f] = move_abs / base_abs;
        } else {
            ratios[f] = base_abs / move_abs;
        }
        
        if (ratios[f] > best_ratio) {
            best_ratio = ratios[f];
            best_ratio_idx = f;
        }
        
        printf("  %s: ratio=%.2fx (μ_base=%.2f, μ_move=%.2f)\n",
               feature_names[f], ratios[f],
               baseline_means[f], movement_means[f]);
    }
    
    printf("\n✅ Best ratio feature: %s (ratio=%.2fx)\n", 
           feature_names[best_ratio_idx], best_ratio);
    printf("   Baseline: %.2f, Movement: %.2f\n", 
           baseline_means[best_ratio_idx], movement_means[best_ratio_idx]);
    printf("   Separation ratio: %.2fx\n\n", best_ratio);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  APPROACH 4: TEMPORAL_DELTA_VARIANCE ONLY\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Direct approach with temporal_delta_variance (index 9)
    int tdv_idx = 9;
    float tdv_baseline = baseline_means[tdv_idx];
    float tdv_movement = movement_means[tdv_idx];
    float tdv_separation = tdv_movement / (tdv_baseline + 1e-6f);
    float tdv_threshold = (tdv_baseline + tdv_movement) / 2.0f;
    
    printf("  Feature: %s\n", feature_names[tdv_idx]);
    printf("  Baseline mean: %.2f (σ²=%.2f)\n", tdv_baseline, baseline_vars[tdv_idx]);
    printf("  Movement mean: %.2f (σ²=%.2f)\n", tdv_movement, movement_vars[tdv_idx]);
    printf("  Separation ratio: %.2fx\n", tdv_separation);
    printf("  Suggested threshold: %.2f\n\n", tdv_threshold);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  APPROACH 5: AMPLITUDE KURTOSIS (Moving Window)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Test amplitude-based kurtosis (same approach as amplitude skewness)
    // Calculate kurtosis on moving window of amplitudes
    #define KURTOSIS_WINDOW 20
    float amp_buffer_kurt[KURTOSIS_WINDOW];
    int amp_idx_kurt = 0;
    int amp_count_kurt = 0;
    
    // Calculate amplitude kurtosis for baseline
    float *baseline_amp_kurtosis = malloc(num_baseline * sizeof(float));
    int baseline_kurt_count = 0;
    
    for (int p = 0; p < num_baseline; p++) {
        // Calculate average amplitude
        float avg_amp = 0.0f;
        for (int i = 0; i < 64; i++) {
            float I = (float)baseline_packets[p][2*i];
            float Q = (float)baseline_packets[p][2*i+1];
            avg_amp += sqrtf(I*I + Q*Q);
        }
        avg_amp /= 64.0f;
        
        // Add to buffer
        amp_buffer_kurt[amp_idx_kurt] = avg_amp;
        amp_idx_kurt = (amp_idx_kurt + 1) % KURTOSIS_WINDOW;
        if (amp_count_kurt < KURTOSIS_WINDOW) amp_count_kurt++;
        
        // Calculate kurtosis inline
        if (amp_count_kurt >= 4) {
            float mean = 0.0f;
            for (int i = 0; i < amp_count_kurt; i++) {
                mean += amp_buffer_kurt[i];
            }
            mean /= amp_count_kurt;
            
            float m2 = 0.0f, m4 = 0.0f;
            for (int i = 0; i < amp_count_kurt; i++) {
                float diff = amp_buffer_kurt[i] - mean;
                float diff2 = diff * diff;
                m2 += diff2;
                m4 += diff2 * diff2;
            }
            m2 /= amp_count_kurt;
            m4 /= amp_count_kurt;
            
            if (m2 > 1e-6f) {
                baseline_amp_kurtosis[baseline_kurt_count++] = (m4 / (m2 * m2)) - 3.0f;
            }
        }
    }
    
    // Calculate amplitude kurtosis for movement
    float *movement_amp_kurtosis = malloc(num_movement * sizeof(float));
    int movement_kurt_count = 0;
    amp_idx_kurt = 0;
    amp_count_kurt = 0;
    
    for (int p = 0; p < num_movement; p++) {
        // Calculate average amplitude
        float avg_amp = 0.0f;
        for (int i = 0; i < 64; i++) {
            float I = (float)movement_packets[p][2*i];
            float Q = (float)movement_packets[p][2*i+1];
            avg_amp += sqrtf(I*I + Q*Q);
        }
        avg_amp /= 64.0f;
        
        // Add to buffer
        amp_buffer_kurt[amp_idx_kurt] = avg_amp;
        amp_idx_kurt = (amp_idx_kurt + 1) % KURTOSIS_WINDOW;
        if (amp_count_kurt < KURTOSIS_WINDOW) amp_count_kurt++;
        
        // Calculate kurtosis inline
        if (amp_count_kurt >= 4) {
            float mean = 0.0f;
            for (int i = 0; i < amp_count_kurt; i++) {
                mean += amp_buffer_kurt[i];
            }
            mean /= amp_count_kurt;
            
            float m2 = 0.0f, m4 = 0.0f;
            for (int i = 0; i < amp_count_kurt; i++) {
                float diff = amp_buffer_kurt[i] - mean;
                float diff2 = diff * diff;
                m2 += diff2;
                m4 += diff2 * diff2;
            }
            m2 /= amp_count_kurt;
            m4 /= amp_count_kurt;
            
            if (m2 > 1e-6f) {
                movement_amp_kurtosis[movement_kurt_count++] = (m4 / (m2 * m2)) - 3.0f;
            }
        }
    }
    
    // Calculate statistics
    float amp_kurt_baseline_mean = calculate_feature_mean(baseline_amp_kurtosis, baseline_kurt_count);
    float amp_kurt_movement_mean = calculate_feature_mean(movement_amp_kurtosis, movement_kurt_count);
    float amp_kurt_baseline_var = calculate_feature_variance(baseline_amp_kurtosis, baseline_kurt_count, amp_kurt_baseline_mean);
    float amp_kurt_movement_var = calculate_feature_variance(movement_amp_kurtosis, movement_kurt_count, amp_kurt_movement_mean);
    
    // Calculate separation
    float amp_kurt_separation = fabsf(amp_kurt_movement_mean) / (fabsf(amp_kurt_baseline_mean) + 1e-6f);
    if (amp_kurt_separation < 1.0f) {
        amp_kurt_separation = 1.0f / amp_kurt_separation;
    }
    
    printf("  Feature: amplitude_kurtosis (moving window)\n");
    printf("  Baseline mean: %.2f (σ²=%.2f)\n", amp_kurt_baseline_mean, amp_kurt_baseline_var);
    printf("  Movement mean: %.2f (σ²=%.2f)\n", amp_kurt_movement_mean, amp_kurt_movement_var);
    printf("  Separation ratio: %.2fx\n", amp_kurt_separation);
    printf("  Comparison with raw kurtosis: %.2fx vs %.2fx\n\n", 
           amp_kurt_separation, ratios[2]);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  FINAL COMPARISON\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("  Approach 1 (Fisher):           %.2fx separation\n", fisher_separation);
    printf("  Approach 2 (Modified Fisher):  %.2fx separation\n", mod_fisher_separation);
    printf("  Approach 3 (Simple Ratio):     %.2fx separation\n", best_ratio);
    printf("  Approach 4 (TDV Only):         %.2fx separation\n", tdv_separation);
    printf("  Approach 5 (Amp Kurtosis):     %.2fx separation\n\n", amp_kurt_separation);
    
    // Determine winner among all approaches
    float max_separation = fisher_separation;
    int winner = 1;
    
    if (mod_fisher_separation > max_separation) {
        max_separation = mod_fisher_separation;
        winner = 2;
    }
    if (best_ratio > max_separation) {
        max_separation = best_ratio;
        winner = 3;
    }
    if (tdv_separation > max_separation) {
        max_separation = tdv_separation;
        winner = 4;
    }
    if (amp_kurt_separation > max_separation) {
        max_separation = amp_kurt_separation;
        winner = 5;
    }
    
    printf("🏆 WINNER: Approach %d with %.2fx separation\n", winner, max_separation);
    
    if (winner == 1) {
        printf("   Best feature: %s (Fisher criterion)\n", feature_names[best_fisher_idx]);
        printf("   💡 Current implementation is optimal!\n");
    } else if (winner == 2) {
        printf("   Best feature: %s (Modified Fisher)\n", feature_names[best_mod_fisher_idx]);
        printf("   💡 Recommendation: Use Modified Fisher criterion\n");
        printf("   💡 Improvement: %.0f%% better than standard Fisher\n", 
               ((mod_fisher_separation / fisher_separation) - 1.0f) * 100.0f);
    } else if (winner == 3) {
        printf("   Best feature: %s (simple ratio)\n", feature_names[best_ratio_idx]);
        printf("   💡 Recommendation: Use this single feature\n");
        printf("   💡 Improvement: %.0f%% better than Fisher\n", 
               ((best_ratio / fisher_separation) - 1.0f) * 100.0f);
    } else if (winner == 4) {
        printf("   Best feature: temporal_delta_variance\n");
        printf("   💡 Recommendation: Use only temporal_delta_variance\n");
        printf("   💡 Improvement: %.0f%% better than Fisher\n", 
               ((tdv_separation / fisher_separation) - 1.0f) * 100.0f);
    } else {
        printf("   Best feature: amplitude_kurtosis (moving window)\n");
        printf("   💡 Recommendation: Implement amplitude kurtosis like skewness\n");
        printf("   💡 Improvement: %.0f%% better than Fisher\n", 
               ((amp_kurt_separation / fisher_separation) - 1.0f) * 100.0f);
        printf("   💡 Comparison: %.0f%% better than raw kurtosis\n",
               ((amp_kurt_separation / ratios[2]) - 1.0f) * 100.0f);
    }
    
    printf("\n");
    
    // Verify we have meaningful separation with at least one approach
    TEST_ASSERT_TRUE(max_separation > 1.5f);
    
    // If temporal approaches win, they should have separation > 2.0
    if (winner == 2 || winner == 3) {
        printf("✅ Temporal features provide significantly better separation!\n");
        TEST_ASSERT_TRUE(max_separation > 2.0f);
    }
    
    // Cleanup
    for (int f = 0; f < NUM_FEATURES; f++) {
        free(baseline_features[f]);
        free(movement_features[f]);
    }
    free(baseline_amp_kurtosis);
    free(movement_amp_kurtosis);
}
