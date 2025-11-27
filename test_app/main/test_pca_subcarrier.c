/*
 * ESPectre - PCA Subcarrier Analysis Test (Incremental Processing)
 * 
 * Analyzes which subcarriers are most informative using incremental
 * covariance calculation to minimize memory usage.
 * 
 * This test:
 * 1. Calculates statistics per subcarrier incrementally
 * 2. Builds covariance matrix on-the-fly without storing all data
 * 3. Performs PCA to find principal components
 * 4. Recommends optimal subcarrier selection
 * 
 * Memory efficient: O(N_subcarriers²) instead of O(N_packets × N_subcarriers)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "real_csi_data_esp32.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

// Include CSI data arrays
#include "real_csi_arrays.inc"

#define NUM_SUBCARRIERS 64
#define PCA_COMPONENTS 6  // Reduced from 10 to save memory

// Helper: Calculate amplitude for a single subcarrier
static float calculate_subcarrier_amplitude(const int8_t *packet, int subcarrier_idx) {
    float I = (float)packet[2 * subcarrier_idx];
    float Q = (float)packet[2 * subcarrier_idx + 1];
    return sqrtf(I * I + Q * Q);
}

// Incremental statistics accumulator
typedef struct {
    int index;
    float baseline_sum;
    float baseline_sum_sq;
    int baseline_count;
    float movement_sum;
    float movement_sum_sq;
    int movement_count;
} incremental_stats_t;

// Simplified Power Iteration for finding dominant eigenvector
static float power_iteration(float **matrix, int n, float *eigenvector, int max_iter) {
    // Initialize eigenvector
    for (int i = 0; i < n; i++) {
        eigenvector[i] = 1.0f / sqrtf((float)n);
    }
    
    float eigenvalue = 0.0f;
    float *temp = malloc(n * sizeof(float));
    if (!temp) return 0.0f;
    
    for (int iter = 0; iter < max_iter; iter++) {
        // Multiply matrix by eigenvector
        for (int i = 0; i < n; i++) {
            temp[i] = 0.0f;
            for (int j = 0; j < n; j++) {
                temp[i] += matrix[i][j] * eigenvector[j];
            }
        }
        
        // Calculate eigenvalue (Rayleigh quotient)
        float numerator = 0.0f;
        float denominator = 0.0f;
        for (int i = 0; i < n; i++) {
            numerator += eigenvector[i] * temp[i];
            denominator += eigenvector[i] * eigenvector[i];
        }
        eigenvalue = numerator / denominator;
        
        // Normalize
        float norm = 0.0f;
        for (int i = 0; i < n; i++) {
            norm += temp[i] * temp[i];
        }
        norm = sqrtf(norm);
        
        if (norm < 1e-10f) break;
        
        for (int i = 0; i < n; i++) {
            eigenvector[i] = temp[i] / norm;
        }
    }
    
    free(temp);
    return eigenvalue;
}

TEST_CASE_ESP(pca_subcarrier_analysis_on_real_data, "[pca][subcarrier]")
{
    printf("\n=== PCA SUBCARRIER ANALYSIS ===\n");
    
    // Allocate incremental statistics (minimal memory)
    incremental_stats_t *stats = malloc(NUM_SUBCARRIERS * sizeof(incremental_stats_t));
    if (!stats) {
        printf("ERROR: Failed to allocate stats array\n");
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    // Initialize stats
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        stats[s].index = s;
        stats[s].baseline_sum = 0.0f;
        stats[s].baseline_sum_sq = 0.0f;
        stats[s].baseline_count = 0;
        stats[s].movement_sum = 0.0f;
        stats[s].movement_sum_sq = 0.0f;
        stats[s].movement_count = 0;
    }
    
    printf("Pass 1: Accumulating statistics from baseline packets...\n");
    
    // First pass: accumulate baseline statistics
    for (int p = 0; p < num_baseline; p++) {
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float amp = calculate_subcarrier_amplitude(baseline_packets[p], s);
            stats[s].baseline_sum += amp;
            stats[s].baseline_sum_sq += amp * amp;
            stats[s].baseline_count++;
        }
    }
    
    printf("Pass 2: Accumulating statistics from movement packets...\n");
    
    // Second pass: accumulate movement statistics
    for (int p = 0; p < num_movement; p++) {
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float amp = calculate_subcarrier_amplitude(movement_packets[p], s);
            stats[s].movement_sum += amp;
            stats[s].movement_sum_sq += amp * amp;
            stats[s].movement_count++;
        }
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SUBCARRIER STATISTICS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Calculate final statistics and Fisher scores
    typedef struct {
        int index;
        float baseline_mean;
        float baseline_var;
        float movement_mean;
        float movement_var;
        float fisher_score;
        float variance_ratio;
        float mean_diff;
    } subcarrier_result_t;
    
    subcarrier_result_t *results = malloc(NUM_SUBCARRIERS * sizeof(subcarrier_result_t));
    if (!results) {
        printf("ERROR: Failed to allocate results array\n");
        free(stats);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        results[s].index = s;
        
        // Calculate means
        results[s].baseline_mean = stats[s].baseline_sum / stats[s].baseline_count;
        results[s].movement_mean = stats[s].movement_sum / stats[s].movement_count;
        
        // Calculate variances using E[X²] - E[X]²
        float baseline_mean_sq = stats[s].baseline_sum_sq / stats[s].baseline_count;
        float movement_mean_sq = stats[s].movement_sum_sq / stats[s].movement_count;
        
        results[s].baseline_var = baseline_mean_sq - (results[s].baseline_mean * results[s].baseline_mean);
        results[s].movement_var = movement_mean_sq - (results[s].movement_mean * results[s].movement_mean);
        
        // Protect against negative variance due to floating point errors
        if (results[s].baseline_var < 0.0f) results[s].baseline_var = 0.0f;
        if (results[s].movement_var < 0.0f) results[s].movement_var = 0.0f;
        
        // Fisher score
        float mean_diff = fabsf(results[s].movement_mean - results[s].baseline_mean);
        float var_sum = results[s].baseline_var + results[s].movement_var;
        results[s].fisher_score = (var_sum > 1e-6f) ? (mean_diff * mean_diff) / var_sum : 0.0f;
        
        // Variance ratio
        results[s].variance_ratio = (results[s].baseline_var > 1e-6f) ? 
            results[s].movement_var / results[s].baseline_var : 1.0f;
        
        results[s].mean_diff = mean_diff;
    }
    
    // Sort by Fisher score (descending)
    for (int i = 0; i < NUM_SUBCARRIERS - 1; i++) {
        for (int j = i + 1; j < NUM_SUBCARRIERS; j++) {
            if (results[j].fisher_score > results[i].fisher_score) {
                subcarrier_result_t temp = results[i];
                results[i] = results[j];
                results[j] = temp;
            }
        }
    }
    
    printf("Top 15 subcarriers by Fisher score:\n");
    printf("Rank  SC#   Fisher    VarRatio  MeanDiff  μ_base   μ_move   σ²_base  σ²_move\n");
    printf("────────────────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < 15 && i < NUM_SUBCARRIERS; i++) {
        printf("%2d    %2d    %.4f    %.2fx      %.2f      %.2f     %.2f     %.2f     %.2f\n",
               i + 1, results[i].index, results[i].fisher_score, results[i].variance_ratio,
               results[i].mean_diff, results[i].baseline_mean, results[i].movement_mean,
               results[i].baseline_var, results[i].movement_var);
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SUBCARRIER ZONE ANALYSIS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Analyze zones
    float zone1_fisher = 0.0f;  // [0-9]
    float zone2_fisher = 0.0f;  // [10-54] - current selection
    float zone3_fisher = 0.0f;  // [55-63]
    int zone1_count = 0, zone2_count = 0, zone3_count = 0;
    
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        // Find this subcarrier in sorted results
        float fisher = 0.0f;
        for (int i = 0; i < NUM_SUBCARRIERS; i++) {
            if (results[i].index == s) {
                fisher = results[i].fisher_score;
                break;
            }
        }
        
        if (s < 10) {
            zone1_fisher += fisher;
            zone1_count++;
        } else if (s < 55) {
            zone2_fisher += fisher;
            zone2_count++;
        } else {
            zone3_fisher += fisher;
            zone3_count++;
        }
    }
    
    printf("Zone [0-9] (edge):      avg Fisher=%.4f (%d subcarriers)\n", 
           zone1_fisher / zone1_count, zone1_count);
    printf("Zone [10-54] (old):     avg Fisher=%.4f (%d subcarriers)\n", 
           zone2_fisher / zone2_count, zone2_count);
    printf("Zone [55-63] (edge):    avg Fisher=%.4f (%d subcarriers)\n\n", 
           zone3_fisher / zone3_count, zone3_count);
    
    // Calculate Fisher scores for different selection strategies
    float total_fisher = zone1_fisher + zone2_fisher + zone3_fisher;
    
    // Strategy 1: ALL subcarriers [0-63]
    float all_subcarriers_fisher = total_fisher;
    float all_subcarriers_avg = total_fisher / NUM_SUBCARRIERS;
    
    // Strategy 2: papaer based selection [10-54]
    float old_selection_fisher = zone2_fisher;
    float old_selection_avg = zone2_fisher / zone2_count;
    
    // Strategy 3: New optimal [29-58] (will be calculated below)
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SELECTION STRATEGY COMPARISON\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Strategy 1: ALL subcarriers [0-63] (64 subcarriers)\n");
    printf("  Total Fisher: %.4f\n", all_subcarriers_fisher);
    printf("  Average Fisher: %.4f\n", all_subcarriers_avg);
    printf("  Baseline (100%%)\n\n");
    
    printf("Strategy 2: Old selection [10-54] (45 subcarriers)\n");
    printf("  Total Fisher: %.4f\n", old_selection_fisher);
    printf("  Average Fisher: %.4f\n", old_selection_avg);
    printf("  vs ALL: %.1f%% of total, %.1f%% avg quality\n\n",
           (old_selection_fisher / all_subcarriers_fisher) * 100.0f,
           (old_selection_avg / all_subcarriers_avg) * 100.0f);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  INCREMENTAL COVARIANCE CALCULATION\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Building covariance matrix incrementally (%dx%d)...\n", NUM_SUBCARRIERS, NUM_SUBCARRIERS);
    
    // Allocate covariance matrix (only matrix we need in memory)
    float **cov_matrix = malloc(NUM_SUBCARRIERS * sizeof(float*));
    if (!cov_matrix) {
        printf("ERROR: Failed to allocate covariance matrix\n");
        free(stats);
        free(results);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    for (int i = 0; i < NUM_SUBCARRIERS; i++) {
        cov_matrix[i] = malloc(NUM_SUBCARRIERS * sizeof(float));
        if (!cov_matrix[i]) {
            printf("ERROR: Failed to allocate covariance matrix row %d\n", i);
            for (int k = 0; k < i; k++) {
                free(cov_matrix[k]);
            }
            free(cov_matrix);
            free(stats);
            free(results);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }
        // Initialize to zero
        memset(cov_matrix[i], 0, NUM_SUBCARRIERS * sizeof(float));
    }
    
    // Calculate means for centering (already have from stats)
    float *means = malloc(NUM_SUBCARRIERS * sizeof(float));
    if (!means) {
        printf("ERROR: Failed to allocate means\n");
        for (int i = 0; i < NUM_SUBCARRIERS; i++) {
            free(cov_matrix[i]);
        }
        free(cov_matrix);
        free(stats);
        free(results);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        means[s] = stats[s].movement_sum / stats[s].movement_count;
    }
    
    // Incremental covariance calculation - process one packet at a time
    for (int p = 0; p < num_movement; p++) {
        // Calculate all subcarrier amplitudes for this packet
        float packet_amps[NUM_SUBCARRIERS];
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            packet_amps[s] = calculate_subcarrier_amplitude(movement_packets[p], s);
        }
        
        // Update covariance matrix incrementally
        for (int i = 0; i < NUM_SUBCARRIERS; i++) {
            float diff_i = packet_amps[i] - means[i];
            for (int j = i; j < NUM_SUBCARRIERS; j++) {
                float diff_j = packet_amps[j] - means[j];
                cov_matrix[i][j] += diff_i * diff_j;
            }
        }
    }
    
    // Finalize covariance (divide by N-1 and make symmetric)
    for (int i = 0; i < NUM_SUBCARRIERS; i++) {
        for (int j = i; j < NUM_SUBCARRIERS; j++) {
            cov_matrix[i][j] /= (num_movement - 1);
            cov_matrix[j][i] = cov_matrix[i][j];  // Symmetric
        }
    }
    
    // Calculate total variance (trace of covariance matrix)
    float total_variance = 0.0f;
    for (int i = 0; i < NUM_SUBCARRIERS; i++) {
        total_variance += cov_matrix[i][i];
    }
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  PCA ANALYSIS (Power Iteration Method)\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Allocate for PCA
    float *eigenvalues = malloc(PCA_COMPONENTS * sizeof(float));
    float **eigenvectors = malloc(PCA_COMPONENTS * sizeof(float*));
    
    if (!eigenvalues || !eigenvectors) {
        printf("ERROR: Failed to allocate PCA arrays\n");
        for (int i = 0; i < NUM_SUBCARRIERS; i++) {
            free(cov_matrix[i]);
        }
        free(cov_matrix);
        free(means);
        free(stats);
        free(results);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    for (int i = 0; i < PCA_COMPONENTS; i++) {
        eigenvectors[i] = malloc(NUM_SUBCARRIERS * sizeof(float));
        if (!eigenvectors[i]) {
            printf("ERROR: Failed to allocate eigenvector %d\n", i);
            for (int k = 0; k < i; k++) {
                free(eigenvectors[k]);
            }
            free(eigenvectors);
            free(eigenvalues);
            for (int j = 0; j < NUM_SUBCARRIERS; j++) {
                free(cov_matrix[j]);
            }
            free(cov_matrix);
            free(means);
            free(stats);
            free(results);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }
    }
    
    // Work matrix for deflation (reuse covariance matrix space)
    float **work_matrix = malloc(NUM_SUBCARRIERS * sizeof(float*));
    if (!work_matrix) {
        printf("ERROR: Failed to allocate work matrix\n");
        for (int i = 0; i < PCA_COMPONENTS; i++) {
            free(eigenvectors[i]);
        }
        free(eigenvectors);
        free(eigenvalues);
        for (int i = 0; i < NUM_SUBCARRIERS; i++) {
            free(cov_matrix[i]);
        }
        free(cov_matrix);
        free(means);
        free(stats);
        free(results);
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    for (int i = 0; i < NUM_SUBCARRIERS; i++) {
        work_matrix[i] = malloc(NUM_SUBCARRIERS * sizeof(float));
        if (!work_matrix[i]) {
            printf("ERROR: Failed to allocate work matrix row %d\n", i);
            for (int k = 0; k < i; k++) {
                free(work_matrix[k]);
            }
            free(work_matrix);
            for (int j = 0; j < PCA_COMPONENTS; j++) {
                free(eigenvectors[j]);
            }
            free(eigenvectors);
            free(eigenvalues);
            for (int j = 0; j < NUM_SUBCARRIERS; j++) {
                free(cov_matrix[j]);
            }
            free(cov_matrix);
            free(means);
            free(stats);
            free(results);
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }
        memcpy(work_matrix[i], cov_matrix[i], NUM_SUBCARRIERS * sizeof(float));
    }
    
    // Find principal components
    for (int pc = 0; pc < PCA_COMPONENTS; pc++) {
        eigenvalues[pc] = power_iteration(work_matrix, NUM_SUBCARRIERS, eigenvectors[pc], 50);
        
        // Deflate matrix for next component
        for (int i = 0; i < NUM_SUBCARRIERS; i++) {
            for (int j = 0; j < NUM_SUBCARRIERS; j++) {
                work_matrix[i][j] -= eigenvalues[pc] * eigenvectors[pc][i] * eigenvectors[pc][j];
            }
        }
    }
    
    printf("\n");
    printf("Principal Components Analysis:\n");
    printf("PC#   Eigenvalue   Variance%%   Cumulative%%   Top 3 Subcarriers (weight)\n");
    printf("─────────────────────────────────────────────────────────────────────────\n");
    
    float cumulative_variance = 0.0f;
    int components_for_95pct = 0;
    
    for (int pc = 0; pc < PCA_COMPONENTS; pc++) {
        float variance_explained = (eigenvalues[pc] / total_variance) * 100.0f;
        cumulative_variance += variance_explained;
        
        if (cumulative_variance < 95.0f) {
            components_for_95pct = pc + 1;
        }
        
        // Find top 3 subcarriers with highest absolute weight
        typedef struct {
            int idx;
            float weight;
        } weighted_sc_t;
        
        weighted_sc_t top3[3] = {{0, 0.0f}, {0, 0.0f}, {0, 0.0f}};
        
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float abs_weight = fabsf(eigenvectors[pc][s]);
            
            for (int t = 0; t < 3; t++) {
                if (abs_weight > top3[t].weight) {
                    for (int k = 2; k > t; k--) {
                        top3[k] = top3[k-1];
                    }
                    top3[t].idx = s;
                    top3[t].weight = abs_weight;
                    break;
                }
            }
        }
        
        printf("PC%-2d  %9.2f    %6.2f%%     %6.2f%%     SC%d(%.3f), SC%d(%.3f), SC%d(%.3f)\n",
               pc + 1, eigenvalues[pc], variance_explained, cumulative_variance,
               top3[0].idx, top3[0].weight,
               top3[1].idx, top3[1].weight,
               top3[2].idx, top3[2].weight);
    }
    
    printf("\n");
    printf("💡 Recommendation: Use %d principal components for 95%% variance\n", 
           components_for_95pct > 0 ? components_for_95pct : PCA_COMPONENTS);
    printf("   Total variance explained: %.1f%%\n\n", cumulative_variance);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  OPTIMAL SUBCARRIER SELECTION\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Find optimal continuous range
    int best_start = 0;
    int best_end = NUM_SUBCARRIERS;
    float best_avg_fisher = 0.0f;
    
    for (int window = 30; window <= 50; window += 5) {
        for (int start = 0; start <= NUM_SUBCARRIERS - window; start++) {
            int end = start + window;
            float sum_fisher = 0.0f;
            
            for (int s = start; s < end; s++) {
                for (int i = 0; i < NUM_SUBCARRIERS; i++) {
                    if (results[i].index == s) {
                        sum_fisher += results[i].fisher_score;
                        break;
                    }
                }
            }
            
            float avg_fisher = sum_fisher / window;
            if (avg_fisher > best_avg_fisher) {
                best_avg_fisher = avg_fisher;
                best_start = start;
                best_end = end;
            }
        }
    }
    
    printf("Strategy 3: PCA-optimized [%d-%d] (%d subcarriers)\n", 
           best_start, best_end - 1, best_end - best_start);
    printf("  Total Fisher: %.4f\n", best_avg_fisher * (best_end - best_start));
    printf("  Average Fisher: %.4f\n", best_avg_fisher);
    printf("  vs ALL: %.1f%% of total, %.1f%% avg quality\n",
           (best_avg_fisher * (best_end - best_start) / all_subcarriers_fisher) * 100.0f,
           (best_avg_fisher / all_subcarriers_avg) * 100.0f);
    printf("  vs [10-54]: %.1f%% improvement in avg quality\n\n",
           ((best_avg_fisher / old_selection_avg) - 1.0f) * 100.0f);
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  COMPARISON SUMMARY\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("                    Count   Avg Fisher   vs ALL   vs [10-54]\n");
    printf("────────────────────────────────────────────────────────────────\n");
    printf("ALL [0-63]          64      %.4f      100%%     +%.1f%%\n",
           all_subcarriers_avg, 
           ((all_subcarriers_avg / old_selection_avg) - 1.0f) * 100.0f);
    printf("Paper-based [10-54] 45      %.4f      %.1f%%    baseline\n",
           old_selection_avg,
           (old_selection_avg / all_subcarriers_avg) * 100.0f);
    printf("PCA-optimal [%d-%d]  %d      %.4f      %.1f%%    +%.1f%%\n",
           best_start, best_end - 1, best_end - best_start,
           best_avg_fisher,
           (best_avg_fisher / all_subcarriers_avg) * 100.0f,
           ((best_avg_fisher / old_selection_avg) - 1.0f) * 100.0f);
    
    if (best_avg_fisher > old_selection_avg * 1.1f) {
        printf("\n🎯 RECOMMENDATION: Use PCA-optimal [%d-%d]\n", best_start, best_end - 1);
        printf("   Improvement: +%.1f%% vs paper-based, +%.1f%% vs ALL\n",
               ((best_avg_fisher / old_selection_avg) - 1.0f) * 100.0f,
               ((best_avg_fisher / all_subcarriers_avg) - 1.0f) * 100.0f);
    } else if (old_selection_avg > all_subcarriers_avg * 1.1f) {
        printf("\n✅ Paper-based [10-54] is better than using ALL\n");
        printf("   But PCA-optimal [%d-%d] is still %.1f%% better\n",
               best_start, best_end - 1,
               ((best_avg_fisher / old_selection_avg) - 1.0f) * 100.0f);
    } else {
        printf("\n⚠️  ALL subcarriers perform similarly to selections\n");
        printf("   Consider using ALL [0-63] for simplicity\n");
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  TOP INDIVIDUAL SUBCARRIERS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Top 10 subcarriers by Fisher score:\n");
    for (int i = 0; i < 10; i++) {
        printf("  %d. SC%d (Fisher=%.4f, VarRatio=%.2fx)\n",
               i + 1, results[i].index, results[i].fisher_score, results[i].variance_ratio);
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  SUMMARY & RECOMMENDATIONS\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("1. PCA Components for 95%% variance: %d\n", 
           components_for_95pct > 0 ? components_for_95pct : PCA_COMPONENTS);
    printf("2. Optimal range: [%d-%d] (%.1f%% better than ALL)\n", 
           best_start, best_end - 1,
           ((best_avg_fisher / all_subcarriers_avg) - 1.0f) * 100.0f);
    printf("3. Paper-based [10-54]: %.1f%% of optimal quality\n", 
           (old_selection_avg / best_avg_fisher) * 100.0f);
    printf("4. Top subcarrier: SC%d (Fisher=%.4f, %.1fx better than avg)\n",
           results[0].index, results[0].fisher_score,
           results[0].fisher_score / all_subcarriers_avg);
    printf("5. Using ALL [0-63]: %.1f%% of optimal quality\n",
           (all_subcarriers_avg / best_avg_fisher) * 100.0f);
    
    // Cleanup
    for (int i = 0; i < NUM_SUBCARRIERS; i++) {
        free(cov_matrix[i]);
        free(work_matrix[i]);
    }
    free(cov_matrix);
    free(work_matrix);
    free(means);
    free(stats);
    free(results);
    free(eigenvalues);
    for (int i = 0; i < PCA_COMPONENTS; i++) {
        free(eigenvectors[i]);
    }
    free(eigenvectors);
    
    printf("\n");
}
