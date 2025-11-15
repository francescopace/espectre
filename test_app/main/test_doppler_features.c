/*
 * ESPectre - Doppler Spectrum Features Test
 * 
 * Evaluates Doppler spread spectrum features on real CSI data
 * to determine if they improve motion detection accuracy compared
 * to existing time/spatial domain features.
 * 
 * Features tested:
 * 1. Doppler Spread (S_D): Width of spectrum above threshold
 * 2. RMS Doppler Spread (R_D): Second moment (weighted std deviation)
 * 3. Peak Doppler (P_D): Maximum value in spectrum
 * 4. Spectral Centroid: Center of mass (weighted mean frequency)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "test_case_esp.h"
#include "real_csi_data.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "esp_dsp.h"

// Include CSI data arrays
#include "real_csi_arrays.inc"

// Doppler spectrum configuration
#define DOPPLER_WINDOW_SIZE 64      // Temporal window size (must be power of 2)
#define DOPPLER_FFT_SIZE 64         // FFT size (same as window)
#define DOPPLER_OVERLAP 32          // 50% overlap
#define DOPPLER_SAMPLING_RATE 50.0f // Estimated CSI packet rate (Hz)
#define DOPPLER_MAX_SHIFT 10.0f     // Maximum Doppler shift for human motion (Hz)
#define DOPPLER_THRESHOLD 0.1f      // Threshold factor for S_D (10% of peak)

#define NUM_DOPPLER_FEATURES 4

// Feature names
static const char* doppler_feature_names[] = {
    "doppler_spread",
    "rms_doppler_spread", 
    "peak_doppler",
    "spectral_centroid"
};

// Doppler features structure
typedef struct {
    float doppler_spread;      // S_D(λ): Width of spectrum above threshold
    float rms_doppler_spread;  // R_D(λ): RMS spread (second moment)
    float peak_doppler;        // P_D(λ): Peak value in spectrum
    float spectral_centroid;   // μ_D: Weighted mean frequency
} doppler_features_t;

// Circular buffer for temporal CSI amplitudes
typedef struct {
    float buffer[DOPPLER_WINDOW_SIZE];
    size_t index;
    size_t count;
    bool ready;
} amplitude_buffer_t;

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
} classification_metrics_t;

// Hanning window coefficients (precomputed for efficiency)
static float hanning_window[DOPPLER_WINDOW_SIZE];
static bool hanning_initialized = false;

// Initialize Hanning window
static void init_hanning_window(void) {
    if (hanning_initialized) return;
    
    for (int i = 0; i < DOPPLER_WINDOW_SIZE; i++) {
        hanning_window[i] = 0.5f * (1.0f - cosf(2.0f * M_PI * i / (DOPPLER_WINDOW_SIZE - 1)));
    }
    hanning_initialized = true;
}

// Calculate average amplitude from CSI packet (I/Q pairs)
static float calculate_csi_amplitude(const int8_t *csi_data, size_t len) {
    float total_amplitude = 0.0f;
    int num_subcarriers = len / 2;  // Each subcarrier has I and Q
    
    for (int i = 0; i < num_subcarriers; i++) {
        float I = (float)csi_data[2 * i];
        float Q = (float)csi_data[2 * i + 1];
        total_amplitude += sqrtf(I * I + Q * Q);
    }
    
    return total_amplitude / num_subcarriers;
}

// Add amplitude sample to circular buffer
static void amplitude_buffer_add(amplitude_buffer_t *buf, float amplitude) {
    buf->buffer[buf->index] = amplitude;
    buf->index = (buf->index + 1) % DOPPLER_WINDOW_SIZE;
    
    if (buf->count < DOPPLER_WINDOW_SIZE) {
        buf->count++;
    }
    
    buf->ready = (buf->count == DOPPLER_WINDOW_SIZE);
}

// Extract Doppler features from amplitude buffer using FFT
static bool extract_doppler_features(amplitude_buffer_t *buf, doppler_features_t *features) {
    if (!buf->ready) return false;
    
    // Allocate FFT buffers (complex format: real, imag, real, imag, ...)
    float *fft_input = malloc(DOPPLER_FFT_SIZE * 2 * sizeof(float));
    float *power_spectrum = malloc((DOPPLER_FFT_SIZE / 2) * sizeof(float));
    
    if (!fft_input || !power_spectrum) {
        free(fft_input);
        free(power_spectrum);
        return false;
    }
    
    // Copy buffer in circular order and apply Hanning window
    for (int i = 0; i < DOPPLER_WINDOW_SIZE; i++) {
        int buf_idx = (buf->index + i) % DOPPLER_WINDOW_SIZE;
        fft_input[i * 2] = buf->buffer[buf_idx] * hanning_window[i];  // Real part
        fft_input[i * 2 + 1] = 0.0f;  // Imaginary part
    }
    
    // Perform FFT using ESP-DSP
    dsps_fft2r_fc32(fft_input, DOPPLER_FFT_SIZE);
    dsps_bit_rev_fc32(fft_input, DOPPLER_FFT_SIZE);
    
    // Calculate power spectrum (magnitude squared)
    // Only use positive frequencies (first half)
    float max_power = 0.0f;
    float total_power = 0.0f;
    
    for (int i = 0; i < DOPPLER_FFT_SIZE / 2; i++) {
        float real = fft_input[i * 2];
        float imag = fft_input[i * 2 + 1];
        float magnitude = sqrtf(real * real + imag * imag);
        power_spectrum[i] = magnitude;
        total_power += magnitude;
        
        if (magnitude > max_power) {
            max_power = magnitude;
        }
    }
    
    // Avoid division by zero
    if (total_power < 1e-6f) {
        free(fft_input);
        free(power_spectrum);
        return false;
    }
    
    // Calculate frequency bins
    float freq_resolution = DOPPLER_SAMPLING_RATE / DOPPLER_FFT_SIZE;
    
    // Feature 1: Doppler Spread (S_D) - Width of spectrum above threshold
    float threshold = max_power * DOPPLER_THRESHOLD;
    int count_above_threshold = 0;
    
    for (int i = 0; i < DOPPLER_FFT_SIZE / 2; i++) {
        if (power_spectrum[i] > threshold) {
            count_above_threshold++;
        }
    }
    features->doppler_spread = (float)count_above_threshold * freq_resolution;
    
    // Feature 2 & 4: Calculate spectral centroid (weighted mean frequency)
    float weighted_freq_sum = 0.0f;
    
    for (int i = 0; i < DOPPLER_FFT_SIZE / 2; i++) {
        float freq = i * freq_resolution;
        weighted_freq_sum += freq * power_spectrum[i];
    }
    features->spectral_centroid = weighted_freq_sum / total_power;
    
    // Feature 2: RMS Doppler Spread (R_D) - Second moment (weighted std deviation)
    float variance_sum = 0.0f;
    
    for (int i = 0; i < DOPPLER_FFT_SIZE / 2; i++) {
        float freq = i * freq_resolution;
        float diff = freq - features->spectral_centroid;
        variance_sum += power_spectrum[i] * diff * diff;
    }
    features->rms_doppler_spread = sqrtf(variance_sum / total_power);
    
    // Feature 3: Peak Doppler (P_D) - Maximum value in spectrum
    features->peak_doppler = max_power;
    
    free(fft_input);
    free(power_spectrum);
    return true;
}

// Find optimal threshold using Otsu's method
static float find_optimal_threshold_otsu(const float *baseline_values, int baseline_count,
                                         const float *movement_values, int movement_count) {
    int total_count = baseline_count + movement_count;
    float *all_values = malloc(total_count * sizeof(float));
    if (!all_values) return 0.0f;
    
    memcpy(all_values, baseline_values, baseline_count * sizeof(float));
    memcpy(all_values + baseline_count, movement_values, movement_count * sizeof(float));
    
    // Find min and max
    float min_val = all_values[0];
    float max_val = all_values[0];
    for (int i = 1; i < total_count; i++) {
        if (all_values[i] < min_val) min_val = all_values[i];
        if (all_values[i] > max_val) max_val = all_values[i];
    }
    
    // Try different thresholds and find the one that maximizes between-class variance
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

// Calculate classification metrics
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
}

TEST_CASE_ESP(doppler_features_classification_performance, "[doppler][classification]")
{
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════╗\n");
    printf("║   DOPPLER SPECTRUM FEATURES TEST                      ║\n");
    printf("║   Frequency-domain features for motion detection     ║\n");
    printf("╚═══════════════════════════════════════════════════════╝\n");
    printf("\n");
    
    printf("Configuration:\n");
    printf("  Window Size: %d samples\n", DOPPLER_WINDOW_SIZE);
    printf("  FFT Size: %d points\n", DOPPLER_FFT_SIZE);
    printf("  Overlap: %d samples (%.0f%%)\n", DOPPLER_OVERLAP, 
           (float)DOPPLER_OVERLAP / DOPPLER_WINDOW_SIZE * 100.0f);
    printf("  Sampling Rate: %.1f Hz\n", DOPPLER_SAMPLING_RATE);
    printf("  Frequency Resolution: %.2f Hz\n", DOPPLER_SAMPLING_RATE / DOPPLER_FFT_SIZE);
    printf("  Max Doppler Shift: ±%.1f Hz\n", DOPPLER_MAX_SHIFT);
    printf("  Threshold Factor: %.1f%%\n", DOPPLER_THRESHOLD * 100.0f);
    printf("\n");
    
    // Initialize Hanning window and ESP-DSP
    init_hanning_window();
    dsps_fft2r_init_fc32(NULL, DOPPLER_FFT_SIZE);
    
    // Allocate storage for Doppler feature values
    float **baseline_doppler = malloc(NUM_DOPPLER_FEATURES * sizeof(float*));
    float **movement_doppler = malloc(NUM_DOPPLER_FEATURES * sizeof(float*));
    
    if (!baseline_doppler || !movement_doppler) {
        printf("ERROR: Failed to allocate feature arrays\n");
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    // Estimate number of feature vectors we'll generate
    // With 50% overlap, we get approximately: (num_packets - window_size) / overlap + 1
    int max_vectors = (num_baseline > num_movement ? num_baseline : num_movement) / DOPPLER_OVERLAP + 10;
    
    for (int f = 0; f < NUM_DOPPLER_FEATURES; f++) {
        baseline_doppler[f] = malloc(max_vectors * sizeof(float));
        movement_doppler[f] = malloc(max_vectors * sizeof(float));
        
        if (!baseline_doppler[f] || !movement_doppler[f]) {
            printf("ERROR: Failed to allocate feature storage\n");
            TEST_FAIL_MESSAGE("Memory allocation failed");
            return;
        }
    }
    
    printf("Processing baseline packets (%d packets)...\n", num_baseline);
    
    // Process baseline packets
    amplitude_buffer_t baseline_buffer = {0};
    int baseline_feature_count = 0;
    
    for (int p = 0; p < num_baseline; p++) {
        // Calculate average amplitude for this packet
        float amplitude = calculate_csi_amplitude(baseline_packets[p], 128);
        
        // Add to buffer
        amplitude_buffer_add(&baseline_buffer, amplitude);
        
        // Extract features when buffer is ready (with overlap)
        if (baseline_buffer.ready && (p % DOPPLER_OVERLAP == 0 || p == num_baseline - 1)) {
            doppler_features_t features;
            if (extract_doppler_features(&baseline_buffer, &features)) {
                baseline_doppler[0][baseline_feature_count] = features.doppler_spread;
                baseline_doppler[1][baseline_feature_count] = features.rms_doppler_spread;
                baseline_doppler[2][baseline_feature_count] = features.peak_doppler;
                baseline_doppler[3][baseline_feature_count] = features.spectral_centroid;
                baseline_feature_count++;
            }
        }
    }
    
    printf("  Generated %d Doppler feature vectors\n", baseline_feature_count);
    
    printf("Processing movement packets (%d packets)...\n", num_movement);
    
    // Process movement packets
    amplitude_buffer_t movement_buffer = {0};
    int movement_feature_count = 0;
    
    for (int p = 0; p < num_movement; p++) {
        // Calculate average amplitude for this packet
        float amplitude = calculate_csi_amplitude(movement_packets[p], 128);
        
        // Add to buffer
        amplitude_buffer_add(&movement_buffer, amplitude);
        
        // Extract features when buffer is ready (with overlap)
        if (movement_buffer.ready && (p % DOPPLER_OVERLAP == 0 || p == num_movement - 1)) {
            doppler_features_t features;
            if (extract_doppler_features(&movement_buffer, &features)) {
                movement_doppler[0][movement_feature_count] = features.doppler_spread;
                movement_doppler[1][movement_feature_count] = features.rms_doppler_spread;
                movement_doppler[2][movement_feature_count] = features.peak_doppler;
                movement_doppler[3][movement_feature_count] = features.spectral_centroid;
                movement_feature_count++;
            }
        }
    }
    
    printf("  Generated %d Doppler feature vectors\n", movement_feature_count);
    printf("\n");
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  DOPPLER FEATURES CLASSIFICATION PERFORMANCE\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Allocate metrics array
    classification_metrics_t *metrics = malloc(NUM_DOPPLER_FEATURES * sizeof(classification_metrics_t));
    if (!metrics) {
        printf("ERROR: Failed to allocate metrics array\n");
        TEST_FAIL_MESSAGE("Memory allocation failed");
        return;
    }
    
    // Evaluate each Doppler feature
    for (int f = 0; f < NUM_DOPPLER_FEATURES; f++) {
        metrics[f].feature_idx = f;
        metrics[f].name = doppler_feature_names[f];
        metrics[f].true_positives = 0;
        metrics[f].true_negatives = 0;
        metrics[f].false_positives = 0;
        metrics[f].false_negatives = 0;
        
        // Find optimal threshold
        metrics[f].threshold = find_optimal_threshold_otsu(
            baseline_doppler[f], baseline_feature_count,
            movement_doppler[f], movement_feature_count
        );
        
        // Classify baseline samples
        for (int i = 0; i < baseline_feature_count; i++) {
            if (baseline_doppler[f][i] >= metrics[f].threshold) {
                metrics[f].false_positives++;
            } else {
                metrics[f].true_negatives++;
            }
        }
        
        // Classify movement samples
        for (int i = 0; i < movement_feature_count; i++) {
            if (movement_doppler[f][i] >= metrics[f].threshold) {
                metrics[f].true_positives++;
            } else {
                metrics[f].false_negatives++;
            }
        }
        
        // Calculate metrics
        calculate_metrics(&metrics[f], baseline_feature_count, movement_feature_count);
        
        // Print results
        printf("Feature: %s\n", metrics[f].name);
        printf("  Threshold: %.4f\n", metrics[f].threshold);
        printf("  Accuracy: %.2f%%\n", metrics[f].accuracy);
        printf("  Precision: %.2f%% | Recall: %.2f%% | F1-Score: %.2f%%\n",
               metrics[f].precision, metrics[f].recall, metrics[f].f1_score);
        printf("  True Positives: %d/%d | True Negatives: %d/%d\n",
               metrics[f].true_positives, movement_feature_count,
               metrics[f].true_negatives, baseline_feature_count);
        printf("  False Positives: %d/%d | False Negatives: %d/%d\n",
               metrics[f].false_positives, baseline_feature_count,
               metrics[f].false_negatives, movement_feature_count);
        printf("  Specificity: %.2f%%\n", metrics[f].specificity);
        printf("\n");
    }
    
    printf("═══════════════════════════════════════════════════════\n");
    printf("  RANKING BY ACCURACY\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    // Sort by accuracy
    classification_metrics_t *sorted = malloc(NUM_DOPPLER_FEATURES * sizeof(classification_metrics_t));
    memcpy(sorted, metrics, NUM_DOPPLER_FEATURES * sizeof(classification_metrics_t));
    
    for (int i = 0; i < NUM_DOPPLER_FEATURES - 1; i++) {
        for (int j = i + 1; j < NUM_DOPPLER_FEATURES; j++) {
            if (sorted[j].accuracy > sorted[i].accuracy) {
                classification_metrics_t temp = sorted[i];
                sorted[i] = sorted[j];
                sorted[j] = temp;
            }
        }
    }
    
    printf("Rank  Feature                      Accuracy   F1-Score   FP+FN\n");
    printf("────────────────────────────────────────────────────────────────\n");
    for (int i = 0; i < NUM_DOPPLER_FEATURES; i++) {
        int total_errors = sorted[i].false_positives + sorted[i].false_negatives;
        printf("%2d    %-25s  %6.2f%%    %6.2f%%    %4d\n",
               i + 1, sorted[i].name,
               sorted[i].accuracy,
               sorted[i].f1_score,
               total_errors);
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  COMPARISON WITH EXISTING FEATURES\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    printf("Best Existing Feature: spatial_gradient (63.18%% accuracy)\n");
    printf("Best Doppler Feature: %s (%.2f%% accuracy)\n", 
           sorted[0].name, sorted[0].accuracy);
    
    float improvement = sorted[0].accuracy - 63.18f;
    if (improvement > 0) {
        printf("\n✅ Improvement: +%.2f percentage points\n", improvement);
        printf("   Doppler features show better discrimination!\n");
    } else {
        printf("\n⚠️  Difference: %.2f percentage points\n", improvement);
        printf("   Doppler features do not improve over existing features.\n");
    }
    
    printf("\n");
    printf("═══════════════════════════════════════════════════════\n");
    printf("  RECOMMENDATION\n");
    printf("═══════════════════════════════════════════════════════\n\n");
    
    if (sorted[0].accuracy > 70.0f) {
        printf("✅ INTEGRATE: Doppler features show excellent performance (>70%%).\n");
        printf("   Recommend adding to the main system.\n");
    } else if (sorted[0].accuracy > 65.0f) {
        printf("⚠️  CONSIDER: Doppler features show moderate improvement.\n");
        printf("   May be worth integrating for marginal gains.\n");
    } else {
        printf("❌ SKIP: Doppler features do not significantly improve detection.\n");
        printf("   Focus on optimizing existing features instead.\n");
    }
    
    printf("\n");
    
    // Cleanup
    for (int f = 0; f < NUM_DOPPLER_FEATURES; f++) {
        free(baseline_doppler[f]);
        free(movement_doppler[f]);
    }
    free(baseline_doppler);
    free(movement_doppler);
    free(metrics);
    free(sorted);
    
    dsps_fft2r_deinit_fc32();
}
