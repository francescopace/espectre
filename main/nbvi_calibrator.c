/*
 * ESPectre - NBVI Calibrator Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "nbvi_calibrator.h"
#include "csi_features.h"
#include "csi_processor.h"
#include "espectre.h"
#include "esp_log.h"
#include "esp_err.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>

static const char *TAG = "NBVI";

// Number of subcarriers per CSI packet
#define NUM_SUBCARRIERS 64

// Number of subcarriers to select
#define SELECTED_SUBCARRIERS_COUNT 12

// ============================================================================
// INTERNAL STRUCTURES
// ============================================================================

// NBVI metrics for a single subcarrier
typedef struct {
    uint8_t subcarrier;     // Subcarrier index (0-63)
    float nbvi;             // NBVI weighted value
    float mean;             // Mean magnitude
    float std;              // Standard deviation
} nbvi_metrics_t;

// Window variance info for baseline detection
typedef struct {
    uint16_t start_idx;     // Start index in buffer
    float variance;         // Variance of this window
} window_variance_t;

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

/**
 * qsort comparator for float (ascending order)
 */
static int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < fb) return -1;
    if (fa > fb) return 1;
    return 0;
}

/**
 * qsort comparator for nbvi_metrics_t (ascending NBVI)
 */
static int compare_nbvi_metrics(const void *a, const void *b) {
    const nbvi_metrics_t *ma = (const nbvi_metrics_t*)a;
    const nbvi_metrics_t *mb = (const nbvi_metrics_t*)b;
    if (ma->nbvi < mb->nbvi) return -1;
    if (ma->nbvi > mb->nbvi) return 1;
    return 0;
}

/**
 * qsort comparator for window_variance_t (ascending variance)
 */
static int compare_window_variance(const void *a, const void *b) {
    const window_variance_t *wa = (const window_variance_t*)a;
    const window_variance_t *wb = (const window_variance_t*)b;
    if (wa->variance < wb->variance) return -1;
    if (wa->variance > wb->variance) return 1;
    return 0;
}

/**
 * Calculate percentile from sorted array
 * 
 * @param sorted_values Sorted array of values
 * @param n Number of values
 * @param p Percentile (0-100)
 * @return Percentile value
 */
static float calculate_percentile(const float *sorted_values, size_t n, uint8_t p) {
    if (n == 0) return 0.0f;
    if (n == 1) return sorted_values[0];
    
    // Linear interpolation between closest ranks
    float k = (n - 1) * p / 100.0f;
    size_t f = (size_t)k;
    size_t c = f + 1;
    
    if (c >= n) {
        return sorted_values[n - 1];
    }
    
    float d0 = sorted_values[f] * (c - k);
    float d1 = sorted_values[c] * (k - f);
    return d0 + d1;
}

/**
 * Calculate magnitude |H| = sqrt(I² + Q²) for a subcarrier
 */
static inline float calculate_magnitude(const int8_t *csi_data, uint8_t subcarrier) {
    size_t i_idx = subcarrier * 2;
    size_t q_idx = subcarrier * 2 + 1;
    
    float I = (float)csi_data[i_idx];
    float Q = (float)csi_data[q_idx];
    
    return sqrtf(I * I + Q * Q);
}

/**
 * Calculate spatial turbulence for a packet
 * (standard deviation of subcarrier magnitudes)
 */
static float calculate_spatial_turbulence(const float *magnitudes,
                                          const uint8_t *subcarriers,
                                          uint8_t num_subcarriers) {
    if (num_subcarriers == 0) return 0.0f;
    
    // Calculate mean
    float sum = 0.0f;
    for (uint8_t i = 0; i < num_subcarriers; i++) {
        sum += magnitudes[subcarriers[i]];
    }
    float mean = sum / num_subcarriers;
    
    // Calculate variance
    float sum_sq_diff = 0.0f;
    for (uint8_t i = 0; i < num_subcarriers; i++) {
        float diff = magnitudes[subcarriers[i]] - mean;
        sum_sq_diff += diff * diff;
    }
    float variance = sum_sq_diff / num_subcarriers;
    
    return sqrtf(variance);
}

// ============================================================================
// NBVI CALIBRATOR API IMPLEMENTATION
// ============================================================================

esp_err_t nbvi_calibrator_init(nbvi_calibrator_t *cal) {
    if (!cal) {
        ESP_LOGE(TAG, "nbvi_calibrator_init: NULL pointer");
        return ESP_ERR_INVALID_ARG;
    }
    
    // Set default parameters (from config.py)
    cal->buffer_size = 500;
    cal->window_size = 100;
    cal->window_step = 50;
    cal->percentile = 10;
    cal->alpha = 0.3f;
    cal->min_spacing = 3;
    cal->noise_gate_percentile = 10;
    
    // Allocate magnitude buffer: buffer_size × 64 floats
    size_t buffer_bytes = cal->buffer_size * NUM_SUBCARRIERS * sizeof(float);
    cal->magnitude_buffer = (float*)malloc(buffer_bytes);
    
    if (!cal->magnitude_buffer) {
        ESP_LOGE(TAG, "Failed to allocate magnitude buffer (%zu bytes)", buffer_bytes);
        return ESP_ERR_NO_MEM;
    }
    
    cal->buffer_count = 0;
    cal->calibrated = false;
    memset(cal->selected_band, 0, sizeof(cal->selected_band));
    
    ESP_LOGI(TAG, "Calibrator initialized (buffer: %zu bytes)", buffer_bytes);
    
    return ESP_OK;
}

bool nbvi_calibrator_add_packet(nbvi_calibrator_t *cal,
                                const int8_t *csi_data,
                                size_t csi_len) {
    if (!cal || !csi_data) {
        return false;
    }
    
    if (cal->buffer_count >= cal->buffer_size) {
        return true;  // Buffer already full
    }
    
    if (csi_len < 128) {
        ESP_LOGW(TAG, "CSI data too short: %zu bytes (need 128)", csi_len);
        return false;
    }
    
    // Calculate magnitudes for all 64 subcarriers
    float *packet_magnitudes = &cal->magnitude_buffer[cal->buffer_count * NUM_SUBCARRIERS];
    
    for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
        packet_magnitudes[sc] = calculate_magnitude(csi_data, sc);
    }
    
    cal->buffer_count++;
    
    return (cal->buffer_count >= cal->buffer_size);
}

/**
 * Find baseline window using percentile-based detection
 * 
 * @param cal Calibrator context
 * @param current_band Current subcarrier band for turbulence calculation
 * @param current_band_size Size of current band
 * @param out_window_start Output: start index of best baseline window
 * @return ESP_OK on success
 */
static esp_err_t find_baseline_window_percentile(nbvi_calibrator_t *cal,
                                                 const uint8_t *current_band,
                                                 uint8_t current_band_size,
                                                 uint16_t *out_window_start) {
    if (cal->buffer_count < cal->window_size) {
        ESP_LOGE(TAG, "Not enough packets for baseline detection (%d < %d)",
                 cal->buffer_count, cal->window_size);
        return ESP_FAIL;
    }
    
    // Calculate number of windows
    uint16_t num_windows = 0;
    for (uint16_t i = 0; i <= cal->buffer_count - cal->window_size; i += cal->window_step) {
        num_windows++;
    }
    
    if (num_windows == 0) {
        ESP_LOGE(TAG, "No windows to analyze");
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Analyzing %d windows (size=%d, step=%d)",
             num_windows, cal->window_size, cal->window_step);
    
    // Allocate window variance array
    window_variance_t *windows = (window_variance_t*)malloc(num_windows * sizeof(window_variance_t));
    if (!windows) {
        ESP_LOGE(TAG, "Failed to allocate window array");
        return ESP_ERR_NO_MEM;
    }
    
    // Allocate turbulence buffer for variance calculation
    float *turbulence_buffer = (float*)malloc(cal->window_size * sizeof(float));
    if (!turbulence_buffer) {
        ESP_LOGE(TAG, "Failed to allocate turbulence buffer");
        free(windows);
        return ESP_ERR_NO_MEM;
    }
    
    // Analyze each window
    uint16_t window_idx = 0;
    for (uint16_t i = 0; i <= cal->buffer_count - cal->window_size; i += cal->window_step) {
        // Calculate turbulence for each packet in window
        for (uint16_t j = 0; j < cal->window_size; j++) {
            uint16_t packet_idx = i + j;
            float *packet_magnitudes = &cal->magnitude_buffer[packet_idx * NUM_SUBCARRIERS];
            turbulence_buffer[j] = calculate_spatial_turbulence(packet_magnitudes,
                                                                current_band,
                                                                current_band_size);
        }
        
        // Calculate variance of turbulence (moving variance)
        float variance = calculate_variance_two_pass(turbulence_buffer, cal->window_size);
        
        windows[window_idx].start_idx = i;
        windows[window_idx].variance = variance;
        window_idx++;
    }
    
    free(turbulence_buffer);
    
    // Sort windows by variance
    qsort(windows, num_windows, sizeof(window_variance_t), compare_window_variance);
    
    // Calculate percentile threshold
    float *variances = (float*)malloc(num_windows * sizeof(float));
    if (!variances) {
        ESP_LOGE(TAG, "Failed to allocate variance array");
        free(windows);
        return ESP_ERR_NO_MEM;
    }
    
    for (uint16_t i = 0; i < num_windows; i++) {
        variances[i] = windows[i].variance;
    }
    
    float p_threshold = calculate_percentile(variances, num_windows, cal->percentile);
    free(variances);
    
    // Find best window (minimum variance below percentile)
    uint16_t best_window_idx = 0;
    float min_variance = windows[0].variance;
    
    for (uint16_t i = 0; i < num_windows; i++) {
        if (windows[i].variance <= p_threshold && windows[i].variance < min_variance) {
            min_variance = windows[i].variance;
            best_window_idx = i;
        }
    }
    
    *out_window_start = windows[best_window_idx].start_idx;
    
    ESP_LOGI(TAG, "Baseline window found:");
    ESP_LOGI(TAG, "  Variance: %.4f", min_variance);
    ESP_LOGI(TAG, "  p%d threshold: %.4f (adaptive)", cal->percentile, p_threshold);
    ESP_LOGI(TAG, "  Windows analyzed: %d", num_windows);
    
    free(windows);
    
    return ESP_OK;
}

/**
 * Calculate NBVI Weighted α=0.3 for a subcarrier
 * 
 * NBVI = 0.3 × (σ/μ²) + 0.7 × (σ/μ)
 */
static void calculate_nbvi_weighted(const float *magnitudes,
                                   size_t count,
                                   float alpha,
                                   nbvi_metrics_t *out_metrics) {
    if (count == 0) {
        out_metrics->nbvi = INFINITY;
        out_metrics->mean = 0.0f;
        out_metrics->std = 0.0f;
        return;
    }
    
    // Calculate mean
    float sum = 0.0f;
    for (size_t i = 0; i < count; i++) {
        sum += magnitudes[i];
    }
    float mean = sum / count;
    
    if (mean < 1e-6f) {
        out_metrics->nbvi = INFINITY;
        out_metrics->mean = mean;
        out_metrics->std = 0.0f;
        return;
    }
    
    // Calculate standard deviation using two-pass variance
    float variance = calculate_variance_two_pass(magnitudes, count);
    float std = sqrtf(variance);
    
    // NBVI Weighted α=0.3
    float cv = std / mean;                      // Coefficient of variation
    float nbvi_energy = std / (mean * mean);    // Energy normalization
    float nbvi_weighted = alpha * nbvi_energy + (1.0f - alpha) * cv;
    
    out_metrics->nbvi = nbvi_weighted;
    out_metrics->mean = mean;
    out_metrics->std = std;
}

/**
 * Apply Noise Gate: exclude weak subcarriers
 */
static uint8_t apply_noise_gate(nbvi_metrics_t *metrics,
                               uint8_t num_metrics,
                               uint8_t percentile) {
    if (num_metrics == 0) return 0;
    
    // Extract means and sort
    float *means = (float*)malloc(num_metrics * sizeof(float));
    if (!means) {
        ESP_LOGE(TAG, "Failed to allocate means array");
        return num_metrics;  // Return all if allocation fails
    }
    
    for (uint8_t i = 0; i < num_metrics; i++) {
        means[i] = metrics[i].mean;
    }
    
    qsort(means, num_metrics, sizeof(float), compare_float);
    
    float threshold = calculate_percentile(means, num_metrics, percentile);
    free(means);
    
    // Filter metrics
    uint8_t write_idx = 0;
    for (uint8_t read_idx = 0; read_idx < num_metrics; read_idx++) {
        if (metrics[read_idx].mean >= threshold) {
            if (write_idx != read_idx) {
                metrics[write_idx] = metrics[read_idx];
            }
            write_idx++;
        }
    }
    
    uint8_t excluded = num_metrics - write_idx;
    ESP_LOGI(TAG, "Noise Gate: %d subcarriers excluded (threshold: %.2f)",
             excluded, threshold);
    
    return write_idx;
}

/**
 * Select subcarriers with spectral spacing
 * 
 * Strategy:
 * - Top 5: Always include (highest priority)
 * - Remaining 7: Select with minimum spacing Δf≥min_spacing
 */
static void select_with_spacing(const nbvi_metrics_t *sorted_metrics,
                                uint8_t num_metrics,
                                uint8_t min_spacing,
                                uint8_t *output_band,
                                uint8_t *output_size) {
    if (num_metrics < SELECTED_SUBCARRIERS_COUNT) {
        ESP_LOGW(TAG, "Not enough subcarriers after filtering (%d < %d)",
                 num_metrics, SELECTED_SUBCARRIERS_COUNT);
        *output_size = 0;
        return;
    }
    
    uint8_t selected[SELECTED_SUBCARRIERS_COUNT];
    uint8_t selected_count = 0;
    
    // Phase 1: Top 5 absolute best
    for (uint8_t i = 0; i < 5 && i < num_metrics; i++) {
        selected[selected_count++] = sorted_metrics[i].subcarrier;
    }
    
    ESP_LOGI(TAG, "Top 5 selected: [%d, %d, %d, %d, %d]",
             selected[0], selected[1], selected[2], selected[3], selected[4]);
    
    // Phase 2: Remaining 7 with spacing
    for (uint8_t i = 5; i < num_metrics && selected_count < SELECTED_SUBCARRIERS_COUNT; i++) {
        uint8_t candidate = sorted_metrics[i].subcarrier;
        
        // Check spacing with already selected
        bool spacing_ok = true;
        for (uint8_t j = 0; j < selected_count; j++) {
            uint8_t dist = (candidate > selected[j]) ?
                          (candidate - selected[j]) : (selected[j] - candidate);
            if (dist < min_spacing) {
                spacing_ok = false;
                break;
            }
        }
        
        if (spacing_ok) {
            selected[selected_count++] = candidate;
        }
    }
    
    // If not enough with spacing, add best remaining regardless
    if (selected_count < SELECTED_SUBCARRIERS_COUNT) {
        for (uint8_t i = 5; i < num_metrics && selected_count < SELECTED_SUBCARRIERS_COUNT; i++) {
            uint8_t candidate = sorted_metrics[i].subcarrier;
            
            // Check if already selected
            bool already_selected = false;
            for (uint8_t j = 0; j < selected_count; j++) {
                if (selected[j] == candidate) {
                    already_selected = true;
                    break;
                }
            }
            
            if (!already_selected) {
                selected[selected_count++] = candidate;
            }
        }
    }
    
    // Sort output band
    for (uint8_t i = 0; i < selected_count - 1; i++) {
        for (uint8_t j = i + 1; j < selected_count; j++) {
            if (selected[i] > selected[j]) {
                uint8_t temp = selected[i];
                selected[i] = selected[j];
                selected[j] = temp;
            }
        }
    }
    
    memcpy(output_band, selected, selected_count);
    *output_size = selected_count;
    
    ESP_LOGI(TAG, "Selected %d subcarriers with spacing Δf≥%d",
             selected_count, min_spacing);
}

esp_err_t nbvi_calibrator_calibrate(nbvi_calibrator_t *cal,
                                    const uint8_t *current_band,
                                    uint8_t current_band_size,
                                    uint8_t *output_band,
                                    uint8_t *output_size) {
    if (!cal || !current_band || !output_band || !output_size) {
        ESP_LOGE(TAG, "nbvi_calibrator_calibrate: NULL pointer");
        return ESP_ERR_INVALID_ARG;
    }
    
    if (cal->buffer_count < cal->buffer_size) {
        ESP_LOGE(TAG, "Buffer not full (%d < %d)", cal->buffer_count, cal->buffer_size);
        return ESP_FAIL;
    }
    
    ESP_LOGI(TAG, "Starting calibration...");
    ESP_LOGI(TAG, "  Window size: %d packets", cal->window_size);
    ESP_LOGI(TAG, "  Step size: %d packets", cal->window_step);
    
    // Step 1: Find baseline window using percentile
    uint16_t baseline_start;
    esp_err_t err = find_baseline_window_percentile(cal, current_band, current_band_size,
                                                    &baseline_start);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to find baseline window");
        return err;
    }
    
    ESP_LOGI(TAG, "Using %d packets for calibration (starting at %d)",
             cal->window_size, baseline_start);
    
    // Step 2: Calculate NBVI for all 64 subcarriers
    nbvi_metrics_t *all_metrics = (nbvi_metrics_t*)malloc(NUM_SUBCARRIERS * sizeof(nbvi_metrics_t));
    if (!all_metrics) {
        ESP_LOGE(TAG, "Failed to allocate metrics array");
        return ESP_ERR_NO_MEM;
    }
    
    float *subcarrier_magnitudes = (float*)malloc(cal->window_size * sizeof(float));
    if (!subcarrier_magnitudes) {
        ESP_LOGE(TAG, "Failed to allocate subcarrier magnitudes");
        free(all_metrics);
        return ESP_ERR_NO_MEM;
    }
    
    for (uint8_t sc = 0; sc < NUM_SUBCARRIERS; sc++) {
        // Extract magnitude series for this subcarrier from baseline window
        for (uint16_t i = 0; i < cal->window_size; i++) {
            uint16_t packet_idx = baseline_start + i;
            subcarrier_magnitudes[i] = cal->magnitude_buffer[packet_idx * NUM_SUBCARRIERS + sc];
        }
        
        // Calculate NBVI
        all_metrics[sc].subcarrier = sc;
        calculate_nbvi_weighted(subcarrier_magnitudes, cal->window_size, cal->alpha,
                               &all_metrics[sc]);
    }
    
    free(subcarrier_magnitudes);
    
    // Step 3: Apply Noise Gate
    uint8_t filtered_count = apply_noise_gate(all_metrics, NUM_SUBCARRIERS,
                                             cal->noise_gate_percentile);
    
    if (filtered_count < SELECTED_SUBCARRIERS_COUNT) {
        ESP_LOGE(TAG, "Not enough subcarriers after Noise Gate (%d < %d)",
                 filtered_count, SELECTED_SUBCARRIERS_COUNT);
        free(all_metrics);
        return ESP_FAIL;
    }
    
    // Step 4: Sort by NBVI (ascending - lower is better)
    qsort(all_metrics, filtered_count, sizeof(nbvi_metrics_t), compare_nbvi_metrics);
    
    // Step 5: Select with spectral spacing
    select_with_spacing(all_metrics, filtered_count, cal->min_spacing,
                       output_band, output_size);
    
    if (*output_size != SELECTED_SUBCARRIERS_COUNT) {
        ESP_LOGE(TAG, "Invalid band size (%d != %d)", *output_size, SELECTED_SUBCARRIERS_COUNT);
        free(all_metrics);
        return ESP_FAIL;
    }
    
    // Calculate average metrics for selected band
    float avg_nbvi = 0.0f;
    float avg_mean = 0.0f;
    for (uint8_t i = 0; i < SELECTED_SUBCARRIERS_COUNT; i++) {
        for (uint8_t j = 0; j < filtered_count; j++) {
            if (all_metrics[j].subcarrier == output_band[i]) {
                avg_nbvi += all_metrics[j].nbvi;
                avg_mean += all_metrics[j].mean;
                break;
            }
        }
    }
    avg_nbvi /= SELECTED_SUBCARRIERS_COUNT;
    avg_mean /= SELECTED_SUBCARRIERS_COUNT;
    
    ESP_LOGI(TAG, "Calibration successful!");
    ESP_LOGI(TAG, "  Band: [%d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d, %d]",
             output_band[0], output_band[1], output_band[2], output_band[3],
             output_band[4], output_band[5], output_band[6], output_band[7],
             output_band[8], output_band[9], output_band[10], output_band[11]);
    ESP_LOGI(TAG, "  Average NBVI: %.6f", avg_nbvi);
    ESP_LOGI(TAG, "  Average magnitude: %.2f", avg_mean);
    
    // Store results
    memcpy(cal->selected_band, output_band, SELECTED_SUBCARRIERS_COUNT);
    cal->calibrated = true;
    
    free(all_metrics);
    
    return ESP_OK;
}

void nbvi_calibrator_free(nbvi_calibrator_t *cal) {
    if (cal && cal->magnitude_buffer) {
        free(cal->magnitude_buffer);
        cal->magnitude_buffer = NULL;
        cal->buffer_count = 0;
        ESP_LOGI(TAG, "Calibrator memory freed");
    }
}
