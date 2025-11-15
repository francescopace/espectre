/*
 * ESPectre - Signal Filtering Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "filters.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_log.h"

static const char *TAG = "Filters";

// Pre-computed Savitzky-Golay coefficients (window=5, poly=2)
static const float savgol_coeffs_5_2[] = {-0.0857f, 0.3429f, 0.4857f, 0.3429f, -0.0857f};

// qsort comparator for float
static int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

void butterworth_init(butterworth_filter_t *filter) {
    if (!filter) {
        ESP_LOGE(TAG, "butterworth_init: NULL filter pointer");
        return;
    }
    
    // Numerator coefficients (b) - pre-computed for 4th order, 8Hz cutoff, 100Hz sampling
    // Calculated using scipy.signal.butter(4, 8, 'low', fs=100)
    filter->b[0] = 0.00482434f;
    filter->b[1] = 0.01929736f;
    filter->b[2] = 0.02894604f;
    filter->b[3] = 0.01929736f;
    filter->b[4] = 0.00482434f;
    
    // Denominator coefficients (a)
    filter->a[0] = 1.0f;
    filter->a[1] = -2.36951301f;
    filter->a[2] = 2.31398841f;
    filter->a[3] = -1.05466541f;
    filter->a[4] = 0.18737949f;
    
    memset(filter->x, 0, sizeof(filter->x));
    memset(filter->y, 0, sizeof(filter->y));
    filter->initialized = true;
}

float butterworth_filter(butterworth_filter_t *filter, float input) {
    if (!filter) {
        ESP_LOGE(TAG, "butterworth_filter: NULL filter pointer");
        return input;
    }
    
    if (!filter->initialized) {
        butterworth_init(filter);
    }
    
    // Validate filter order is within bounds
    if (BUTTERWORTH_ORDER > 4 || BUTTERWORTH_ORDER < 1) {
        ESP_LOGE(TAG, "Invalid Butterworth filter order: %d", BUTTERWORTH_ORDER);
        return input;
    }
    
    // Shift input history (newest at index 0)
    for (int i = BUTTERWORTH_ORDER; i > 0; i--) {
        filter->x[i] = filter->x[i-1];
    }
    filter->x[0] = input;
    
    // Calculate output using Direct Form II
    float output = 0.0f;
    for (int i = 0; i <= BUTTERWORTH_ORDER; i++) {
        output += filter->b[i] * filter->x[i];
    }
    for (int i = 1; i <= BUTTERWORTH_ORDER; i++) {
        output -= filter->a[i] * filter->y[i];
    }
    
    // Shift output history
    for (int i = BUTTERWORTH_ORDER; i > 0; i--) {
        filter->y[i] = filter->y[i-1];
    }
    filter->y[0] = output;
    
    return output;
}

float hampel_filter(const float *window, size_t window_size, 
                   float current_value, float threshold) {
    if (!window || window_size < 3) {
        return current_value;
    }
    
    // Create sorted copy for median calculation
    float *sorted = (float*)malloc(window_size * sizeof(float));
    if (!sorted) {
        ESP_LOGE(TAG, "hampel_filter: Failed to allocate memory");
        return current_value;
    }
    
    memcpy(sorted, window, window_size * sizeof(float));
    qsort(sorted, window_size, sizeof(float), compare_float);
    
    // Calculate median
    float median = (window_size % 2 == 1) ? 
                   sorted[window_size / 2] : 
                   (sorted[window_size / 2 - 1] + sorted[window_size / 2]) / 2.0f;
    
    // Calculate absolute deviations
    float *abs_deviations = (float*)malloc(window_size * sizeof(float));
    if (!abs_deviations) {
        ESP_LOGE(TAG, "hampel_filter: Failed to allocate memory for deviations");
        free(sorted);
        return current_value;
    }
    
    for (size_t i = 0; i < window_size; i++) {
        abs_deviations[i] = fabsf(window[i] - median);
    }
    qsort(abs_deviations, window_size, sizeof(float), compare_float);
    
    // Calculate MAD (Median Absolute Deviation)
    float mad = (window_size % 2 == 1) ? 
                abs_deviations[window_size / 2] : 
                (abs_deviations[window_size / 2 - 1] + abs_deviations[window_size / 2]) / 2.0f;
    
    free(abs_deviations);
    free(sorted);
    
    // Scale MAD to approximate standard deviation
    float mad_scaled = MAD_SCALE_FACTOR * mad;
    
    float deviation = fabsf(current_value - median);
    float threshold_value = threshold * mad_scaled;
    
    // Replace outliers with median
    if (deviation > threshold_value) {
        return median;
    }
    
    return current_value;
}

float savitzky_golay_filter(const float *window, size_t window_size) {
    if (!window) {
        ESP_LOGE(TAG, "savitzky_golay_filter: NULL window pointer");
        return 0.0f;
    }
    
    // If window size doesn't match, fall back to simple average
    if (window_size != SAVGOL_WINDOW_SIZE) {
        float sum = 0.0f;
        for (size_t i = 0; i < window_size; i++) {
            sum += window[i];
        }
        return sum / window_size;
    }
    
    // Apply pre-computed coefficients
    float result = 0.0f;
    for (size_t i = 0; i < window_size; i++) {
        result += window[i] * savgol_coeffs_5_2[i];
    }
    
    return result;
}


void filter_buffer_init(filter_buffer_t *fb) {
    if (!fb) {
        ESP_LOGE(TAG, "filter_buffer_init: NULL buffer pointer");
        return;
    }
    
    memset(fb->data, 0, sizeof(fb->data));
    fb->index = 0;
    fb->count = 0;
}

void filter_buffer_add(filter_buffer_t *fb, float value) {
    if (!fb) {
        ESP_LOGE(TAG, "filter_buffer_add: NULL buffer pointer");
        return;
    }
    
    fb->data[fb->index] = value;
    fb->index = (fb->index + 1) % SAVGOL_WINDOW_SIZE;
    if (fb->count < SAVGOL_WINDOW_SIZE) {
        fb->count++;
    }
}

void filter_buffer_get_window(const filter_buffer_t *fb, float *window, 
                              size_t window_capacity, size_t *size) {
    // Validate all input parameters
    if (!fb || !window) {
        ESP_LOGE(TAG, "filter_buffer_get_window: NULL pointer (fb=%p, window=%p)", 
                 (void*)fb, (void*)window);
        if (size) *size = 0;
        return;
    }
    
    if (!size) {
        ESP_LOGE(TAG, "filter_buffer_get_window: size parameter is NULL");
        return;
    }
    
    // Validate window_capacity
    if (window_capacity < SAVGOL_WINDOW_SIZE) {
        ESP_LOGE(TAG, "filter_buffer_get_window: window_capacity (%zu) < SAVGOL_WINDOW_SIZE (%d)", 
                 window_capacity, SAVGOL_WINDOW_SIZE);
        *size = 0;
        return;
    }
    
    // Validate fb->count is within bounds
    if (fb->count > SAVGOL_WINDOW_SIZE) {
        ESP_LOGE(TAG, "filter_buffer_get_window: fb->count (%zu) exceeds SAVGOL_WINDOW_SIZE (%d)", 
                 fb->count, SAVGOL_WINDOW_SIZE);
        *size = 0;
        return;
    }
    
    // Validate fb->index is within bounds
    if (fb->index >= SAVGOL_WINDOW_SIZE) {
        ESP_LOGE(TAG, "filter_buffer_get_window: fb->index (%zu) exceeds SAVGOL_WINDOW_SIZE (%d)", 
                 fb->index, SAVGOL_WINDOW_SIZE);
        *size = 0;
        return;
    }
    
    *size = fb->count;
    if (fb->count < SAVGOL_WINDOW_SIZE) {
        // Buffer not full yet - copy what we have
        size_t copy_size = (fb->count < window_capacity) ? fb->count : window_capacity;
        memcpy(window, fb->data, copy_size * sizeof(float));
        *size = copy_size;
    } else {
        // Buffer is full - copy in chronological order
        size_t start = fb->index;
        size_t copy_size = (SAVGOL_WINDOW_SIZE < window_capacity) ? SAVGOL_WINDOW_SIZE : window_capacity;
        for (size_t i = 0; i < copy_size; i++) {
            window[i] = fb->data[(start + i) % SAVGOL_WINDOW_SIZE];
        }
        *size = copy_size;
    }
}

float apply_filter_pipeline(float raw_value, 
                            const filter_config_t *config,
                            butterworth_filter_t *butterworth,
                            wavelet_state_t *wavelet,
                            filter_buffer_t *buffer) {
    if (!config || !butterworth || !wavelet || !buffer) {
        ESP_LOGE(TAG, "apply_filter_pipeline: NULL pointer in parameters");
        return raw_value;
    }
    
    float filtered_value = raw_value;
    
    // Step 1: Butterworth low-pass removes high frequency noise (>8Hz)
    if (config->butterworth_enabled) {
        filtered_value = butterworth_filter(butterworth, filtered_value);
    }
    
    // Step 2: Wavelet denoising removes low frequency noise (NEW)
    // Based on research: "Location intelligence system" (ESP32 + db4)
    // and "CSI-HC" (Butterworth + Sym8 combination)
    if (config->wavelet_enabled) {
        filtered_value = wavelet_denoise_sample(wavelet, filtered_value);
    }
    
    // Add to buffer for windowed operations
    filter_buffer_add(buffer, filtered_value);
    
    float window[SAVGOL_WINDOW_SIZE];
    size_t window_size;
    filter_buffer_get_window(buffer, window, SAVGOL_WINDOW_SIZE, &window_size);
    
    // Step 3: Hampel filter catches outliers
    if (config->hampel_enabled && window_size >= 3) {
        filtered_value = hampel_filter(window, window_size, filtered_value, 
                                      config->hampel_threshold);
    }
    
    // Step 4: Savitzky-Golay smooths the signal while preserving shape
    if (config->savgol_enabled && window_size == SAVGOL_WINDOW_SIZE) {
        // Update window with Hampel-filtered value if Hampel was applied
        if (config->hampel_enabled) {
            window[window_size - 1] = filtered_value;
        }
        filtered_value = savitzky_golay_filter(window, window_size);
    }
    
    return filtered_value;
}
