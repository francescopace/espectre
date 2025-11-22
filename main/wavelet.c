/*
 * ESPectre - Wavelet Transform Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "wavelet.h"
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "esp_log.h"

static const char *TAG = "Wavelet";

// Daubechies db4 coefficients (normalized)
// Low-pass decomposition filter (scaling function)
const float WAVELET_DB4_LP[WAVELET_DB4_LENGTH] = {
    -0.010597401785f,
     0.032883011667f,
     0.030841381836f,
    -0.187034811719f,
    -0.027983769417f,
     0.630880767930f,
     0.714846570553f,
     0.230377813309f
};

// High-pass decomposition filter (wavelet function)
const float WAVELET_DB4_HP[WAVELET_DB4_LENGTH] = {
    -0.230377813309f,
     0.714846570553f,
    -0.630880767930f,
    -0.027983769417f,
     0.187034811719f,
     0.030841381836f,
    -0.032883011667f,
    -0.010597401785f
};

// Low-pass reconstruction filter
const float WAVELET_DB4_LR[WAVELET_DB4_LENGTH] = {
     0.230377813309f,
     0.714846570553f,
     0.630880767930f,
    -0.027983769417f,
    -0.187034811719f,
     0.030841381836f,
     0.032883011667f,
    -0.010597401785f
};

// High-pass reconstruction filter
const float WAVELET_DB4_HR[WAVELET_DB4_LENGTH] = {
    -0.010597401785f,
    -0.032883011667f,
     0.030841381836f,
     0.187034811719f,
    -0.027983769417f,
    -0.630880767930f,
     0.714846570553f,
    -0.230377813309f
};

void wavelet_init(wavelet_state_t *state, int level, float threshold,
                  wavelet_threshold_method_t method) {
    if (!state) {
        ESP_LOGE(TAG, "wavelet_init: NULL state pointer");
        return;
    }
    
    if (level < 1 || level > WAVELET_MAX_LEVEL) {
        ESP_LOGW(TAG, "wavelet_init: level %d out of range, clamping to [1,%d]",
                 level, WAVELET_MAX_LEVEL);
        level = (level < 1) ? 1 : WAVELET_MAX_LEVEL;
    }
    
    memset(state->buffer, 0, sizeof(state->buffer));
    state->buffer_index = 0;
    state->buffer_count = 0;
    state->decomp_level = level;
    state->threshold = threshold;
    state->method = method;
    state->initialized = true;
    
    ESP_LOGD(TAG, "Wavelet initialized: level=%d, threshold=%.2f, method=%s",
             level, threshold, method == WAVELET_THRESH_SOFT ? "SOFT" : "HARD");
}

float wavelet_soft_threshold(float value, float threshold) {
    float abs_val = fabsf(value);
    if (abs_val <= threshold) {
        return 0.0f;
    }
    return (value > 0.0f ? 1.0f : -1.0f) * (abs_val - threshold);
}

float wavelet_hard_threshold(float value, float threshold) {
    return (fabsf(value) > threshold) ? value : 0.0f;
}

// Comparison function for qsort (for MAD calculation)
static int compare_float_abs(const void *a, const void *b) {
    float fa = fabsf(*(const float*)a);
    float fb = fabsf(*(const float*)b);
    return (fa > fb) - (fa < fb);
}

float wavelet_estimate_noise(const float *coeffs, size_t length) {
    if (!coeffs || length == 0) {
        return 1.0f; // Default threshold
    }
    
    // Create temporary array for sorting
    float *temp = (float*)malloc(length * sizeof(float));
    if (!temp) {
        ESP_LOGE(TAG, "wavelet_estimate_noise: malloc failed");
        return 1.0f;
    }
    
    memcpy(temp, coeffs, length * sizeof(float));
    qsort(temp, length, sizeof(float), compare_float_abs);
    
    // Calculate median of absolute values
    float median;
    if (length % 2 == 0) {
        median = (fabsf(temp[length/2 - 1]) + fabsf(temp[length/2])) / 2.0f;
    } else {
        median = fabsf(temp[length/2]);
    }
    
    free(temp);
    
    // MAD-based noise estimation: sigma = MAD / 0.6745
    return median / 0.6745f;
}

void wavelet_decompose_level(const float *input, size_t length,
                             float *approx, float *detail) {
    if (!input || !approx || !detail || length < WAVELET_DB4_LENGTH) {
        ESP_LOGE(TAG, "wavelet_decompose_level: invalid parameters");
        return;
    }
    
    size_t half_len = length / 2;
    
    // Convolution with downsampling by 2
    for (size_t i = 0; i < half_len; i++) {
        float sum_lp = 0.0f;
        float sum_hp = 0.0f;
        
        for (int j = 0; j < WAVELET_DB4_LENGTH; j++) {
            int idx = (2 * i + j) % length; // Circular boundary
            sum_lp += input[idx] * WAVELET_DB4_LP[j];
            sum_hp += input[idx] * WAVELET_DB4_HP[j];
        }
        
        approx[i] = sum_lp;
        detail[i] = sum_hp;
    }
}

void wavelet_reconstruct_level(const float *approx, const float *detail,
                               size_t length, float *output) {
    if (!approx || !detail || !output || length == 0) {
        ESP_LOGE(TAG, "wavelet_reconstruct_level: invalid parameters");
        return;
    }
    
    size_t full_len = length * 2;
    
    // Initialize output
    memset(output, 0, full_len * sizeof(float));
    
    // Upsampling and convolution
    for (size_t i = 0; i < length; i++) {
        for (int j = 0; j < WAVELET_DB4_LENGTH; j++) {
            int idx = (2 * i + j) % full_len;
            output[idx] += approx[i] * WAVELET_DB4_LR[j];
            output[idx] += detail[i] * WAVELET_DB4_HR[j];
        }
    }
}

int wavelet_denoise(const float *input, float *output, size_t length,
                    int level, float threshold, wavelet_threshold_method_t method) {
    if (!input || !output) {
        ESP_LOGE(TAG, "wavelet_denoise: NULL pointer");
        return -1;
    }
    
    if (length < (1 << level) * WAVELET_DB4_LENGTH) {
        ESP_LOGE(TAG, "wavelet_denoise: signal too short for level %d", level);
        return -1;
    }
    
    // Check if length is power of 2
    if ((length & (length - 1)) != 0) {
        ESP_LOGW(TAG, "wavelet_denoise: length %zu not power of 2, may cause artifacts", length);
    }
    
    // Allocate buffers for decomposition
    size_t max_size = length;
    
    // Overflow protection
    if (max_size == 0 || max_size > SIZE_MAX / sizeof(float)) {
        ESP_LOGE(TAG, "wavelet_denoise: requested size too large or zero");
        return -1;
    }

    float *approx = NULL;
    float *detail = NULL;
    float *temp = NULL;

    approx = (float*)calloc(max_size, sizeof(float));
    if (!approx) {
        ESP_LOGE(TAG, "wavelet_denoise: calloc failed for approx");
        return -1;
    }

    detail = (float*)calloc(max_size, sizeof(float));
    if (!detail) {
        ESP_LOGE(TAG, "wavelet_denoise: calloc failed for detail");
        free(approx);
        return -1;
    }

    temp = (float*)calloc(max_size, sizeof(float));
    if (!temp) {
        ESP_LOGE(TAG, "wavelet_denoise: calloc failed for temp");
        free(approx);
        free(detail);
        return -1;
    }
    
    // Copy input to working buffer
    memcpy(temp, input, length * sizeof(float));
    
    // Forward transform (decomposition)
    size_t current_len = length;
    for (int l = 0; l < level; l++) {
        wavelet_decompose_level(temp, current_len, approx, detail);
        
        // Apply thresholding to detail coefficients
        size_t half_len = current_len / 2;
        for (size_t i = 0; i < half_len; i++) {
            if (method == WAVELET_THRESH_SOFT) {
                detail[i] = wavelet_soft_threshold(detail[i], threshold);
            } else {
                detail[i] = wavelet_hard_threshold(detail[i], threshold);
            }
        }
        
        // Store thresholded details back
        memcpy(&temp[half_len], detail, half_len * sizeof(float));
        
        // Continue with approximation for next level
        memcpy(temp, approx, half_len * sizeof(float));
        current_len = half_len;
    }
    
    // Inverse transform (reconstruction)
    for (int l = level - 1; l >= 0; l--) {
        current_len = length >> l;
        size_t half_len = current_len / 2;
        
        // Get approximation and detail coefficients
        memcpy(approx, temp, half_len * sizeof(float));
        memcpy(detail, &temp[half_len], half_len * sizeof(float));
        
        // Reconstruct
        wavelet_reconstruct_level(approx, detail, half_len, temp);
    }
    
    // Copy result to output
    memcpy(output, temp, length * sizeof(float));
    
    free(approx);
    free(detail);
    free(temp);
    
    return 0;
}

float wavelet_denoise_sample(wavelet_state_t *state, float input) {
    if (!state || !state->initialized) {
        ESP_LOGE(TAG, "wavelet_denoise_sample: uninitialized state");
        return input;
    }
    
    // Add sample to circular buffer
    state->buffer[state->buffer_index] = input;
    state->buffer_index = (state->buffer_index + 1) % WAVELET_BUFFER_SIZE;
    
    if (state->buffer_count < WAVELET_BUFFER_SIZE) {
        state->buffer_count++;
        // Not enough samples yet, return input
        return input;
    }
    
    // Create linear buffer from circular buffer
    float linear_buffer[WAVELET_BUFFER_SIZE];
    size_t read_idx = state->buffer_index;
    for (size_t i = 0; i < WAVELET_BUFFER_SIZE; i++) {
        linear_buffer[i] = state->buffer[read_idx];
        read_idx = (read_idx + 1) % WAVELET_BUFFER_SIZE;
    }
    
    // Apply wavelet denoising
    float output_buffer[WAVELET_BUFFER_SIZE];
    int result = wavelet_denoise(linear_buffer, output_buffer, WAVELET_BUFFER_SIZE,
                                 state->decomp_level, state->threshold, state->method);
    
    if (result != 0) {
        ESP_LOGW(TAG, "wavelet_denoise_sample: denoising failed, returning input");
        return input;
    }
    
    // Return the middle sample (to minimize edge effects)
    return output_buffer[WAVELET_BUFFER_SIZE / 2];
}
