/*
 * ESPectre - Statistics Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "statistics.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "esp_log.h"

static const char *TAG = "Statistics";

int stats_buffer_init(stats_buffer_t *buffer, size_t size) {
    if (!buffer || size == 0) {
        ESP_LOGE(TAG, "stats_buffer_init: Invalid parameters");
        return -1;
    }
    
    buffer->data = (float*)malloc(size * sizeof(float));
    if (!buffer->data) {
        ESP_LOGE(TAG, "stats_buffer_init: Failed to allocate memory");
        return -1;
    }
    
    buffer->size = size;
    buffer->index = 0;
    buffer->count = 0;
    
    return 0;
}

void stats_buffer_free(stats_buffer_t *buffer) {
    if (buffer && buffer->data) {
        free(buffer->data);
        buffer->data = NULL;
        buffer->size = 0;
        buffer->index = 0;
        buffer->count = 0;
    }
}

void stats_buffer_add(stats_buffer_t *buffer, float value) {
    if (!buffer || !buffer->data) {
        ESP_LOGE(TAG, "stats_buffer_add: Invalid buffer");
        return;
    }
    
    buffer->data[buffer->index] = value;
    buffer->index = (buffer->index + 1) % buffer->size;
    if (buffer->count < buffer->size) {
        buffer->count++;
    }
}

void stats_buffer_analyze(const stats_buffer_t *buffer, stats_result_t *result) {
    if (!buffer || !result) {
        ESP_LOGE(TAG, "stats_buffer_analyze: NULL pointer");
        return;
    }
    
    if (buffer->count == 0) {
        result->min = result->max = result->mean = result->stddev = 0.0f;
        result->count = 0;
        return;
    }
    
    result->min = buffer->data[0];
    result->max = buffer->data[0];
    result->mean = 0.0f;
    
    for (size_t i = 0; i < buffer->count; i++) {
        float val = buffer->data[i];
        if (val < result->min) result->min = val;
        if (val > result->max) result->max = val;
        result->mean += val;
    }
    result->mean /= buffer->count;
    
    float variance = 0.0f;
    for (size_t i = 0; i < buffer->count; i++) {
        float diff = buffer->data[i] - result->mean;
        variance += diff * diff;
    }
    result->stddev = sqrtf(variance / buffer->count);
    result->count = buffer->count;
}

// Partition function for quickselect
static size_t partition(float *arr, size_t left, size_t right, size_t pivot_idx) {
    float pivot = arr[pivot_idx];
    
    // Move pivot to end
    float temp = arr[pivot_idx];
    arr[pivot_idx] = arr[right];
    arr[right] = temp;
    
    size_t store_idx = left;
    for (size_t i = left; i < right; i++) {
        if (arr[i] < pivot) {
            temp = arr[store_idx];
            arr[store_idx] = arr[i];
            arr[i] = temp;
            store_idx++;
        }
    }
    
    // Move pivot to final position
    temp = arr[right];
    arr[right] = arr[store_idx];
    arr[store_idx] = temp;
    
    return store_idx;
}

// Quickselect algorithm - finds k-th smallest element in-place
static float quickselect(float *arr, size_t left, size_t right, size_t k) {
    if (left == right) {
        return arr[left];
    }
    
    // Choose pivot (median-of-three for better performance)
    size_t mid = left + (right - left) / 2;
    size_t pivot_idx = mid;
    
    pivot_idx = partition(arr, left, right, pivot_idx);
    
    if (k == pivot_idx) {
        return arr[k];
    } else if (k < pivot_idx) {
        return quickselect(arr, left, pivot_idx - 1, k);
    } else {
        return quickselect(arr, pivot_idx + 1, right, k);
    }
}

float stats_buffer_percentile(const stats_buffer_t *buffer, float percentile) {
    if (!buffer || buffer->count == 0) {
        return 0.0f;
    }
    
    // Use static buffer to avoid malloc (reuse iqr_sort_buffer concept)
    static float percentile_buffer[100];  // Matches STATS_BUFFER_SIZE
    
    if (buffer->count > 100) {
        ESP_LOGE(TAG, "Buffer size exceeds static buffer capacity");
        return 0.0f;
    }
    
    // Copy data to working buffer
    memcpy(percentile_buffer, buffer->data, buffer->count * sizeof(float));
    
    // Calculate target index
    size_t k = (size_t)((percentile / 100.0f) * (buffer->count - 1));
    
    // Use quickselect to find k-th element in-place
    return quickselect(percentile_buffer, 0, buffer->count - 1, k);
}

float stats_calculate_recommended_threshold(const stats_buffer_t *buffer) {
    if (!buffer || buffer->count < 50) {
        return 0.0f;
    }
    
    float p50 = stats_buffer_percentile(buffer, 50.0f);
    float p75 = stats_buffer_percentile(buffer, 75.0f);
    
    return (p50 + p75) / 2.0f;
}
