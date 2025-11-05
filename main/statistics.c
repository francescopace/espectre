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

// Comparison function for qsort
static int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

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

float stats_buffer_percentile(const stats_buffer_t *buffer, float percentile) {
    if (!buffer || buffer->count == 0) {
        return 0.0f;
    }
    
    float *sorted = (float*)malloc(buffer->count * sizeof(float));
    if (!sorted) {
        ESP_LOGE(TAG, "stats_buffer_percentile: Failed to allocate memory");
        return 0.0f;
    }
    
    memcpy(sorted, buffer->data, buffer->count * sizeof(float));
    qsort(sorted, buffer->count, sizeof(float), compare_float);
    
    size_t index = (size_t)((percentile / 100.0f) * (buffer->count - 1));
    float result = sorted[index];
    
    free(sorted);
    return result;
}

float stats_calculate_recommended_threshold(const stats_buffer_t *buffer) {
    if (!buffer || buffer->count < 50) {
        return 0.0f;
    }
    
    float p50 = stats_buffer_percentile(buffer, 50.0f);
    float p75 = stats_buffer_percentile(buffer, 75.0f);
    
    return (p50 + p75) / 2.0f;
}
