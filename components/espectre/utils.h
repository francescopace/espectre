/*
 * ESPectre - Utility Functions
 * 
 * Shared utility functions used across multiple modules.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cstdarg>
#include <cstdio>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

/**
 * Calculate variance using two-pass algorithm (numerically stable)
 * 
 * Two-pass algorithm: variance = sum((x - mean)^2) / n
 * More stable than single-pass E[X²] - E[X]² for float32 arithmetic.
 * 
 * @param values Array of float values
 * @param n Number of values
 * @return Variance (0.0 if n == 0)
 */
inline float calculate_variance_two_pass(const float *values, size_t n) {
    if (n == 0 || !values) {
        return 0.0f;
    }
    
    // First pass: calculate mean
    float mean = 0.0f;
    for (size_t i = 0; i < n; i++) {
        mean += values[i];
    }
    mean /= n;
    
    // Second pass: calculate variance
    float variance = 0.0f;
    for (size_t i = 0; i < n; i++) {
        float diff = values[i] - mean;
        variance += diff * diff;
    }
    variance /= n;
    
    return variance;
}

/**
 * Compare two float values for qsort
 * 
 * @param a Pointer to first float
 * @param b Pointer to second float
 * @return -1 if a < b, 0 if a == b, 1 if a > b
 */
inline int compare_float(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

/**
 * Compare two int8_t values for qsort
 * 
 * @param a Pointer to first int8_t
 * @param b Pointer to second int8_t
 * @return Difference between values
 */
inline int compare_int8(const void *a, const void *b) {
    return (*(const int8_t*)a - *(const int8_t*)b);
}

/**
 * Compare absolute values of two floats for qsort
 * 
 * @param a Pointer to first float
 * @param b Pointer to second float
 * @return -1 if |a| < |b|, 0 if |a| == |b|, 1 if |a| > |b|
 */
inline int compare_float_abs(const void *a, const void *b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    if (fa < 0) fa = -fa;
    if (fb < 0) fb = -fb;
    return (fa > fb) - (fa < fb);
}

/**
 * Create and log a progress bar with optional metrics
 * 
 * @param tag Log tag (e.g., TAG)
 * @param progress Progress value (0.0 to 1.0+)
 * @param width Bar width in characters (default: 20)
 * @param threshold_pos Optional threshold marker position (-1 = no threshold marker)
 * @param format Optional format string for additional text after the bar (can be NULL)
 * @param ... Variable arguments for format string
 * 
 * Examples:
 *   log_progress_bar(TAG, 0.8f, 20, 15, "%d%% | mvmt:%.4f thr:%.4f", percent, mv, thr);
 *   log_progress_bar(TAG, progress, 20, -1, "%d%% (%d/%d)", percent, current, total);
 */
inline void log_progress_bar(const char* tag, float progress, int width = 20, 
                             int threshold_pos = -1, const char* format = nullptr, ...) {
  // Create progress bar
  int filled = (int)(progress * (threshold_pos > 0 ? threshold_pos : width));
  filled = (filled < 0) ? 0 : (filled > width ? width : filled);
  
  char bar[24];  // '[' + 20 chars + '|' + ']' + '\0'
  int idx = 0;
  bar[idx++] = '[';
  
  for (int i = 0; i < width; i++) {
    if (threshold_pos >= 0 && i == threshold_pos) {
      bar[idx++] = '|';
    } else if (i < filled) {
      bar[idx++] = '#';
    } else {
      bar[idx++] = '-';
    }
  }
  
  bar[idx++] = ']';
  bar[idx] = '\0';
  
  // Log with optional formatted text
  if (format != nullptr) {
    char text[256];
    va_list args;
    va_start(args, format);
    vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    ESP_LOGI(tag, "%s %s", bar, text);
  } else {
    ESP_LOGI(tag, "%s", bar);
  }
}

}  // namespace espectre
}  // namespace esphome
