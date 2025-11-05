/*
 * ESPectre - Statistics Module
 * 
 * Statistical analysis and buffer management:
 * - Circular buffer for statistics
 * - Min/max/mean/stddev calculations
 * - Percentile calculations
 * - Threshold recommendations
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef STATISTICS_H
#define STATISTICS_H

#include <stddef.h>

// Statistics buffer
typedef struct {
    float *data;
    size_t size;
    size_t index;
    size_t count;
} stats_buffer_t;

// Statistics result
typedef struct {
    float min;
    float max;
    float mean;
    float stddev;
    size_t count;
} stats_result_t;

/**
 * Initialize statistics buffer
 * 
 * @param buffer Statistics buffer
 * @param size Buffer size
 * @return 0 on success, -1 on failure
 */
int stats_buffer_init(stats_buffer_t *buffer, size_t size);

/**
 * Free statistics buffer
 * 
 * @param buffer Statistics buffer
 */
void stats_buffer_free(stats_buffer_t *buffer);

/**
 * Add value to statistics buffer
 * 
 * @param buffer Statistics buffer
 * @param value Value to add
 */
void stats_buffer_add(stats_buffer_t *buffer, float value);

/**
 * Analyze statistics buffer
 * 
 * @param buffer Statistics buffer
 * @param result Output statistics result
 */
void stats_buffer_analyze(const stats_buffer_t *buffer, stats_result_t *result);

/**
 * Calculate percentile from statistics buffer
 * 
 * @param buffer Statistics buffer
 * @param percentile Percentile to calculate (0-100)
 * @return Percentile value
 */
float stats_buffer_percentile(const stats_buffer_t *buffer, float percentile);

/**
 * Calculate recommended threshold from statistics
 * Uses median and 75th percentile
 * 
 * @param buffer Statistics buffer
 * @return Recommended threshold value
 */
float stats_calculate_recommended_threshold(const stats_buffer_t *buffer);

#endif // STATISTICS_H
