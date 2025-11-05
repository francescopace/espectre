/*
 * ESPectre - Signal Filtering Module
 * 
 * Provides various signal processing filters for CSI data:
 * - Butterworth low-pass filter (removes high frequency noise >8Hz)
 * - Hampel filter (outlier detection and removal)
 * - Savitzky-Golay filter (smoothing while preserving shape)
 * - Adaptive normalizer (tracks running mean and variance)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef FILTERS_H
#define FILTERS_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "wavelet.h"

// Filter configuration constants
#define BUTTERWORTH_ORDER 4
#define BUTTERWORTH_CUTOFF 8.0f  // Hz
#define SAVGOL_WINDOW_SIZE 5     // must be odd
#define SAVGOL_POLY_ORDER 2
#define HAMPEL_DEFAULT_THRESHOLD 2.0f
#define MAD_SCALE_FACTOR 1.4826f

// Butterworth filter state
typedef struct {
    float b[BUTTERWORTH_ORDER + 1];  // numerator coefficients
    float a[BUTTERWORTH_ORDER + 1];  // denominator coefficients
    float x[BUTTERWORTH_ORDER + 1];  // input history
    float y[BUTTERWORTH_ORDER + 1];  // output history
    bool initialized;
} butterworth_filter_t;

// Adaptive normalizer state
typedef struct {
    float running_mean;
    float running_variance;
    size_t sample_count;
    float alpha;
    bool initialized;
} adaptive_normalizer_t;

// Filter buffer for windowed operations
typedef struct {
    float data[SAVGOL_WINDOW_SIZE];
    size_t index;
    size_t count;
} filter_buffer_t;

// Filter configuration
typedef struct {
    bool butterworth_enabled;
    bool wavelet_enabled;           // NEW: Wavelet denoising
    int wavelet_level;              // NEW: Decomposition level (1-3)
    float wavelet_threshold;        // NEW: Noise threshold
    bool hampel_enabled;
    float hampel_threshold;
    bool savgol_enabled;
    bool adaptive_normalizer_enabled;
} filter_config_t;

/**
 * Initialize Butterworth low-pass filter
 * Pre-computed coefficients for 4th order, 8Hz cutoff at 100Hz sampling
 * 
 * @param filter Pointer to filter state structure
 */
void butterworth_init(butterworth_filter_t *filter);

/**
 * Apply Butterworth IIR filter
 * Removes high frequency noise above cutoff frequency
 * 
 * @param filter Pointer to filter state
 * @param input Input sample
 * @return Filtered output sample
 */
float butterworth_filter(butterworth_filter_t *filter, float input);

/**
 * Apply Hampel filter for outlier detection
 * Uses Median Absolute Deviation (MAD) to identify and replace outliers
 * 
 * @param window Array of samples in the window
 * @param window_size Number of samples in window
 * @param current_value Current value to filter
 * @param threshold MAD threshold multiplier (typically 2-3)
 * @return Filtered value (median if outlier, original otherwise)
 */
float hampel_filter(const float *window, size_t window_size, 
                   float current_value, float threshold);

/**
 * Apply Savitzky-Golay filter
 * Smooths signal while preserving shape using polynomial fitting
 * 
 * @param window Array of samples (must be SAVGOL_WINDOW_SIZE)
 * @param window_size Number of samples in window
 * @return Filtered value
 */
float savitzky_golay_filter(const float *window, size_t window_size);

/**
 * Initialize adaptive normalizer
 * 
 * @param norm Pointer to normalizer state
 * @param alpha Learning rate (typically 0.01)
 */
void adaptive_normalizer_init(adaptive_normalizer_t *norm, float alpha);

/**
 * Update adaptive normalizer with new sample
 * Uses Welford's algorithm for online variance calculation
 * 
 * @param norm Pointer to normalizer state
 * @param value New sample value
 */
void adaptive_normalizer_update(adaptive_normalizer_t *norm, float value);

/**
 * Get current normalizer statistics
 * 
 * @param norm Pointer to normalizer state
 * @param mean Output: current running mean
 * @param variance Output: current running variance
 */
void adaptive_normalizer_get_stats(const adaptive_normalizer_t *norm, 
                                   float *mean, float *variance);

/**
 * Initialize filter buffer
 * 
 * @param fb Pointer to filter buffer
 */
void filter_buffer_init(filter_buffer_t *fb);

/**
 * Add sample to filter buffer
 * 
 * @param fb Pointer to filter buffer
 * @param value Sample value to add
 */
void filter_buffer_add(filter_buffer_t *fb, float value);

/**
 * Get window of samples from filter buffer
 * Returns samples in chronological order (oldest to newest)
 * 
 * @param fb Pointer to filter buffer
 * @param window Output buffer for samples
 * @param window_capacity Size of output buffer
 * @param size Output: number of samples returned
 */
void filter_buffer_get_window(const filter_buffer_t *fb, float *window, 
                              size_t window_capacity, size_t *size);

/**
 * Apply complete filter pipeline
 * Applies filters in sequence: Butterworth -> Wavelet -> Hampel -> Savitzky-Golay -> Normalization
 * 
 * @param raw_value Input sample
 * @param config Filter configuration
 * @param butterworth Butterworth filter state
 * @param wavelet Wavelet filter state (NEW)
 * @param buffer Filter buffer for windowed operations
 * @param normalizer Adaptive normalizer state
 * @return Filtered and normalized value
 */
float apply_filter_pipeline(float raw_value, 
                            const filter_config_t *config,
                            butterworth_filter_t *butterworth,
                            wavelet_state_t *wavelet,
                            filter_buffer_t *buffer,
                            adaptive_normalizer_t *normalizer);

#endif // FILTERS_H
