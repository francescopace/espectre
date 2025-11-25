/*
 * ESPectre - Wavelet Transform Module
 * 
 * Provides Daubechies db4 wavelet transform for signal denoising
 * Optimized for ESP32 with minimal memory footprint
 * 
 * Based on research:
 * - "Location intelligence system for people estimation" (ESP32 + wavelet)
 * - "CSI-HC: WiFi-Based Indoor Complex Human Motion Recognition" (Butterworth + Sym8)
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef WAVELET_H
#define WAVELET_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include "espectre.h"

// Thresholding methods
typedef enum {
    WAVELET_THRESH_SOFT,    // Soft thresholding (recommended)
    WAVELET_THRESH_HARD     // Hard thresholding
} wavelet_threshold_method_t;

// Wavelet filter state for streaming operation
typedef struct {
    float buffer[WAVELET_BUFFER_SIZE];  // Circular buffer for input samples
    size_t buffer_index;                 // Current position in buffer
    size_t buffer_count;                 // Number of samples in buffer
    int decomp_level;                    // Decomposition level (1-3)
    float threshold;                     // Noise threshold
    wavelet_threshold_method_t method;   // Thresholding method
    bool initialized;
} wavelet_state_t;

// Daubechies db4 coefficients (pre-computed)
extern const float WAVELET_DB4_LP[WAVELET_DB4_LENGTH];  // Low-pass decomposition
extern const float WAVELET_DB4_HP[WAVELET_DB4_LENGTH];  // High-pass decomposition
extern const float WAVELET_DB4_LR[WAVELET_DB4_LENGTH];  // Low-pass reconstruction
extern const float WAVELET_DB4_HR[WAVELET_DB4_LENGTH];  // High-pass reconstruction

/**
 * Initialize wavelet filter state
 * 
 * @param state Pointer to wavelet state structure
 * @param level Decomposition level (1-3, recommended: 3)
 * @param threshold Noise threshold (recommended: 0.5-2.0)
 * @param method Thresholding method (SOFT recommended)
 */
void wavelet_init(wavelet_state_t *state, int level, float threshold, 
                  wavelet_threshold_method_t method);

/**
 * Apply wavelet denoising to a signal buffer
 * Performs full DWT decomposition, thresholding, and reconstruction
 * 
 * @param input Input signal array
 * @param output Output denoised signal array
 * @param length Signal length (must be power of 2)
 * @param level Decomposition level (1-3)
 * @param threshold Noise threshold
 * @param method Thresholding method
 * @return 0 on success, -1 on error
 */
int wavelet_denoise(const float *input, float *output, size_t length,
                    int level, float threshold, wavelet_threshold_method_t method);

/**
 * Process single sample with wavelet denoising (streaming mode)
 * Uses circular buffer to maintain state for real-time processing
 * 
 * @param state Pointer to wavelet state
 * @param input New input sample
 * @return Denoised output sample
 */
float wavelet_denoise_sample(wavelet_state_t *state, float input);

/**
 * Apply soft thresholding
 * y = sign(x) * max(|x| - threshold, 0)
 * 
 * @param value Input value
 * @param threshold Threshold value
 * @return Thresholded value
 */
float wavelet_soft_threshold(float value, float threshold);

/**
 * Apply hard thresholding
 * y = x if |x| > threshold, else 0
 * 
 * @param value Input value
 * @param threshold Threshold value
 * @return Thresholded value
 */
float wavelet_hard_threshold(float value, float threshold);

/**
 * Estimate noise level using Median Absolute Deviation (MAD)
 * Used to automatically set threshold
 * 
 * @param coeffs Wavelet coefficients (detail coefficients at finest level)
 * @param length Number of coefficients
 * @return Estimated noise standard deviation
 */
float wavelet_estimate_noise(const float *coeffs, size_t length);

/**
 * Single-level wavelet decomposition
 * Internal function for DWT
 * 
 * @param input Input signal
 * @param length Input length
 * @param approx Output approximation coefficients
 * @param detail Output detail coefficients
 */
void wavelet_decompose_level(const float *input, size_t length,
                             float *approx, float *detail);

/**
 * Single-level wavelet reconstruction
 * Internal function for inverse DWT
 * 
 * @param approx Approximation coefficients
 * @param detail Detail coefficients
 * @param length Coefficient length
 * @param output Reconstructed signal
 */
void wavelet_reconstruct_level(const float *approx, const float *detail,
                               size_t length, float *output);

#endif // WAVELET_H
