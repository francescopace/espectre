/*
 * ESPectre - Mock CSI Data Generator for Testing
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef MOCK_CSI_DATA_H
#define MOCK_CSI_DATA_H

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include "esp_wifi_types.h"

// Mock CSI data types
typedef enum {
    MOCK_CSI_STATIC,      // Static environment (low noise)
    MOCK_CSI_WALKING,     // Walking movement
    MOCK_CSI_RUNNING,     // Running movement
    MOCK_CSI_OUTLIERS     // Data with outliers
} mock_csi_type_t;

// Pattern configuration for synthetic CSI data
typedef struct {
    float baseline_noise;      // Gaussian noise std dev (e.g., 0.5 for static)
    float movement_amplitude;  // Movement signal amplitude (e.g., 5.0 for walking)
    float frequency;          // Movement frequency in Hz (e.g., 2.0 for walking)
} csi_pattern_t;

/**
 * Generate CSI baseline data (static environment with small noise)
 * @param length Number of CSI samples to generate
 * @return Pointer to allocated CSI data (caller must free)
 */
static inline int8_t* generate_csi_baseline(size_t length) {
    int8_t *data = (int8_t*)malloc(length * sizeof(int8_t));
    if (!data) return NULL;
    
    for (size_t i = 0; i < length; i++) {
        // Small Gaussian noise around zero
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;  // -1.0 to 1.0
        data[i] = (int8_t)(noise * 0.5f);
    }
    return data;
}

/**
 * Generate CSI movement data (baseline + sinusoidal movement pattern)
 * @param length Number of CSI samples to generate
 * @param pattern Movement pattern configuration
 * @return Pointer to allocated CSI data (caller must free)
 */
static inline int8_t* generate_csi_movement(size_t length, const csi_pattern_t *pattern) {
    int8_t *data = (int8_t*)malloc(length * sizeof(int8_t));
    if (!data) return NULL;
    
    for (size_t i = 0; i < length; i++) {
        // Baseline noise
        float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * pattern->baseline_noise;
        
        // Movement signal (sinusoid)
        float signal = pattern->movement_amplitude * 
                      sinf(2.0f * M_PI * pattern->frequency * i / 100.0f);
        
        data[i] = (int8_t)(noise + signal);
    }
    return data;
}

/**
 * Generate constant CSI data (all values identical - tests NaN protection)
 * @param value Constant value to fill
 * @param length Number of samples
 * @return Pointer to allocated CSI data (caller must free)
 */
static inline int8_t* generate_constant_csi(int8_t value, size_t length) {
    int8_t *data = (int8_t*)malloc(length * sizeof(int8_t));
    if (!data) return NULL;
    
    for (size_t i = 0; i < length; i++) {
        data[i] = value;
    }
    return data;
}

/**
 * Generate CSI with outliers (tests Hampel filter)
 * @param length Number of samples
 * @param outlier_count Number of outliers to inject
 * @return Pointer to allocated CSI data (caller must free)
 */
static inline int8_t* generate_csi_with_outliers(size_t length, int outlier_count) {
    int8_t *data = generate_csi_baseline(length);
    if (!data) return NULL;
    
    // Inject random outliers
    for (int i = 0; i < outlier_count && i < (int)length; i++) {
        size_t idx = rand() % length;
        data[idx] = (rand() % 2) ? 127 : -128;  // Extreme values
    }
    return data;
}

/**
 * Generate mock CSI data in wifi_csi_info_t format
 * @param csi_info Pointer to wifi_csi_info_t structure to fill
 * @param type Type of mock data to generate
 */
static inline void generate_mock_csi_data(wifi_csi_info_t *csi_info, mock_csi_type_t type) {
    if (!csi_info) return;
    
    // Initialize random seed (use a fixed seed for reproducible tests, or time-based for variety)
    static int seed_initialized = 0;
    if (!seed_initialized) {
        srand(12345);  // Fixed seed for reproducible test results
        seed_initialized = 1;
    }
    
    // Set standard CSI length
    csi_info->len = 128;
    
    switch (type) {
        case MOCK_CSI_STATIC: {
            // Static environment - low noise (increased amplitude to ensure non-zero values)
            for (int i = 0; i < 128; i++) {
                float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
                csi_info->buf[i] = (int8_t)(noise * 2.0f);  // Increased from 0.5 to 2.0
            }
            break;
        }
        
        case MOCK_CSI_WALKING: {
            // Walking movement - moderate amplitude, ~2Hz
            for (int i = 0; i < 128; i++) {
                float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
                float signal = 5.0f * sinf(2.0f * M_PI * 2.0f * i / 100.0f);
                csi_info->buf[i] = (int8_t)(noise + signal);
            }
            break;
        }
        
        case MOCK_CSI_RUNNING: {
            // Running movement - high amplitude, ~4Hz
            for (int i = 0; i < 128; i++) {
                float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
                float signal = 10.0f * sinf(2.0f * M_PI * 4.0f * i / 100.0f);
                csi_info->buf[i] = (int8_t)(noise + signal);
            }
            break;
        }
        
        case MOCK_CSI_OUTLIERS: {
            // Baseline with outliers
            for (int i = 0; i < 128; i++) {
                float noise = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
                csi_info->buf[i] = (int8_t)(noise * 0.5f);
            }
            // Inject some outliers
            for (int i = 0; i < 10; i++) {
                int idx = rand() % 128;
                csi_info->buf[idx] = (rand() % 2) ? 127 : -128;
            }
            break;
        }
    }
}

#endif // MOCK_CSI_DATA_H
