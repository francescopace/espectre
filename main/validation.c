/*
 * ESPectre - Validation Module Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "validation.h"
#include "espectre.h"
#include <math.h>

// Validate float value (check for NaN and Infinity)
bool validate_float(float value) {
    return !isnan(value) && !isinf(value);
}

// Validate Hampel filter threshold
bool validate_hampel_threshold(float value) {
    if (!validate_float(value)) {
        return false;
    }
    return value >= HAMPEL_THRESHOLD_MIN && value <= HAMPEL_THRESHOLD_MAX;
}

// Validate wavelet decomposition level
bool validate_wavelet_level(int value) {
    return value >= WAVELET_LEVEL_MIN && value <= WAVELET_LEVEL_MAX;
}

// Validate wavelet threshold
bool validate_wavelet_threshold(float value) {
    if (!validate_float(value)) {
        return false;
    }
    return value >= WAVELET_THRESHOLD_MIN && value <= WAVELET_THRESHOLD_MAX;
}

// Validate segmentation threshold
bool validate_segmentation_threshold(float value) {
    if (!validate_float(value)) {
        return false;
    }
    return value >= 0.5f && value <= 10.0f;
}

// Validate segmentation window size
bool validate_segmentation_window_size(uint16_t value) {
    return value >= SEGMENTATION_WINDOW_SIZE_MIN && value <= SEGMENTATION_WINDOW_SIZE_MAX;
}

// Validate single subcarrier index
bool validate_subcarrier_index(uint8_t index) {
    return index < 64;  // Valid range: 0-63
}

// Validate number of selected subcarriers
bool validate_subcarrier_count(uint8_t count) {
    return count >= 1 && count <= MAX_SUBCARRIERS;
}

// Validate traffic generator rate
bool validate_traffic_rate(uint32_t rate) {
    return rate <= TRAFFIC_RATE_MAX;  // 0 is valid (disabled)
}

// Validate Savitzky-Golay window size
bool validate_savgol_window_size(int value) {
    // Must be odd number between 3 and 11
    if (value < 3 || value > 11) {
        return false;
    }
    return (value % 2) == 1;  // Must be odd
}
