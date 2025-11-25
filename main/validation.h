/*
 * ESPectre - Validation Module
 * 
 * Centralized validation functions for all configuration parameters.
 * Used at system boundaries (MQTT commands) and defensively (NVS loading).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef VALIDATION_H
#define VALIDATION_H

#include <stdint.h>
#include <stdbool.h>

/**
 * Validate float value (check for NaN and Infinity)
 * 
 * @param value Float value to validate
 * @return true if valid (not NaN, not Inf), false otherwise
 */
bool validate_float(float value);

/**
 * Validate Hampel filter threshold
 * Range: HAMPEL_THRESHOLD_MIN to HAMPEL_THRESHOLD_MAX
 * 
 * @param value Threshold value to validate
 * @return true if valid, false otherwise
 */
bool validate_hampel_threshold(float value);

/**
 * Validate wavelet decomposition level
 * Range: WAVELET_LEVEL_MIN to WAVELET_LEVEL_MAX
 * 
 * @param value Level value to validate
 * @return true if valid, false otherwise
 */
bool validate_wavelet_level(int value);

/**
 * Validate wavelet threshold
 * Range: WAVELET_THRESHOLD_MIN to WAVELET_THRESHOLD_MAX
 * 
 * @param value Threshold value to validate
 * @return true if valid, false otherwise
 */
bool validate_wavelet_threshold(float value);

/**
 * Validate segmentation threshold
 * Range: 0.5 to 10.0
 * 
 * @param value Threshold value to validate
 * @return true if valid, false otherwise
 */
bool validate_segmentation_threshold(float value);

/**
 * Validate segmentation window size
 * Range: SEGMENTATION_WINDOW_SIZE_MIN to SEGMENTATION_WINDOW_SIZE_MAX
 * 
 * @param value Window size to validate
 * @return true if valid, false otherwise
 */
bool validate_segmentation_window_size(uint16_t value);

/**
 * Validate single subcarrier index
 * Range: 0 to 63
 * 
 * @param index Subcarrier index to validate
 * @return true if valid, false otherwise
 */
bool validate_subcarrier_index(uint8_t index);

/**
 * Validate number of selected subcarriers
 * Range: 1 to MAX_SUBCARRIERS
 * 
 * @param count Number of subcarriers to validate
 * @return true if valid, false otherwise
 */
bool validate_subcarrier_count(uint8_t count);

/**
 * Validate traffic generator rate
 * Range: 0 (disabled) to TRAFFIC_RATE_MAX
 * 
 * @param rate Rate in packets per second
 * @return true if valid, false otherwise
 */
bool validate_traffic_rate(uint32_t rate);

/**
 * Validate Savitzky-Golay window size
 * Must be odd number between 3 and 11
 * 
 * @param value Window size to validate
 * @return true if valid, false otherwise
 */
bool validate_savgol_window_size(int value);

#endif // VALIDATION_H
