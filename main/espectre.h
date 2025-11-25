/*
 * ESPectre - Centralized Configuration Constants
 * 
 * All system-wide constants and default values in one place.
 * Similar to config.py in the Python version.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef ESPECTRE_H
#define ESPECTRE_H

// ============================================================================
// SYSTEM CONFIGURATION
// ============================================================================

// Logging intervals
#define LOG_CSI_VALUES_INTERVAL     1       // seconds

// Publishing configuration
#define PUBLISH_INTERVAL            1.0f    // seconds

// WiFi promiscuous mode (false = receive CSI only from connected AP, true = all WiFi packets)
#define PROMISCUOUS_MODE            false

// ============================================================================
// CSI PROCESSING
// ============================================================================

// Maximum CSI data length (ESP32-S3: 256 bytes, ESP32-C6: 128 bytes, buffer sized for largest)
#define CSI_MAX_LENGTH              384

// Amplitude moments calculation window
#define AMPLITUDE_MOMENTS_WINDOW    20

// Enable subcarrier filtering
#define ENABLE_SUBCARRIER_FILTERING 1

// Numerical stability constant
#define EPSILON_SMALL               1e-6f

// Subcarrier selection limits
#define MAX_SUBCARRIERS             64      // Maximum number of subcarriers that can be selected

// ============================================================================
// FILTERS
// ============================================================================

// Butterworth filter
#define BUTTERWORTH_ORDER           4
#define BUTTERWORTH_CUTOFF          8.0f    // Hz

// Savitzky-Golay filter
#define SAVGOL_WINDOW_SIZE          5       // must be odd
#define SAVGOL_POLY_ORDER           2

// Hampel filter
#define HAMPEL_THRESHOLD_MIN        1.0f
#define HAMPEL_THRESHOLD_MAX        10.0f
#define HAMPEL_DEFAULT_THRESHOLD    2.0f
#define MAD_SCALE_FACTOR            1.4826f // Median Absolute Deviation scale factor

// Wavelet filter
#define WAVELET_LEVEL_MIN           1
#define WAVELET_LEVEL_MAX           3
#define WAVELET_THRESHOLD_MIN       0.5f
#define WAVELET_THRESHOLD_MAX       2.0f
#define WAVELET_DB4_LENGTH          8       // Daubechies db4 filter length
#define WAVELET_MAX_LEVEL           3       // Maximum decomposition level
#define WAVELET_BUFFER_SIZE         32      // Circular buffer for streaming (power of 2)

// ============================================================================
// SEGMENTATION
// ============================================================================

// Window size limits (moving variance window)
#define SEGMENTATION_WINDOW_SIZE_MIN     10     // packets
#define SEGMENTATION_WINDOW_SIZE_MAX     200    // packets
#define SEGMENTATION_DEFAULT_WINDOW_SIZE 50     // packets

// Threshold for motion detection
#define SEGMENTATION_DEFAULT_THRESHOLD   3.0f   // Lower values = more sensitive to motion
#define DEFAULT_THRESHOLD                0.50f  // Default threshold for motion detection (optimized for amplitude skewness)

// ============================================================================
// TRAFFIC GENERATOR
// ============================================================================

#define TRAFFIC_RATE_MAX            1000    // packets/sec (maximum allowed rate)

// ============================================================================
// NVS STORAGE
// ============================================================================

// NVS Namespaces
#define NVS_NAMESPACE_CONFIG        "espectre_cfg"

// Versioning for future compatibility
#define NVS_CONFIG_VERSION          10      // Incremented: removed min_length parameter

// Total number of CSI features
#define NUM_TOTAL_FEATURES          10

// NVS keys for config data
#define NVS_KEY_CFG_VERSION         "cfg_ver"
#define NVS_KEY_CFG_DATA            "cfg_data"

#endif // ESPECTRE_H
