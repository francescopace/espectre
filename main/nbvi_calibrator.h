/*
 * ESPectre - NBVI (Normalized Baseline Variability Index) Calibrator
 * 
 * Automatic subcarrier selection using percentile-based baseline detection.
 * Implements NBVI Weighted α=0.3 algorithm for optimal subcarrier selection.
 * 
 * NBVI FORMULA (Optimized):
 *     NBVI_weighted = 0.3 × (σ/μ²) + 0.7 × (σ/μ)
 *     
 *     Where:
 *     - σ: Standard deviation of subcarrier magnitude in baseline
 *     - μ: Mean magnitude of subcarrier in baseline
 *     - 0.3: Energy normalization weight (penalizes weak subcarriers)
 *     - 0.7: Stability weight (rewards low variance)
 * 
 * ALGORITHM:
 * 1. Collect CSI packets at boot (500 packets @ 100Hz = 5 seconds)
 * 2. Find baseline window using percentile-based detection (NO threshold needed)
 * 3. Calculate NBVI for all 64 subcarriers
 * 4. Apply Noise Gate (exclude weak subcarriers below 10th percentile)
 * 5. Select top 12 subcarriers with spectral spacing (Δf≥3)
 * 
 * PERFORMANCE:
 * - Pure data: F1=97.1% (gap to manual: -0.2%)
 * - Mixed data: F1=91.2% (best automatic method)
 * - Zero configuration: NO threshold needed
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#ifndef NBVI_CALIBRATOR_H
#define NBVI_CALIBRATOR_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

// ============================================================================
// NBVI CALIBRATOR CONTEXT
// ============================================================================

/**
 * NBVI calibrator context
 * 
 * Manages CSI packet collection and NBVI-based subcarrier selection.
 * Memory footprint: ~128KB for magnitude buffer (500 packets × 64 subcarriers × 4 bytes)
 */
typedef struct {
    // Configuration parameters
    uint16_t buffer_size;           // Number of packets to collect (default: 500)
    uint16_t window_size;           // Window size for baseline detection (default: 100)
    uint16_t window_step;           // Step size for sliding window (default: 50)
    uint8_t percentile;             // Percentile for baseline detection (default: 10)
    float alpha;                    // NBVI weighting factor (default: 0.3)
    uint8_t min_spacing;            // Minimum spacing between subcarriers (default: 3)
    uint8_t noise_gate_percentile;  // Noise gate percentile (default: 10)
    
    // Magnitude buffer: [packet0_sc0, packet0_sc1, ..., packet0_sc63, packet1_sc0, ...]
    // Flat array for memory efficiency: buffer_size × 64 floats
    float *magnitude_buffer;
    uint16_t buffer_count;          // Current number of packets collected
    
    // Results
    uint8_t selected_band[12];      // Output: 12 selected subcarrier indices
    bool calibrated;                // Calibration status
} nbvi_calibrator_t;

// ============================================================================
// NBVI CALIBRATOR API
// ============================================================================

/**
 * Initialize NBVI calibrator with default parameters
 * 
 * Allocates memory for magnitude buffer (buffer_size × 64 floats).
 * Must call nbvi_calibrator_free() to release memory.
 * 
 * @param cal Calibrator context to initialize
 * @return ESP_OK on success, ESP_ERR_NO_MEM if allocation fails
 */
esp_err_t nbvi_calibrator_init(nbvi_calibrator_t *cal);

/**
 * Add CSI packet to calibration buffer
 * 
 * Extracts magnitude |H| = sqrt(I² + Q²) for all 64 subcarriers
 * and stores in magnitude buffer.
 * 
 * @param cal Calibrator context
 * @param csi_data Raw CSI data (int8_t array, I/Q interleaved)
 * @param csi_len Length of CSI data (must be at least 128 bytes)
 * @return true if buffer is full (ready for calibration), false otherwise
 */
bool nbvi_calibrator_add_packet(nbvi_calibrator_t *cal, 
                                const int8_t *csi_data, 
                                size_t csi_len);

/**
 * Run NBVI calibration
 * 
 * Performs percentile-based baseline detection and NBVI-based subcarrier selection:
 * 1. Find baseline window using sliding window analysis (percentile-based)
 * 2. Calculate NBVI Weighted α=0.3 for all 64 subcarriers
 * 3. Apply Noise Gate (exclude weak subcarriers)
 * 4. Select top 12 with spectral spacing (Δf≥3)
 * 
 * @param cal Calibrator context (must have buffer_size packets collected)
 * @param current_band Current subcarrier band (for baseline detection)
 * @param current_band_size Size of current band
 * @param output_band Output array for selected subcarriers (must have space for 12)
 * @param output_size Output: number of selected subcarriers (should be 12)
 * @return ESP_OK on success, ESP_FAIL if calibration fails
 */
esp_err_t nbvi_calibrator_calibrate(nbvi_calibrator_t *cal,
                                    const uint8_t *current_band,
                                    uint8_t current_band_size,
                                    uint8_t *output_band,
                                    uint8_t *output_size);

/**
 * Free calibrator memory
 * 
 * Releases magnitude buffer allocated by nbvi_calibrator_init().
 * 
 * @param cal Calibrator context
 */
void nbvi_calibrator_free(nbvi_calibrator_t *cal);

#endif // NBVI_CALIBRATOR_H
