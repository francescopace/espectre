/*
 * ESPectre - PCA Detector
 * 
 * Alternative motion detection algorithm based on PCA and Pearson correlation.
 * 
 * Algorithm:
 * 1. Collect CSI amplitudes into sliding window
 * 2. Apply PCA (power method) to extract principal component
 * 3. Calculate Pearson correlation between current and past PCA vectors
 * 4. Invert correlation: jitter = 1 - max(|correlation|)
 * 5. Count-based detection: if N violations in window, declare motion
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "detector_interface.h"
#include <cstdint>
#include <cstddef>

namespace esphome {
namespace espectre {

// PCA-specific constants
static constexpr size_t PCA_WINDOW_SIZE = 10;      // Packets for PCA computation
static constexpr size_t PCA_BUFFER_SIZE = 25;      // Max PCA vectors to store
static constexpr size_t PCA_CALIBRATION_SAMPLES = 10;  // Baseline samples for wander
static constexpr size_t PCA_JITTER_BUFFER_SIZE = 25;   // Jitter values for smoothing
static constexpr size_t PCA_MOVE_BUFFER_SIZE = 5;      // Window for count-based detection
static constexpr size_t PCA_OUTLIERS_NUM = 2;          // Violations needed (2/5)
static constexpr size_t PCA_SUBCARRIER_STEP = 4;       // Use every Nth subcarrier
static constexpr size_t PCA_NUM_SUBCARRIERS = 64 / PCA_SUBCARRIER_STEP;  // 16 subcarriers

// Scale factor for PCA values (jitter/threshold) to match MVS range
// PCA jitter is ~0.0001-0.001, MVS variance is ~0.01-1.0
// Scaling by 1000 makes both algorithms use similar threshold ranges (0.1-10.0)
static constexpr float PCA_SCALE = 1000.0f;
static constexpr float PCA_DEFAULT_THRESHOLD = 10.0f;  // 0.01 * PCA_SCALE

/**
 * @brief Calculate Pearson correlation coefficient between two vectors
 * 
 * @param a First vector
 * @param b Second vector
 * @param len Vector length
 * @return Correlation coefficient [-1, 1]
 */
float pearson_correlation(const float* a, const float* b, size_t len);

/**
 * @brief PCA using power method to find principal eigenvector
 * 
 * @param data_matrix 2D array [num_rows][num_cols]
 * @param num_rows Number of rows (packets)
 * @param num_cols Number of columns (subcarriers)
 * @param output Output vector of size num_cols
 * @param max_iters Maximum iterations for power method
 * @param precision Convergence threshold
 * @return true if successful
 */
bool pca_power_method(float** data_matrix, size_t num_rows, size_t num_cols,
                      float* output, size_t max_iters = 30, float precision = 0.0001f);

/**
 * @brief PCA-based motion detector
 * 
 * Implements IDetector interface for use with ESPectre.
 */
class PCADetector : public IDetector {
public:
    PCADetector();
    ~PCADetector() override;
    
    // IDetector interface
    void process_packet(const int8_t* csi_data, size_t csi_len,
                        const uint8_t* selected_subcarriers = nullptr,
                        uint8_t num_subcarriers = 0) override;
    
    void update_state() override;
    MotionState get_state() const override { return state_; }
    float get_motion_metric() const override { return current_jitter_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    void reset() override;
    bool is_ready() const override { return threshold_externally_set_; }
    uint32_t get_total_packets() const override { return static_cast<uint32_t>(packet_count_); }
    const char* get_name() const override { return "PCA"; }
    
    // PCA-specific getters
    
    /**
     * @brief Get current jitter value
     */
    float get_jitter() const { return current_jitter_; }
    
    /**
     * @brief Get current wander value
     */
    float get_wander() const { return current_wander_; }
    
    /**
     * @brief Get calibrated threshold (set externally by PCACalibrator)
     */
    float get_calibrated_threshold() const { return threshold_; }

private:
    // Extract amplitudes from CSI data (every PCA_SUBCARRIER_STEP-th)
    void extract_amplitudes(const int8_t* csi_data, size_t len, float* amplitudes);
    
    // Compute waveform jitter (correlation with past PCA vectors)
    float compute_jitter(const float* pca_current);
    
    // Compute waveform wander (correlation with calibration samples)
    float compute_wander(const float* pca_current);
    
    // State
    MotionState state_;
    
    // CSI amplitude buffer for PCA [PCA_WINDOW_SIZE][PCA_NUM_SUBCARRIERS]
    float* csi_buffer_[PCA_WINDOW_SIZE];
    size_t csi_buffer_count_;
    
    // PCA output buffer [PCA_BUFFER_SIZE][PCA_NUM_SUBCARRIERS]
    float* pca_buffer_[PCA_BUFFER_SIZE];
    size_t pca_buffer_count_;
    size_t pca_buffer_idx_;
    
    // Calibration samples [PCA_CALIBRATION_SAMPLES][PCA_NUM_SUBCARRIERS]
    float* calibration_data_[PCA_CALIBRATION_SAMPLES];
    size_t calibration_count_;
    
    // Jitter buffer for smoothing
    float jitter_buffer_[PCA_JITTER_BUFFER_SIZE];
    size_t jitter_buffer_count_;
    size_t jitter_buffer_idx_;
    
    // Current values
    float current_jitter_;
    float current_wander_;
    float threshold_;
    
    // Packet counter
    size_t packet_count_;
    
    // Last detection result (from process_packet)
    bool last_detection_result_;
    
    // Flag to skip internal calibration when threshold is set externally
    bool threshold_externally_set_;
};

}  // namespace espectre
}  // namespace esphome
