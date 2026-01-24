/*
 * ESPectre - PCA Detector Implementation
 * 
 * Alternative motion detection algorithm based on PCA and Pearson correlation.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "pca_detector.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "PCADetector";

// ============================================================================
// Utility Functions
// ============================================================================

float pearson_correlation(const float* a, const float* b, size_t len) {
    if (len == 0) return 0.0f;
    
    // Calculate means
    float mean_a = 0.0f, mean_b = 0.0f;
    for (size_t i = 0; i < len; i++) {
        mean_a += a[i];
        mean_b += b[i];
    }
    mean_a /= static_cast<float>(len);
    mean_b /= static_cast<float>(len);
    
    // Calculate covariance and variances
    float cov_sum = 0.0f;
    float var_sum_a = 0.0f;
    float var_sum_b = 0.0f;
    
    for (size_t i = 0; i < len; i++) {
        float diff_a = a[i] - mean_a;
        float diff_b = b[i] - mean_b;
        cov_sum += diff_a * diff_b;
        var_sum_a += diff_a * diff_a;
        var_sum_b += diff_b * diff_b;
    }
    
    float denominator = std::sqrt(var_sum_a * var_sum_b);
    if (denominator < 1e-10f) return 0.0f;
    
    return cov_sum / denominator;
}

bool pca_power_method(float** data_matrix, size_t num_rows, size_t num_cols,
                      float* output, size_t max_iters, float precision) {
    if (num_rows == 0 || num_cols == 0 || !data_matrix || !output) {
        return false;
    }
    
    // Compute covariance matrix (num_rows x num_rows)
    float zoom_out = static_cast<float>(num_rows * num_cols);
    
    // Allocate covariance matrix on stack (small size)
    float cov_matrix[PCA_WINDOW_SIZE][PCA_WINDOW_SIZE] = {0};
    
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t j = 0; j <= i; j++) {
            float cov_sum = 0.0f;
            for (size_t k = 0; k < num_cols; k++) {
                cov_sum += data_matrix[i][k] * data_matrix[j][k];
            }
            cov_matrix[i][j] = cov_sum / zoom_out;
            if (i != j) {
                cov_matrix[j][i] = cov_matrix[i][j];
            }
        }
    }
    
    // Power method to find principal eigenvector
    float eigenvector[PCA_WINDOW_SIZE];
    for (size_t i = 0; i < num_rows; i++) {
        eigenvector[i] = 1.0f;
    }
    
    float eigenvalue = 1.0f;
    float eigenvalue_last = 0.0f;
    
    for (size_t iter = 0; iter < max_iters; iter++) {
        if (std::fabs(eigenvalue - eigenvalue_last) <= precision) {
            break;
        }
        
        eigenvalue_last = eigenvalue;
        eigenvalue = 0.0f;
        
        // Multiply: new_vec = cov_matrix @ eigenvector
        float new_vec[PCA_WINDOW_SIZE] = {0};
        for (size_t i = 0; i < num_rows; i++) {
            for (size_t j = 0; j < num_rows; j++) {
                new_vec[i] += cov_matrix[i][j] * eigenvector[j];
            }
            if (new_vec[i] > eigenvalue) {
                eigenvalue = new_vec[i];
            }
        }
        
        // Normalize
        if (eigenvalue > 1e-10f) {
            for (size_t i = 0; i < num_rows; i++) {
                eigenvector[i] = new_vec[i] / eigenvalue;
            }
        }
    }
    
    // Project data onto eigenvector to get principal component
    for (size_t k = 0; k < num_cols; k++) {
        output[k] = 0.0f;
        for (size_t i = 0; i < num_rows; i++) {
            output[k] += data_matrix[i][k] * eigenvector[i];
        }
        output[k] /= static_cast<float>(num_rows);
    }
    
    return true;
}

// ============================================================================
// PCADetector Implementation
// ============================================================================

PCADetector::PCADetector()
    : state_(MotionState::IDLE)
    , csi_buffer_count_(0)
    , pca_buffer_count_(0)
    , pca_buffer_idx_(0)
    , calibration_count_(0)
    , jitter_buffer_count_(0)
    , jitter_buffer_idx_(0)
    , current_jitter_(0.0f)
    , current_wander_(0.0f)
    , threshold_(PCA_DEFAULT_THRESHOLD)
    , packet_count_(0)
    , last_detection_result_(false)
    , threshold_externally_set_(false) {
    
    // Allocate buffers
    for (size_t i = 0; i < PCA_WINDOW_SIZE; i++) {
        csi_buffer_[i] = new float[PCA_NUM_SUBCARRIERS]();
    }
    for (size_t i = 0; i < PCA_BUFFER_SIZE; i++) {
        pca_buffer_[i] = new float[PCA_NUM_SUBCARRIERS]();
    }
    for (size_t i = 0; i < PCA_CALIBRATION_SAMPLES; i++) {
        calibration_data_[i] = new float[PCA_NUM_SUBCARRIERS]();
    }
    
    std::memset(jitter_buffer_, 0, sizeof(jitter_buffer_));
    
    ESP_LOGI(TAG, "Initialized (subcarriers=%zu, threshold=%.3f)", PCA_NUM_SUBCARRIERS, threshold_);
}

PCADetector::~PCADetector() {
    for (size_t i = 0; i < PCA_WINDOW_SIZE; i++) {
        delete[] csi_buffer_[i];
    }
    for (size_t i = 0; i < PCA_BUFFER_SIZE; i++) {
        delete[] pca_buffer_[i];
    }
    for (size_t i = 0; i < PCA_CALIBRATION_SAMPLES; i++) {
        delete[] calibration_data_[i];
    }
}

// ============================================================================
// IDetector Interface
// ============================================================================

void PCADetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                  const uint8_t* /* selected_subcarriers */,
                                  uint8_t /* num_subcarriers */) {
    // PCA uses its own subcarrier selection (every PCA_SUBCARRIER_STEP-th)
    // Ignores the passed subcarrier selection
    
    packet_count_++;
    
    // Extract amplitudes (every PCA_SUBCARRIER_STEP-th subcarrier)
    size_t buffer_idx = csi_buffer_count_ % PCA_WINDOW_SIZE;
    extract_amplitudes(csi_data, csi_len, csi_buffer_[buffer_idx]);
    csi_buffer_count_++;
    
    // Need enough data for PCA
    if (csi_buffer_count_ < PCA_WINDOW_SIZE) {
        current_jitter_ = 0.0f;
        current_wander_ = 0.0f;
        last_detection_result_ = false;
        return;
    }
    
    // Compute PCA on current window
    float pca_current[PCA_NUM_SUBCARRIERS];
    
    // Build pointer array for pca_power_method
    float* window_ptrs[PCA_WINDOW_SIZE];
    for (size_t i = 0; i < PCA_WINDOW_SIZE; i++) {
        size_t idx = (csi_buffer_count_ - PCA_WINDOW_SIZE + i) % PCA_WINDOW_SIZE;
        window_ptrs[i] = csi_buffer_[idx];
    }
    
    if (!pca_power_method(window_ptrs, PCA_WINDOW_SIZE, PCA_NUM_SUBCARRIERS, pca_current)) {
        current_jitter_ = 0.0f;
        current_wander_ = 0.0f;
        last_detection_result_ = false;
        return;
    }
    
    // Calculate waveform metrics (as correlations, 0-1 range)
    float jitter_corr = compute_jitter(pca_current);
    float wander_corr = compute_wander(pca_current);
    
    // Invert: 1 - correlation, so high value = movement/change
    // Scale by PCA_SCALE (1000) to match MVS threshold range (0.1-10.0)
    float jitter_inverted = (1.0f - jitter_corr) * PCA_SCALE;
    float wander_inverted = (1.0f - wander_corr) * PCA_SCALE;
    
    current_jitter_ = jitter_inverted;
    current_wander_ = wander_inverted;
    
    // Store PCA vector for future comparisons
    std::memcpy(pca_buffer_[pca_buffer_idx_], pca_current, PCA_NUM_SUBCARRIERS * sizeof(float));
    pca_buffer_idx_ = (pca_buffer_idx_ + 1) % PCA_BUFFER_SIZE;
    pca_buffer_count_++;
    
    // Collect calibration samples during initial phase (for wander calculation)
    if (calibration_count_ < PCA_CALIBRATION_SAMPLES && pca_buffer_count_ % 5 == 0) {
        std::memcpy(calibration_data_[calibration_count_], pca_current, PCA_NUM_SUBCARRIERS * sizeof(float));
        calibration_count_++;
    }
    
    // Add to jitter buffer for smoothing
    jitter_buffer_[jitter_buffer_idx_] = jitter_inverted;
    jitter_buffer_idx_ = (jitter_buffer_idx_ + 1) % PCA_JITTER_BUFFER_SIZE;
    if (jitter_buffer_count_ < PCA_JITTER_BUFFER_SIZE) {
        jitter_buffer_count_++;
    }
    
    // Need enough jitter samples for detection
    if (jitter_buffer_count_ < PCA_MOVE_BUFFER_SIZE) {
        last_detection_result_ = false;
        return;
    }
    
    // Count threshold violations in recent window
    size_t move_count = 0;
    
    // Calculate median of jitter buffer
    float sorted_jitter[PCA_JITTER_BUFFER_SIZE];
    std::memcpy(sorted_jitter, jitter_buffer_, jitter_buffer_count_ * sizeof(float));
    std::sort(sorted_jitter, sorted_jitter + jitter_buffer_count_);
    float jitter_median = sorted_jitter[jitter_buffer_count_ / 2];
    
    for (size_t i = 0; i < PCA_MOVE_BUFFER_SIZE; i++) {
        size_t idx = (jitter_buffer_idx_ + PCA_JITTER_BUFFER_SIZE - 1 - i) % PCA_JITTER_BUFFER_SIZE;
        float jitter_val = jitter_buffer_[idx];
        
        // Dual condition (values are scaled by PCA_SCALE)
        if (jitter_val > threshold_ || 
            (jitter_val > jitter_median && jitter_val > 10.0f)) {
            move_count++;
        }
    }
    
    // Motion detected if enough violations
    last_detection_result_ = (move_count >= PCA_OUTLIERS_NUM);
}

void PCADetector::update_state() {
    // Update state based on last detection result
    if (last_detection_result_) {
        state_ = MotionState::MOTION;
    } else {
        state_ = MotionState::IDLE;
    }
}

bool PCADetector::set_threshold(float threshold) {
    // Threshold is now scaled by PCA_SCALE (1000), so valid range is 0.0-10.0
    // matching the MVS threshold range (SEGMENTATION_MIN/MAX_THRESHOLD)
    if (threshold < 0.0f || threshold > 10.0f) {
        ESP_LOGE(TAG, "Invalid threshold: %.3f (must be 0.0-10.0)", threshold);
        return false;
    }
    threshold_ = threshold;
    threshold_externally_set_ = true;
    ESP_LOGI(TAG, "Threshold updated: %.4f", threshold);
    return true;
}

void PCADetector::reset() {
    state_ = MotionState::IDLE;
    csi_buffer_count_ = 0;
    pca_buffer_count_ = 0;
    pca_buffer_idx_ = 0;
    calibration_count_ = 0;
    jitter_buffer_count_ = 0;
    jitter_buffer_idx_ = 0;
    current_jitter_ = 0.0f;
    current_wander_ = 0.0f;
    packet_count_ = 0;
    last_detection_result_ = false;
    // Note: threshold_ and threshold_externally_set_ are preserved across reset
    
    ESP_LOGD(TAG, "Reset");
}

// ============================================================================
// Private Methods
// ============================================================================

void PCADetector::extract_amplitudes(const int8_t* csi_data, size_t len, float* amplitudes) {
    size_t max_sc = len / 2;  // I/Q pairs
    size_t out_idx = 0;
    
    for (size_t sc = 0; sc < max_sc && out_idx < PCA_NUM_SUBCARRIERS; sc += PCA_SUBCARRIER_STEP) {
        // CSI format: [Q, I, Q, I, ...] (Imaginary first, then Real)
        float q = static_cast<float>(csi_data[sc * 2]);
        float i = static_cast<float>(csi_data[sc * 2 + 1]);
        amplitudes[out_idx++] = std::sqrt(i * i + q * q);
    }
    
    // Fill remaining with zeros if needed
    while (out_idx < PCA_NUM_SUBCARRIERS) {
        amplitudes[out_idx++] = 0.0f;
    }
}

float PCADetector::compute_jitter(const float* pca_current) {
    if (pca_buffer_count_ < 2) {
        return 1.0f;  // No past data, assume max correlation (no movement)
    }
    
    float max_corr = 0.0f;
    size_t num_to_check = std::min(pca_buffer_count_ - 1, PCA_MOVE_BUFFER_SIZE - 1);
    
    for (size_t i = 0; i < num_to_check; i++) {
        size_t past_idx = (pca_buffer_idx_ + PCA_BUFFER_SIZE - 1 - i) % PCA_BUFFER_SIZE;
        float corr = std::fabs(pearson_correlation(pca_current, pca_buffer_[past_idx], PCA_NUM_SUBCARRIERS));
        if (corr > max_corr) {
            max_corr = corr;
        }
    }
    
    return max_corr;
}

float PCADetector::compute_wander(const float* pca_current) {
    if (calibration_count_ == 0) {
        return 1.0f;
    }
    
    float max_corr = 0.0f;
    for (size_t i = 0; i < calibration_count_; i++) {
        float corr = std::fabs(pearson_correlation(pca_current, calibration_data_[i], PCA_NUM_SUBCARRIERS));
        if (corr > max_corr) {
            max_corr = corr;
        }
    }
    
    return max_corr;
}

}  // namespace espectre
}  // namespace esphome
