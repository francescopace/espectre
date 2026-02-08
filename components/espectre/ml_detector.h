/*
 * ESPectre - ML Detector
 * 
 * Neural network-based motion detection algorithm.
 * 
 * Algorithm:
 * 1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
 * 2. Apply optional Hampel filter to remove outliers
 * 3. Apply optional low-pass filter for noise reduction
 * 4. Extract 12 statistical features from turbulence buffer
 * 5. Run MLP inference (12 -> 16 -> 8 -> 1)
 * 6. Compare probability to threshold for motion detection
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "base_detector.h"
#include <cstdint>
#include <cstddef>

namespace esphome {
namespace espectre {

// ML-specific constants
constexpr float ML_DEFAULT_THRESHOLD = 0.5f;
constexpr float ML_MIN_THRESHOLD = 0.0f;
constexpr float ML_MAX_THRESHOLD = 1.0f;

// Fixed subcarriers for ML (12 evenly distributed across 64, excluding guard bands and DC)
// These must match the subcarriers used during model training
constexpr uint8_t ML_SUBCARRIERS[12] = {11, 14, 17, 21, 24, 28, 31, 35, 39, 42, 46, 49};

/**
 * ML (Machine Learning) Detector
 * 
 * Neural network-based motion detector using MLP inference.
 * Inherits buffer management from BaseDetector.
 */
class MLDetector : public BaseDetector {
public:
    /**
     * Constructor
     * 
     * @param window_size Feature extraction window size (10-200 packets)
     * @param threshold Motion probability threshold (0.0-1.0)
     */
    MLDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE, 
               float threshold = ML_DEFAULT_THRESHOLD);
    
    ~MLDetector() override = default;
    
    // Move semantics inherited from BaseDetector
    MLDetector(MLDetector&& other) noexcept;
    MLDetector& operator=(MLDetector&& other) noexcept;
    
    // Disable copy
    MLDetector(const MLDetector&) = delete;
    MLDetector& operator=(const MLDetector&) = delete;
    
    // ========================================================================
    // BaseDetector interface implementation
    // ========================================================================
    
    void update_state() override;
    float get_motion_metric() const override { return current_probability_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    const char* get_name() const override { return "ML"; }

private:
    /**
     * Extract 12 features from turbulence buffer
     */
    void extract_features(float* features_out);
    
    /**
     * Run MLP inference on features
     * 
     * Architecture: 12 -> 16 (ReLU) -> 8 (ReLU) -> 1 (Sigmoid)
     * 
     * @param features Normalized feature vector (12 values)
     * @return Motion probability (0.0-1.0)
     */
    float predict(const float* features);
    
    float threshold_;
    float current_probability_;
};

}  // namespace espectre
}  // namespace esphome
