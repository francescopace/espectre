/*
 * ESPectre - MVS Detector
 * 
 * Moving Variance Segmentation (MVS) motion detection algorithm.
 * 
 * Algorithm:
 * 1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
 * 2. Apply optional Hampel filter to remove outliers
 * 3. Apply optional low-pass filter for noise reduction
 * 4. Compute moving variance on turbulence signal
 * 5. Apply configurable threshold for motion segmentation
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

// MVS-specific constants
constexpr float MVS_DEFAULT_THRESHOLD = 1.0f;
// Min threshold lowered to support CV normalization (std/mean produces smaller values)
constexpr float MVS_MIN_THRESHOLD = 1e-9f;
constexpr float MVS_MAX_THRESHOLD = 10.0f;

/**
 * MVS (Moving Variance Segmentation) Detector
 * 
 * Implements the default ESPectre motion detection algorithm.
 * Inherits buffer management from BaseDetector.
 */
class MVSDetector : public BaseDetector {
public:
    /**
     * Constructor
     * 
     * @param window_size Moving variance window size (10-200 packets)
     * @param threshold Motion detection threshold (0.1-10.0)
     */
    MVSDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE, 
                float threshold = MVS_DEFAULT_THRESHOLD);
    
    ~MVSDetector() override = default;
    
    // Move semantics inherited from BaseDetector
    MVSDetector(MVSDetector&& other) noexcept;
    MVSDetector& operator=(MVSDetector&& other) noexcept;
    
    // Disable copy
    MVSDetector(const MVSDetector&) = delete;
    MVSDetector& operator=(const MVSDetector&) = delete;
    
    // ========================================================================
    // BaseDetector interface implementation
    // ========================================================================
    
    void update_state() override;
    float get_motion_metric() const override { return current_moving_variance_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    const char* get_name() const override { return "MVS"; }

private:
    float calculate_moving_variance() const;
    
    float threshold_;
    float current_moving_variance_;
};

}  // namespace espectre
}  // namespace esphome
