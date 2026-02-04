/*
 * ESPectre - MVS Detector Implementation
 * 
 * Moving Variance Segmentation (MVS) motion detection algorithm.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "mvs_detector.h"
#include "utils.h"
#include <cmath>
#include <utility>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "MVSDetector";

// ============================================================================
// CONSTRUCTOR
// ============================================================================

MVSDetector::MVSDetector(uint16_t window_size, float threshold)
    : BaseDetector(window_size)
    , threshold_(threshold)
    , current_moving_variance_(0.0f) {
    
    // Validate and clamp threshold
    if (threshold_ < MVS_MIN_THRESHOLD) {
        threshold_ = MVS_MIN_THRESHOLD;
    } else if (threshold_ > MVS_MAX_THRESHOLD) {
        threshold_ = MVS_MAX_THRESHOLD;
    }
    
    ESP_LOGI(TAG, "Initialized (window=%d, threshold=%.2f)", window_size_, threshold_);
}

MVSDetector::MVSDetector(MVSDetector&& other) noexcept
    : BaseDetector(std::move(other))
    , threshold_(other.threshold_)
    , current_moving_variance_(other.current_moving_variance_) {
}

MVSDetector& MVSDetector::operator=(MVSDetector&& other) noexcept {
    if (this != &other) {
        BaseDetector::operator=(std::move(other));
        threshold_ = other.threshold_;
        current_moving_variance_ = other.current_moving_variance_;
    }
    return *this;
}

// ============================================================================
// DETECTION LOGIC
// ============================================================================

void MVSDetector::update_state() {
    // Calculate moving variance (lazy evaluation)
    current_moving_variance_ = calculate_moving_variance();
    
    // State machine
    if (state_ == MotionState::IDLE) {
        if (current_moving_variance_ > threshold_) {
            state_ = MotionState::MOTION;
            ESP_LOGV(TAG, "Motion started at packet %lu", (unsigned long)packet_index_);
        }
    } else {
        if (current_moving_variance_ < threshold_) {
            state_ = MotionState::IDLE;
            ESP_LOGV(TAG, "Motion ended at packet %lu", (unsigned long)packet_index_);
        }
    }
}

bool MVSDetector::set_threshold(float threshold) {
    if (std::isnan(threshold) || std::isinf(threshold) ||
        threshold < MVS_MIN_THRESHOLD || threshold > MVS_MAX_THRESHOLD) {
        ESP_LOGE(TAG, "Invalid threshold: %.2f (must be %.1f-%.1f)",
                 threshold, MVS_MIN_THRESHOLD, MVS_MAX_THRESHOLD);
        return false;
    }
    
    threshold_ = threshold;
    ESP_LOGI(TAG, "Threshold updated: %.2f", threshold);
    return true;
}

// ============================================================================
// PRIVATE METHODS
// ============================================================================

float MVSDetector::calculate_moving_variance() const {
    if (buffer_count_ < window_size_) {
        return 0.0f;
    }
    
    return calculate_variance_two_pass(turbulence_buffer_, window_size_);
}

}  // namespace espectre
}  // namespace esphome
