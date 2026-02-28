/*
 * ESPectre - ML Detector Implementation
 * 
 * Neural network-based motion detection algorithm.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "ml_detector.h"
#include "ml_features.h"
#include "ml_weights.h"
#include <cmath>
#include <algorithm>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "MLDetector";

// ============================================================================
// CONSTRUCTOR
// ============================================================================

MLDetector::MLDetector(uint16_t window_size, float threshold)
    : BaseDetector(window_size)
    , threshold_(threshold)
    , current_probability_(0.0f)
    , current_class_idx_(0) {
    
    // Validate and clamp threshold
    if (threshold_ < ML_MIN_THRESHOLD) {
        threshold_ = ML_MIN_THRESHOLD;
    } else if (threshold_ > ML_MAX_THRESHOLD) {
        threshold_ = ML_MAX_THRESHOLD;
    }
    
    ESP_LOGI(TAG, "Initialized (window=%d, threshold=%.2f)", window_size_, threshold_);
}

MLDetector::MLDetector(MLDetector&& other) noexcept
    : BaseDetector(std::move(other))
    , threshold_(other.threshold_)
    , current_probability_(other.current_probability_) {
}

MLDetector& MLDetector::operator=(MLDetector&& other) noexcept {
    if (this != &other) {
        BaseDetector::operator=(std::move(other));
        threshold_ = other.threshold_;
        current_probability_ = other.current_probability_;
        current_class_idx_ = other.current_class_idx_;
    }
    return *this;
}

// ============================================================================
// DETECTION LOGIC
// ============================================================================

void MLDetector::update_state() {
    if (!is_ready()) {
        current_probability_ = 0.0f;
        return;
    }
    
    // Extract 12 features
    float features[ML_NUM_FEATURES];
    extract_features(features);
    
    // Run MLP inference
    current_probability_ = predict(features);
    
    // State machine
    if (state_ == MotionState::IDLE) {
        if (current_probability_ > threshold_) {
            state_ = MotionState::MOTION;
            const char* label = ML_CLASS_LABELS[current_class_idx_];
            ESP_LOGV(TAG, "Motion started (prob=%.3f, class=%s)", current_probability_, label);
        }
    } else {
        if (current_probability_ <= threshold_) {
            state_ = MotionState::IDLE;
            ESP_LOGV(TAG, "Motion ended (prob=%.3f)", current_probability_);
        }
    }
}

bool MLDetector::set_threshold(float threshold) {
    if (std::isnan(threshold) || std::isinf(threshold) ||
        threshold < ML_MIN_THRESHOLD || threshold > ML_MAX_THRESHOLD) {
        ESP_LOGE(TAG, "Invalid threshold: %.2f (must be %.1f-%.1f)",
                 threshold, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD);
        return false;
    }
    
    threshold_ = threshold;
    ESP_LOGI(TAG, "Threshold updated: %.2f", threshold);
    return true;
}

// ============================================================================
// FEATURE EXTRACTION
// ============================================================================

void MLDetector::extract_features(float* features_out) {
    extract_ml_features(turbulence_buffer_, buffer_count_,
                        amplitude_buffer_, num_amplitudes_,
                        features_out);
}

// ============================================================================
// MLP INFERENCE
// ============================================================================

float MLDetector::predict(const float* features) {
    // Normalize
    float normalized[12];
    for (int i = 0; i < 12; i++) {
        normalized[i] = (features[i] - ML_FEATURE_MEAN[i]) / ML_FEATURE_SCALE[i];
    }

    // Hidden layer (ReLU) - size derived from weight matrix
    constexpr int H1 = sizeof(ML_B1) / sizeof(ML_B1[0]);
    float h1[H1];
    for (int j = 0; j < H1; j++) {
        h1[j] = ML_B1[j];
        for (int i = 0; i < 12; i++) {
            h1[j] += normalized[i] * ML_W1[i][j];
        }
        h1[j] = std::max(0.0f, h1[j]);
    }

    // Output layer (logits)
    float logits[ML_NUM_CLASSES];
    for (int j = 0; j < ML_NUM_CLASSES; j++) {
        logits[j] = ML_B2[j];
        for (int i = 0; i < H1; i++) {
            logits[j] += h1[i] * ML_W2[i][j];
        }
    }

    // Softmax with numerical stability
    float max_logit = logits[0];
    for (int j = 1; j < ML_NUM_CLASSES; j++) {
        if (logits[j] > max_logit) max_logit = logits[j];
    }
    float probs[ML_NUM_CLASSES];
    float sum_exp = 0.0f;
    for (int j = 0; j < ML_NUM_CLASSES; j++) {
        probs[j] = std::exp(logits[j] - max_logit);
        sum_exp += probs[j];
    }
    for (int j = 0; j < ML_NUM_CLASSES; j++) {
        probs[j] /= sum_exp;
    }

    // Store argmax class index for gesture logging
    current_class_idx_ = 0;
    for (int j = 1; j < ML_NUM_CLASSES; j++) {
        if (probs[j] > probs[current_class_idx_]) {
            current_class_idx_ = j;
        }
    }

    // Return probability of being non-idle (1 - prob[class 0])
    // Keeps threshold-based state machine compatible: prob > 0.5 → MOTION
    return 1.0f - probs[0];
}

}  // namespace espectre
}  // namespace esphome
