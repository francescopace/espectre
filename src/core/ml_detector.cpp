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
static_assert(ML_MODEL_INPUT_SIZE == ML_NUM_FEATURES,
              "Exported model input size must match extracted ML feature count");

// ============================================================================
// CONSTRUCTOR
// ============================================================================

MLDetector::MLDetector(uint16_t window_size, float threshold)
    : BaseDetector(window_size)
    , threshold_(threshold)
    , current_probability_(0.0f) {
    threshold_ = clamp_threshold(threshold_, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD);
    
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
    
    // Extract ML features expected by the exported model
    float features[ML_NUM_FEATURES];
    extract_features(features);
    
    // Run MLP inference
    current_probability_ = predict(features);
    
    // State machine
    if (state_ == MotionState::IDLE) {
        if (current_probability_ > threshold_) {
            state_ = MotionState::MOTION;
            ESP_LOGV(TAG, "Motion started (prob=%.3f)", current_probability_);
        }
    } else {
        if (current_probability_ <= threshold_) {
            state_ = MotionState::IDLE;
            ESP_LOGV(TAG, "Motion ended (prob=%.3f)", current_probability_);
        }
    }
}

bool MLDetector::set_threshold(float threshold) {
    if (!is_valid_threshold(threshold, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD)) {
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
    if (buffer_count_ < window_size_) {
        extract_ml_features(turbulence_buffer_, buffer_count_, features_out);
        return;
    }

    // Reconstruct chronological order from the circular buffer.
    // buffer_index_ points to the next write slot, i.e. the oldest sample.
    float ordered_buffer[DETECTOR_MAX_WINDOW_SIZE];
    for (uint16_t i = 0; i < buffer_count_; i++) {
        ordered_buffer[i] = turbulence_buffer_[(buffer_index_ + i) % window_size_];
    }

    extract_ml_features(ordered_buffer, buffer_count_, features_out);
}

// ============================================================================
// MLP INFERENCE
// ============================================================================

float MLDetector::predict(const float* features) {
    constexpr size_t kBufferSize =
        (ML_MAX_LAYER_WIDTH > ML_MODEL_INPUT_SIZE) ? ML_MAX_LAYER_WIDTH : ML_MODEL_INPUT_SIZE;
    float buffer_a[kBufferSize] = {0.0f};
    float buffer_b[kBufferSize] = {0.0f};

    // Normalize features using pre-computed mean and scale
    for (int i = 0; i < ML_MODEL_INPUT_SIZE; i++) {
        buffer_a[i] = (features[i] - ML_FEATURE_MEAN[i]) / ML_FEATURE_SCALE[i];
    }

    float *current = buffer_a;
    float *next = buffer_b;
    float out = 0.0f;

    for (int layer = 0; layer < ML_MODEL_NUM_LAYERS; layer++) {
        const int in_size = ML_MODEL_LAYER_INPUT_SIZES[layer];
        const int out_size = ML_MODEL_LAYER_OUTPUT_SIZES[layer];
        const float *weights = ML_MODEL_WEIGHTS[layer];
        const float *biases = ML_MODEL_BIASES[layer];
        const bool is_output_layer = (layer == ML_MODEL_NUM_LAYERS - 1);

        for (int j = 0; j < out_size; j++) {
            float val = biases[j];
            for (int i = 0; i < in_size; i++) {
                val += current[i] * weights[i * out_size + j];
            }

            if (is_output_layer) {
                out = val;
            } else {
                next[j] = std::max(0.0f, val);
            }
        }

        if (!is_output_layer) {
            std::swap(current, next);
        }
    }

    // Temperature scaling keeps the published score more gradual
    // without changing the default 5.0 decision boundary.
    out /= ML_TEMPERATURE;

    // Sigmoid with overflow protection and scaling to 0-10 range
    if (out < -20.0f) return 0.0f;
    if (out > 20.0f) return ML_METRIC_SCALE;
    return (1.0f / (1.0f + std::exp(-out))) * ML_METRIC_SCALE;
}

}  // namespace espectre
}  // namespace esphome
