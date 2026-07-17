/*
 * ESPectre - ML Detector Implementation
 *
 * Neural network-based motion detection algorithm.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "ml_detector.h"
#include "ml_weights.h"
#include "threshold.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include "espectre_log.h"

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
    , current_probability_(0.0f)
    , uses_l1_features_(false) {
    threshold_ = clamp_threshold(threshold_, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD);

    // Maintain the L1-delta rings only when the exported model needs them.
    for (uint8_t i = 0; i < ML_MODEL_INPUT_SIZE; i++) {
        if (ml_feature_is_l1(ML_FEATURE_IDS[i])) {
            uses_l1_features_ = true;
            break;
        }
    }
    l1_tracker_.configure(uses_l1_features_ ? l1_delta_capacity_() : 0U);

    ESP_LOGI(TAG, "Initialized (window=%d, threshold=%.2f, l1=%d)",
             window_size_, threshold_, uses_l1_features_ ? 1 : 0);
}

void MLDetector::configure_hampel(bool enabled, uint8_t window_size,
                                  float threshold) {
    BaseDetector::configure_hampel(enabled, window_size, threshold);
    l1_tracker_.configure_hampel(enabled, window_size, threshold);
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
        }
    } else {
        if (current_probability_ <= threshold_) {
            state_ = MotionState::IDLE;
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
    // Reconstruct the L1-delta series (chronological) when the model uses it.
    float delta_series[DETECTOR_MAX_WINDOW_SIZE];
    const uint16_t delta_len = uses_l1_features_ ? l1_tracker_.build_series(delta_series) : 0;

    if (buffer_count_ < window_size_) {
        extract_ml_features_by_id(turbulence_buffer_, buffer_count_,
                                  delta_series, delta_len,
                                  ML_FEATURE_IDS, ML_MODEL_INPUT_SIZE, features_out);
        return;
    }

    // Reconstruct chronological order from the circular buffer.
    // buffer_index_ points to the next write slot, i.e. the oldest sample.
    float ordered_buffer[DETECTOR_MAX_WINDOW_SIZE];
    for (uint16_t i = 0; i < buffer_count_; i++) {
        ordered_buffer[i] = turbulence_buffer_[(buffer_index_ + i) % window_size_];
    }

    extract_ml_features_by_id(ordered_buffer, buffer_count_,
                              delta_series, delta_len,
                              ML_FEATURE_IDS, ML_MODEL_INPUT_SIZE, features_out);
}

// ============================================================================
// L1-DELTA PROFILE PIPELINE
// ============================================================================

uint16_t MLDetector::l1_delta_capacity_() const {
    // window_size profiles yield window_size - lag deltas, matching the Python
    // features.l1_delta_series window semantics.
    return window_size_ > L1_DELTA_LAG
        ? static_cast<uint16_t>(window_size_ - L1_DELTA_LAG)
        : 0;
}

void MLDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                const uint8_t* selected_subcarriers,
                                uint8_t num_subcarriers) {
    if (csi_data == nullptr) {
        ESP_LOGE(TAG, "process_packet: NULL CSI data");
        return;
    }

    float amplitudes[HT20_SELECTED_BAND_SIZE];
    const uint8_t amplitude_count = extract_subcarrier_amplitudes(
        csi_data, csi_len, selected_subcarriers, num_subcarriers,
        amplitudes, HT20_SELECTED_BAND_SIZE);
    process_amplitudes(amplitudes, amplitude_count);

    if (!uses_l1_features_) {
        return;
    }
    l1_tracker_.process(amplitudes, amplitude_count);
}

void MLDetector::clear_buffer() {
    BaseDetector::clear_buffer();
    l1_tracker_.clear();
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

    // Sigmoid with overflow protection on the direct 0-1 probability scale
    if (out < -20.0f) return 0.0f;
    if (out > 20.0f) return ML_METRIC_SCALE;
    return (1.0f / (1.0f + std::exp(-out))) * ML_METRIC_SCALE;
}

}  // namespace espectre
