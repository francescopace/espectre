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
    , current_probability_(0.0f)
    , uses_l1_features_(false)
    , delta_index_(0)
    , delta_count_(0)
    , l1_packet_count_(0) {
    threshold_ = clamp_threshold(threshold_, ML_MIN_THRESHOLD, ML_MAX_THRESHOLD);

    // Maintain the L1-delta rings only when the exported model needs them.
    for (uint8_t i = 0; i < ML_MODEL_INPUT_SIZE; i++) {
        if (ml_feature_is_l1(ML_FEATURE_IDS[i])) {
            uses_l1_features_ = true;
            break;
        }
    }
    clear_l1_state_();

    ESP_LOGI(TAG, "Initialized (window=%d, threshold=%.2f, l1=%d)",
             window_size_, threshold_, uses_l1_features_ ? 1 : 0);
}

MLDetector::MLDetector(MLDetector&& other) noexcept
    : BaseDetector(std::move(other))
    , threshold_(other.threshold_)
    , current_probability_(other.current_probability_)
    , uses_l1_features_(other.uses_l1_features_)
    , delta_index_(other.delta_index_)
    , delta_count_(other.delta_count_)
    , l1_packet_count_(other.l1_packet_count_) {
    std::memcpy(profile_ring_, other.profile_ring_, sizeof(profile_ring_));
    std::memcpy(profile_len_, other.profile_len_, sizeof(profile_len_));
    std::memcpy(delta_ring_, other.delta_ring_, sizeof(delta_ring_));
}

MLDetector& MLDetector::operator=(MLDetector&& other) noexcept {
    if (this != &other) {
        BaseDetector::operator=(std::move(other));
        threshold_ = other.threshold_;
        current_probability_ = other.current_probability_;
        uses_l1_features_ = other.uses_l1_features_;
        delta_index_ = other.delta_index_;
        delta_count_ = other.delta_count_;
        l1_packet_count_ = other.l1_packet_count_;
        std::memcpy(profile_ring_, other.profile_ring_, sizeof(profile_ring_));
        std::memcpy(profile_len_, other.profile_len_, sizeof(profile_len_));
        std::memcpy(delta_ring_, other.delta_ring_, sizeof(delta_ring_));
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
    // Reconstruct the L1-delta series (chronological) when the model uses it.
    float delta_series[DETECTOR_MAX_WINDOW_SIZE];
    const uint16_t delta_len = uses_l1_features_ ? build_delta_series(delta_series) : 0;

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
    // Shared turbulence pipeline (buffer + filters) feeds the turbulence
    // features and telemetry.
    BaseDetector::process_packet(csi_data, csi_len, selected_subcarriers, num_subcarriers);

    if (!uses_l1_features_) {
        return;
    }

    const uint16_t capacity = l1_delta_capacity_();
    if (capacity == 0) {
        return;
    }

    l1_packet_count_++;

    float amplitudes[HT20_SELECTED_BAND_SIZE];
    uint8_t amplitude_count = extract_subcarrier_amplitudes(
        csi_data, csi_len, selected_subcarriers, num_subcarriers,
        amplitudes, HT20_SELECTED_BAND_SIZE);

    float profile[HT20_SELECTED_BAND_SIZE];
    uint8_t profile_len = normalize_amplitude_profile(amplitudes, amplitude_count, profile);

    const uint32_t ring_slot = (l1_packet_count_ - 1) % L1_DELTA_LAG;
    const float* reference = profile_ring_[ring_slot];
    const uint8_t reference_len = profile_len_[ring_slot];

    // Warmup or malformed packets have no comparable lagged profile.
    if (profile_len > 0 && reference_len == profile_len) {
        float total = 0.0f;
        for (uint8_t i = 0; i < profile_len; i++) {
            float diff = profile[i] - reference[i];
            total += diff >= 0.0f ? diff : -diff;
        }
        const float delta = total / profile_len;

        delta_ring_[delta_index_] = delta;
        delta_index_ = (delta_index_ + 1) % capacity;
        if (delta_count_ < capacity) {
            delta_count_++;
        }
    }

    // Store the current profile in the lag ring.
    std::memcpy(profile_ring_[ring_slot], profile, profile_len * sizeof(float));
    profile_len_[ring_slot] = profile_len;
}

uint16_t MLDetector::build_delta_series(float* out) const {
    const uint16_t capacity = l1_delta_capacity_();
    if (capacity == 0 || delta_count_ == 0) {
        return 0;
    }
    // delta_index_ points to the next write slot: the oldest sample once full.
    const uint16_t start = (delta_count_ < capacity) ? 0 : delta_index_;
    for (uint16_t i = 0; i < delta_count_; i++) {
        out[i] = delta_ring_[(start + i) % capacity];
    }
    return delta_count_;
}

void MLDetector::clear_buffer() {
    BaseDetector::clear_buffer();
    clear_l1_state_();
}

void MLDetector::clear_l1_state_() {
    std::memset(profile_ring_, 0, sizeof(profile_ring_));
    std::memset(profile_len_, 0, sizeof(profile_len_));
    std::memset(delta_ring_, 0, sizeof(delta_ring_));
    delta_index_ = 0;
    delta_count_ = 0;
    l1_packet_count_ = 0;
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
}  // namespace esphome
