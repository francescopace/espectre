/*
 * ESPectre - L1-Delta Detector Implementation
 *
 * Normalized amplitude-profile displacement motion detection algorithm.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "l1_delta_detector.h"
#include "utils.h"
#include <cstring>
#include "espectre_log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "L1DeltaDetector";

// ============================================================================
// CONSTRUCTOR
// ============================================================================

L1DeltaDetector::L1DeltaDetector(uint16_t window_size, float threshold)
    : BaseDetector(window_size)
    , threshold_(threshold)
    , current_metric_(0.0f)
    , delta_index_(0)
    , delta_count_(0)
    , l1_packet_count_(0) {
    threshold_ = clamp_threshold(threshold_, L1_DELTA_MIN_THRESHOLD, L1_DELTA_MAX_THRESHOLD);
    clear_l1_state_();

    ESP_LOGI(TAG, "Initialized (window=%d, lag=%d, threshold=%.2f)",
             window_size_, L1_DELTA_LAG, threshold_);
}

// ============================================================================
// DETECTION LOGIC
// ============================================================================

void L1DeltaDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* selected_subcarriers,
                                     uint8_t num_subcarriers) {
    // Keep the shared turbulence pipeline running (telemetry and logging
    // parity with the other detectors).
    BaseDetector::process_packet(csi_data, csi_len, selected_subcarriers, num_subcarriers);

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
        delta_index_ = (delta_index_ + 1) % window_size_;
        if (delta_count_ < window_size_) {
            delta_count_++;
        }
    }

    // Store the current profile in the lag ring.
    std::memcpy(profile_ring_[ring_slot], profile, profile_len * sizeof(float));
    profile_len_[ring_slot] = profile_len;
}

void L1DeltaDetector::update_state() {
    // Match MVS semantics: not ready until the metric window is full.
    if (delta_count_ >= window_size_) {
        float total = 0.0f;
        for (uint16_t i = 0; i < window_size_; i++) {
            total += delta_ring_[i];
        }
        current_metric_ = total / window_size_;
    } else {
        current_metric_ = 0.0f;
    }

    // State machine (same shape as MVS)
    if (state_ == MotionState::IDLE) {
        if (current_metric_ > threshold_) {
            state_ = MotionState::MOTION;
            ESP_LOGV(TAG, "Motion started at packet %lu", (unsigned long)packet_index_);
        }
    } else {
        if (current_metric_ < threshold_) {
            state_ = MotionState::IDLE;
            ESP_LOGV(TAG, "Motion ended at packet %lu", (unsigned long)packet_index_);
        }
    }
}

void L1DeltaDetector::reset() {
    BaseDetector::reset();
    // Warm restart: keep profile/delta rings so re-detection stays fast.
}

void L1DeltaDetector::clear_buffer() {
    BaseDetector::clear_buffer();
    clear_l1_state_();
}

bool L1DeltaDetector::set_threshold(float threshold) {
    if (!is_valid_threshold(threshold, L1_DELTA_MIN_THRESHOLD, L1_DELTA_MAX_THRESHOLD)) {
        ESP_LOGE(TAG, "Invalid threshold: %.2f (must be %.1f-%.1f)",
                 threshold, L1_DELTA_MIN_THRESHOLD, L1_DELTA_MAX_THRESHOLD);
        return false;
    }

    threshold_ = threshold;
    ESP_LOGD(TAG, "Threshold updated: %.2f", threshold);
    return true;
}

// ============================================================================
// PRIVATE METHODS
// ============================================================================

void L1DeltaDetector::clear_l1_state_() {
    std::memset(profile_ring_, 0, sizeof(profile_ring_));
    std::memset(profile_len_, 0, sizeof(profile_len_));
    std::memset(delta_ring_, 0, sizeof(delta_ring_));
    delta_index_ = 0;
    delta_count_ = 0;
    current_metric_ = 0.0f;
    l1_packet_count_ = 0;
}

}  // namespace espectre
}  // namespace esphome
