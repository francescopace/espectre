/*
 * ESPectre - Classic Detector Implementation
 *
 * L1-Delta primary plus a gated variance recovery vote.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "classic_detector.h"
#include "threshold.h"

#include <algorithm>
#include <cstring>

#include "espectre_log.h"

namespace espectre {

static const char *TAG = "ClassicDetector";

ClassicDetector::ClassicDetector(uint16_t window_size, float threshold,
                                 bool recovery_vote_enabled)
    : BaseDetector(window_size)
    , threshold_(threshold)
    , current_l1_metric_(0.0f)
    , current_moving_variance_(0.0f)
    , floor_count_(0)
    , variance_floor_(0.0f)
    , recovery_vote_configured_(recovery_vote_enabled)
    , recovery_vote_enabled_(false)
    , floor_frozen_(false) {
    threshold_ = clamp_threshold(threshold_, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD);
    l1_tracker_.configure(window_size_);

    ESP_LOGI(TAG, "Initialized (window=%d, lag=%d, threshold=%.2f)",
             window_size_, CLASSIC_L1_LAG, threshold_);
}

void ClassicDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* selected_subcarriers,
                                     uint8_t num_subcarriers) {
    if (!csi_data) {
        ESP_LOGE(TAG, "process_packet: NULL CSI data");
        return;
    }

    float amplitudes[HT20_SELECTED_BAND_SIZE];
    const uint8_t amplitude_count = extract_subcarrier_amplitudes(
        csi_data, csi_len, selected_subcarriers, num_subcarriers,
        amplitudes, HT20_SELECTED_BAND_SIZE);

    if (should_collect_recovery_sample_()) {
        process_amplitudes(amplitudes, amplitude_count);
    } else {
        packet_index_++;
        total_packets_++;
    }

    l1_tracker_.process(amplitudes, amplitude_count);
}

void ClassicDetector::update_state() {
    if (l1_tracker_.count() >= window_size_) {
        current_l1_metric_ = l1_tracker_.mean();
    } else {
        current_l1_metric_ = 0.0f;
    }

    const bool recovery_band = current_l1_metric_ > (CLASSIC_BAND_ALPHA * threshold_) &&
                               current_l1_metric_ <= threshold_;
    const bool variance_needed = recovery_vote_configured_ && buffer_count_ >= window_size_ &&
                                 (!floor_frozen_ || recovery_band);
    current_moving_variance_ = variance_needed ? calculate_moving_variance_() : 0.0f;

    bool motion = false;
    if (is_ready()) {
        if (current_l1_metric_ > threshold_) {
            motion = true;
        } else if (recovery_vote_configured_ && recovery_vote_enabled_ &&
                   floor_count_ >= CLASSIC_VARIANCE_FLOOR_MIN &&
                   current_l1_metric_ > (CLASSIC_BAND_ALPHA * threshold_) &&
                   current_moving_variance_ > (CLASSIC_RECOVERY_VOTE_RATIO * variance_floor_)) {
            motion = true;
        }
    }

    const MotionState next_state = motion ? MotionState::MOTION : MotionState::IDLE;
    state_ = next_state;
}

void ClassicDetector::reset() {
    BaseDetector::reset();
    current_moving_variance_ = 0.0f;
    current_l1_metric_ = 0.0f;
}

void ClassicDetector::clear_buffer() {
    const bool preserve_frozen_floor = recovery_vote_configured_ && floor_frozen_;
    const float preserved_floor = variance_floor_;
    const bool preserved_vote = recovery_vote_enabled_;
    const uint16_t preserved_floor_count = floor_count_;

    BaseDetector::clear_buffer();
    clear_l1_state_();

    if (preserve_frozen_floor) {
        floor_count_ = preserved_floor_count;
        variance_floor_ = preserved_floor;
        recovery_vote_enabled_ = preserved_vote;
    } else {
        floor_count_ = 0;
        variance_floor_ = 0.0f;
        recovery_vote_enabled_ = false;
    }

    floor_frozen_ = preserve_frozen_floor;
    current_moving_variance_ = 0.0f;
}

bool ClassicDetector::set_threshold(float threshold) {
    if (!is_valid_threshold(threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD)) {
        ESP_LOGE(TAG, "Invalid threshold: %.2f (must be %.1f-%.1f)",
                 threshold, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD);
        return false;
    }

    threshold_ = threshold;
    ESP_LOGD(TAG, "Threshold updated: %.2f", threshold_);
    return true;
}

void ClassicDetector::on_startup_calibration_complete() {
    floor_frozen_ = recovery_vote_configured_;
    ESP_LOGD(TAG, "Startup calibration frozen (floor=%.6f, vote=%s, samples=%u)",
             variance_floor_, recovery_vote_enabled_ ? "on" : "off", floor_count_);
}

void ClassicDetector::apply_startup_floor(float variance_floor, bool recovery_vote_enabled,
                                          uint16_t sample_count) {
    if (!recovery_vote_configured_) {
        floor_count_ = 0;
        variance_floor_ = 0.0f;
        recovery_vote_enabled_ = false;
        return;
    }

    floor_count_ = sample_count;
    variance_floor_ = (floor_count_ > 0) ? variance_floor : 0.0f;
    recovery_vote_enabled_ = recovery_vote_configured_ && recovery_vote_enabled &&
                             floor_count_ >= CLASSIC_VARIANCE_FLOOR_MIN;
}

void ClassicDetector::clear_l1_state_() {
    l1_tracker_.clear();
    current_l1_metric_ = 0.0f;
}

float ClassicDetector::calculate_moving_variance_() const {
    return calculate_variance_two_pass(turbulence_buffer_, window_size_);
}

bool ClassicDetector::should_collect_recovery_sample_() const {
    if (!recovery_vote_configured_) {
        return false;
    }
    if (!floor_frozen_) {
        // Startup calibration owns the variance floor and needs every sample.
        return true;
    }
    if (!recovery_vote_enabled_ || floor_count_ < CLASSIC_VARIANCE_FLOOR_MIN) {
        return false;
    }
    return current_l1_metric_ > (CLASSIC_BAND_ALPHA * threshold_) &&
           current_l1_metric_ <= threshold_;
}

}  // namespace espectre
