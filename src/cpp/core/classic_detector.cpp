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
    , delta_index_(0)
    , delta_count_(0)
    , l1_packet_count_(0)
    , floor_idx_(0)
    , floor_count_(0)
    , since_refresh_(0)
    , variance_floor_(0.0f)
    , recovery_vote_configured_(recovery_vote_enabled)
    , recovery_vote_enabled_(false)
    , floor_frozen_(false) {
    threshold_ = clamp_threshold(threshold_, CLASSIC_MIN_THRESHOLD, CLASSIC_MAX_THRESHOLD);
    clear_l1_state_();
    std::memset(variance_floor_ring_, 0, sizeof(variance_floor_ring_));

    ESP_LOGI(TAG, "Initialized (window=%d, lag=%d, threshold=%.2f)",
             window_size_, CLASSIC_L1_LAG, threshold_);
}

void ClassicDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                     const uint8_t* selected_subcarriers,
                                     uint8_t num_subcarriers) {
    if (recovery_vote_configured_) {
        BaseDetector::process_packet(csi_data, csi_len, selected_subcarriers, num_subcarriers);
    } else {
        if (!csi_data) {
            ESP_LOGE(TAG, "process_packet: NULL CSI data");
            return;
        }
        packet_index_++;
        total_packets_++;
    }

    l1_packet_count_++;

    float amplitudes[HT20_SELECTED_BAND_SIZE];
    uint8_t amplitude_count = extract_subcarrier_amplitudes(
        csi_data, csi_len, selected_subcarriers, num_subcarriers,
        amplitudes, HT20_SELECTED_BAND_SIZE);

    float profile[HT20_SELECTED_BAND_SIZE];
    uint8_t profile_len = normalize_amplitude_profile(amplitudes, amplitude_count, profile);

    const uint32_t ring_slot = (l1_packet_count_ - 1) % CLASSIC_L1_LAG;
    const float* reference = profile_ring_[ring_slot];
    const uint8_t reference_len = profile_len_[ring_slot];

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

    std::memcpy(profile_ring_[ring_slot], profile, profile_len * sizeof(float));
    profile_len_[ring_slot] = profile_len;
}

void ClassicDetector::update_state() {
    if (delta_count_ >= window_size_) {
        float total = 0.0f;
        for (uint16_t i = 0; i < window_size_; i++) {
            total += delta_ring_[i];
        }
        current_l1_metric_ = total / window_size_;
    } else {
        current_l1_metric_ = 0.0f;
    }

    current_moving_variance_ =
        (recovery_vote_configured_ && buffer_count_ >= window_size_)
            ? calculate_moving_variance_()
            : 0.0f;

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
    const uint16_t preserved_floor_idx = floor_idx_;
    const uint16_t preserved_floor_count = floor_count_;
    float preserved_ring[CLASSIC_VARIANCE_FLOOR_SIZE];
    if (preserve_frozen_floor) {
        std::memcpy(preserved_ring, variance_floor_ring_, sizeof(variance_floor_ring_));
    }

    BaseDetector::clear_buffer();
    clear_l1_state_();

    if (preserve_frozen_floor) {
        std::memcpy(variance_floor_ring_, preserved_ring, sizeof(variance_floor_ring_));
        floor_idx_ = preserved_floor_idx % CLASSIC_VARIANCE_FLOOR_SIZE;
        floor_count_ = std::min<uint16_t>(preserved_floor_count, CLASSIC_VARIANCE_FLOOR_SIZE);
        variance_floor_ = preserved_floor;
        recovery_vote_enabled_ = preserved_vote;
    } else {
        std::memset(variance_floor_ring_, 0, sizeof(variance_floor_ring_));
        floor_idx_ = 0;
        floor_count_ = 0;
        variance_floor_ = 0.0f;
        recovery_vote_enabled_ = false;
    }

    since_refresh_ = 0;
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
        floor_idx_ = 0;
        floor_count_ = 0;
        variance_floor_ = 0.0f;
        recovery_vote_enabled_ = false;
        return;
    }

    floor_count_ = std::min<uint16_t>(sample_count, CLASSIC_VARIANCE_FLOOR_SIZE);
    floor_idx_ = floor_count_ % CLASSIC_VARIANCE_FLOOR_SIZE;
    for (uint16_t i = 0; i < floor_count_; i++) {
        variance_floor_ring_[i] = variance_floor;
    }
    for (uint16_t i = floor_count_; i < CLASSIC_VARIANCE_FLOOR_SIZE; i++) {
        variance_floor_ring_[i] = 0.0f;
    }
    variance_floor_ = (floor_count_ > 0) ? variance_floor : 0.0f;
    recovery_vote_enabled_ = recovery_vote_configured_ && recovery_vote_enabled &&
                             floor_count_ >= CLASSIC_VARIANCE_FLOOR_MIN;
}

void ClassicDetector::clear_l1_state_() {
    std::memset(profile_ring_, 0, sizeof(profile_ring_));
    std::memset(profile_len_, 0, sizeof(profile_len_));
    std::memset(delta_ring_, 0, sizeof(delta_ring_));
    delta_index_ = 0;
    delta_count_ = 0;
    l1_packet_count_ = 0;
    current_l1_metric_ = 0.0f;
}

float ClassicDetector::calculate_moving_variance_() const {
    return calculate_variance_two_pass(turbulence_buffer_, window_size_);
}

void ClassicDetector::push_variance_floor_(float value) {
    variance_floor_ring_[floor_idx_] = value;
    floor_idx_ = (floor_idx_ + 1) % CLASSIC_VARIANCE_FLOOR_SIZE;
    if (floor_count_ < CLASSIC_VARIANCE_FLOOR_SIZE) {
        floor_count_++;
    }
    since_refresh_++;

    if (floor_count_ >= CLASSIC_VARIANCE_FLOOR_MIN && since_refresh_ >= CLASSIC_VARIANCE_FLOOR_REFRESH) {
        refresh_variance_floor_();
        since_refresh_ = 0;
    }
}

void ClassicDetector::refresh_variance_floor_() {
    if (floor_count_ == 0) {
        variance_floor_ = 0.0f;
        recovery_vote_enabled_ = false;
        return;
    }

    std::array<float, CLASSIC_VARIANCE_FLOOR_SIZE> ordered{};
    std::copy(variance_floor_ring_, variance_floor_ring_ + floor_count_, ordered.begin());
    std::sort(ordered.begin(), ordered.begin() + floor_count_);

    if (floor_count_ % 2 == 0) {
        const size_t hi = floor_count_ / 2;
        const size_t lo = hi - 1;
        variance_floor_ = 0.5f * (ordered[lo] + ordered[hi]);
    } else {
        variance_floor_ = ordered[floor_count_ / 2];
    }

    const size_t p99_index = std::min<size_t>(floor_count_ - 1, static_cast<size_t>(0.99f * floor_count_));
    const float p99 = ordered[p99_index];
    recovery_vote_enabled_ = variance_floor_ > 0.0f && (p99 / variance_floor_) < CLASSIC_RECOVERY_DISPERSION_CUT;
}

}  // namespace espectre
