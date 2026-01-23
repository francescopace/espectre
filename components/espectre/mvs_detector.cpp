/*
 * ESPectre - MVS Detector Implementation
 * 
 * Moving Variance Segmentation (MVS) motion detection algorithm.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "mvs_detector.h"
#include <cmath>
#include <cstring>
#include <new>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "MVSDetector";

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

MVSDetector::MVSDetector(uint16_t window_size, float threshold)
    : window_size_(window_size)
    , threshold_(threshold)
    , state_(MotionState::IDLE)
    , current_moving_variance_(0.0f)
    , total_packets_(0)
    , packet_index_(0)
    , turbulence_buffer_(nullptr)
    , buffer_index_(0)
    , buffer_count_(0) {
    
    // Validate and clamp window size
    if (window_size_ < MVS_MIN_WINDOW_SIZE) {
        window_size_ = MVS_MIN_WINDOW_SIZE;
    } else if (window_size_ > MVS_MAX_WINDOW_SIZE) {
        window_size_ = MVS_MAX_WINDOW_SIZE;
    }
    
    // Validate and clamp threshold
    if (threshold_ < MVS_MIN_THRESHOLD) {
        threshold_ = MVS_MIN_THRESHOLD;
    } else if (threshold_ > MVS_MAX_THRESHOLD) {
        threshold_ = MVS_MAX_THRESHOLD;
    }
    
    // Allocate turbulence buffer
    turbulence_buffer_ = new (std::nothrow) float[window_size_];
    if (!turbulence_buffer_) {
        ESP_LOGE(TAG, "Failed to allocate turbulence buffer (%d elements)", window_size_);
    } else {
        std::memset(turbulence_buffer_, 0, window_size_ * sizeof(float));
    }
    
    // Initialize filters (disabled by default)
    lowpass_filter_init(&lowpass_state_, LOWPASS_CUTOFF_DEFAULT, LOWPASS_SAMPLE_RATE, false);
    hampel_turbulence_init(&hampel_state_, HAMPEL_TURBULENCE_WINDOW_DEFAULT, HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, false);
    
    ESP_LOGI(TAG, "Initialized (window=%d, threshold=%.2f)", window_size_, threshold_);
}

MVSDetector::~MVSDetector() {
    if (turbulence_buffer_) {
        delete[] turbulence_buffer_;
        turbulence_buffer_ = nullptr;
    }
}

MVSDetector::MVSDetector(MVSDetector&& other) noexcept
    : window_size_(other.window_size_),
      threshold_(other.threshold_),
      turbulence_buffer_(other.turbulence_buffer_),
      packet_index_(other.packet_index_),
      buffer_count_(other.buffer_count_),
      total_packets_(other.total_packets_),
      state_(other.state_),
      current_moving_variance_(other.current_moving_variance_),
      lowpass_state_(other.lowpass_state_),
      hampel_state_(other.hampel_state_) {
    // Transfer ownership - null out source pointer
    other.turbulence_buffer_ = nullptr;
}

MVSDetector& MVSDetector::operator=(MVSDetector&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        delete[] turbulence_buffer_;
        
        // Transfer all state
        window_size_ = other.window_size_;
        threshold_ = other.threshold_;
        turbulence_buffer_ = other.turbulence_buffer_;
        packet_index_ = other.packet_index_;
        buffer_count_ = other.buffer_count_;
        total_packets_ = other.total_packets_;
        state_ = other.state_;
        current_moving_variance_ = other.current_moving_variance_;
        lowpass_state_ = other.lowpass_state_;
        hampel_state_ = other.hampel_state_;
        
        // Transfer ownership - null out source pointer
        other.turbulence_buffer_ = nullptr;
    }
    return *this;
}

// ============================================================================
// IDetector INTERFACE
// ============================================================================

void MVSDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                  const uint8_t* selected_subcarriers,
                                  uint8_t num_subcarriers) {
    if (!csi_data || !turbulence_buffer_) {
        ESP_LOGE(TAG, "process_packet: NULL pointer");
        return;
    }
    
    // Calculate spatial turbulence
    float turbulence = calculate_spatial_turbulence_from_csi(csi_data, csi_len,
                                                             selected_subcarriers,
                                                             num_subcarriers);
    
    // Add to buffer with filtering
    add_turbulence_to_buffer(turbulence);
}

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

void MVSDetector::reset() {
    state_ = MotionState::IDLE;
    packet_index_ = 0;
    total_packets_ = 0;
    current_moving_variance_ = 0.0f;
    
    // Don't clear buffer - preserve "warm" state
}

// ============================================================================
// MVS-SPECIFIC METHODS
// ============================================================================

void MVSDetector::configure_lowpass(bool enabled, float cutoff_hz) {
    lowpass_filter_init(&lowpass_state_, cutoff_hz, LOWPASS_SAMPLE_RATE, enabled);
    ESP_LOGI(TAG, "Low-pass filter %s (cutoff=%.1f Hz)", enabled ? "enabled" : "disabled", cutoff_hz);
}

void MVSDetector::configure_hampel(bool enabled, uint8_t window_size, float threshold) {
    hampel_turbulence_init(&hampel_state_, window_size, threshold, enabled);
    ESP_LOGI(TAG, "Hampel filter %s (window=%d, threshold=%.1f)", 
             enabled ? "enabled" : "disabled", window_size, threshold);
}

float MVSDetector::get_last_turbulence() const {
    if (!turbulence_buffer_ || buffer_count_ == 0) {
        return 0.0f;
    }
    
    int16_t last_idx = static_cast<int16_t>(buffer_index_) - 1;
    if (last_idx < 0) {
        last_idx = window_size_ - 1;
    }
    
    return turbulence_buffer_[last_idx];
}

void MVSDetector::clear_buffer() {
    if (turbulence_buffer_) {
        std::memset(turbulence_buffer_, 0, window_size_ * sizeof(float));
    }
    buffer_index_ = 0;
    buffer_count_ = 0;
    current_moving_variance_ = 0.0f;
    state_ = MotionState::IDLE;
    
    // Reset filters
    lowpass_filter_reset(&lowpass_state_);
    hampel_turbulence_init(&hampel_state_, hampel_state_.window_size, 
                           hampel_state_.threshold, hampel_state_.enabled);
    
    ESP_LOGD(TAG, "Buffer cleared");
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

void MVSDetector::add_turbulence_to_buffer(float turbulence) {
    // Apply Hampel filter to remove outliers
    float hampel_filtered = hampel_filter_turbulence(&const_cast<hampel_filter_state_t&>(hampel_state_), turbulence);
    
    // Apply low-pass filter for noise reduction
    float filtered_turbulence = lowpass_filter_apply(&const_cast<lowpass_filter_state_t&>(lowpass_state_), hampel_filtered);
    
    // Add to circular buffer
    turbulence_buffer_[buffer_index_] = filtered_turbulence;
    buffer_index_ = (buffer_index_ + 1) % window_size_;
    if (buffer_count_ < window_size_) {
        buffer_count_++;
    }
    
    packet_index_++;
    total_packets_++;
}

}  // namespace espectre
}  // namespace esphome
