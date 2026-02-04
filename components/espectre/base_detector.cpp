/*
 * ESPectre - Base Detector Implementation
 * 
 * Abstract base class for motion detection algorithms.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "base_detector.h"
#include "utils.h"
#include <cmath>
#include <cstring>
#include <new>
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *TAG = "BaseDetector";

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

BaseDetector::BaseDetector(uint16_t window_size)
    : turbulence_buffer_(nullptr)
    , num_amplitudes_(0)
    , buffer_index_(0)
    , buffer_count_(0)
    , window_size_(window_size)
    , state_(MotionState::IDLE)
    , total_packets_(0)
    , packet_index_(0)
    , gain_compensation_(1.0f) {
    
    // Validate and clamp window size
    if (window_size_ < DETECTOR_MIN_WINDOW_SIZE) {
        window_size_ = DETECTOR_MIN_WINDOW_SIZE;
    } else if (window_size_ > DETECTOR_MAX_WINDOW_SIZE) {
        window_size_ = DETECTOR_MAX_WINDOW_SIZE;
    }
    
    // Allocate turbulence buffer
    turbulence_buffer_ = new (std::nothrow) float[window_size_];
    if (!turbulence_buffer_) {
        ESP_LOGE(TAG, "Failed to allocate turbulence buffer (%d elements)", window_size_);
    } else {
        std::memset(turbulence_buffer_, 0, window_size_ * sizeof(float));
    }
    
    // Initialize amplitude buffer
    std::memset(amplitude_buffer_, 0, sizeof(amplitude_buffer_));
    
    // Initialize filters (disabled by default)
    lowpass_filter_init(&lowpass_state_, LOWPASS_CUTOFF_DEFAULT, LOWPASS_SAMPLE_RATE, false);
    hampel_turbulence_init(&hampel_state_, HAMPEL_TURBULENCE_WINDOW_DEFAULT, HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, false);
}

BaseDetector::~BaseDetector() {
    if (turbulence_buffer_) {
        delete[] turbulence_buffer_;
        turbulence_buffer_ = nullptr;
    }
}

BaseDetector::BaseDetector(BaseDetector&& other) noexcept
    : turbulence_buffer_(other.turbulence_buffer_)
    , num_amplitudes_(other.num_amplitudes_)
    , buffer_index_(other.buffer_index_)
    , buffer_count_(other.buffer_count_)
    , window_size_(other.window_size_)
    , state_(other.state_)
    , total_packets_(other.total_packets_)
    , packet_index_(other.packet_index_)
    , lowpass_state_(other.lowpass_state_)
    , hampel_state_(other.hampel_state_)
    , gain_compensation_(other.gain_compensation_) {
    
    // Copy amplitude buffer
    std::memcpy(amplitude_buffer_, other.amplitude_buffer_, sizeof(amplitude_buffer_));
    
    // Transfer ownership - null out source pointer
    other.turbulence_buffer_ = nullptr;
}

BaseDetector& BaseDetector::operator=(BaseDetector&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        delete[] turbulence_buffer_;
        
        // Transfer all state
        turbulence_buffer_ = other.turbulence_buffer_;
        num_amplitudes_ = other.num_amplitudes_;
        buffer_index_ = other.buffer_index_;
        buffer_count_ = other.buffer_count_;
        window_size_ = other.window_size_;
        state_ = other.state_;
        total_packets_ = other.total_packets_;
        packet_index_ = other.packet_index_;
        lowpass_state_ = other.lowpass_state_;
        hampel_state_ = other.hampel_state_;
        gain_compensation_ = other.gain_compensation_;
        
        // Copy amplitude buffer
        std::memcpy(amplitude_buffer_, other.amplitude_buffer_, sizeof(amplitude_buffer_));
        
        // Transfer ownership - null out source pointer
        other.turbulence_buffer_ = nullptr;
    }
    return *this;
}

// ============================================================================
// VIRTUAL INTERFACE IMPLEMENTATION
// ============================================================================

void BaseDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                   const uint8_t* selected_subcarriers,
                                   uint8_t num_subcarriers) {
    if (!csi_data || !turbulence_buffer_) {
        ESP_LOGE(TAG, "process_packet: NULL pointer");
        return;
    }
    
    // Store amplitudes for feature extraction (before computing turbulence)
    // HT20: 64 subcarriers max
    int total_subcarriers = static_cast<int>(csi_len / 2);
    num_amplitudes_ = 0;
    
    if (selected_subcarriers && num_subcarriers > 0) {
        for (int i = 0; i < num_subcarriers && num_amplitudes_ < DETECTOR_NUM_SUBCARRIERS; i++) {
            int sc_idx = selected_subcarriers[i];
            if (sc_idx >= total_subcarriers) continue;
            
            // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
            float Q = static_cast<float>(csi_data[sc_idx * 2]);      // Imaginary first
            float I = static_cast<float>(csi_data[sc_idx * 2 + 1]);  // Real second
            float amplitude = std::sqrt(I * I + Q * Q);
            
            // Apply gain compensation
            amplitude_buffer_[num_amplitudes_++] = amplitude * gain_compensation_;
        }
    }
    
    // Calculate spatial turbulence with gain compensation
    float turbulence = calculate_spatial_turbulence_from_csi(csi_data, csi_len,
                                                             selected_subcarriers,
                                                             num_subcarriers,
                                                             gain_compensation_);
    
    // Add to buffer with filtering
    add_turbulence_to_buffer(turbulence);
}

void BaseDetector::reset() {
    state_ = MotionState::IDLE;
    packet_index_ = 0;
    total_packets_ = 0;
    
    // Don't clear buffer - preserve "warm" state
}

// ============================================================================
// FILTER CONFIGURATION
// ============================================================================

void BaseDetector::configure_lowpass(bool enabled, float cutoff_hz) {
    lowpass_filter_init(&lowpass_state_, cutoff_hz, LOWPASS_SAMPLE_RATE, enabled);
    ESP_LOGI(TAG, "Low-pass filter %s (cutoff=%.1f Hz)", enabled ? "enabled" : "disabled", cutoff_hz);
}

void BaseDetector::configure_hampel(bool enabled, uint8_t window_size, float threshold) {
    hampel_turbulence_init(&hampel_state_, window_size, threshold, enabled);
    ESP_LOGI(TAG, "Hampel filter %s (window=%d, threshold=%.1f)", 
             enabled ? "enabled" : "disabled", window_size, threshold);
}

void BaseDetector::clear_buffer() {
    if (turbulence_buffer_) {
        std::memset(turbulence_buffer_, 0, window_size_ * sizeof(float));
    }
    buffer_index_ = 0;
    buffer_count_ = 0;
    state_ = MotionState::IDLE;
    
    // Reset filters
    lowpass_filter_reset(&lowpass_state_);
    hampel_turbulence_init(&hampel_state_, hampel_state_.window_size, 
                           hampel_state_.threshold, hampel_state_.enabled);
    
    ESP_LOGD(TAG, "Buffer cleared");
}

// ============================================================================
// BUFFER ACCESSORS
// ============================================================================

float BaseDetector::get_last_turbulence() const {
    if (!turbulence_buffer_ || buffer_count_ == 0) {
        return 0.0f;
    }
    
    int16_t last_idx = static_cast<int16_t>(buffer_index_) - 1;
    if (last_idx < 0) {
        last_idx = window_size_ - 1;
    }
    
    return turbulence_buffer_[last_idx];
}

// ============================================================================
// PROTECTED METHODS
// ============================================================================

void BaseDetector::add_turbulence_to_buffer(float turbulence) {
    // Apply Hampel filter to remove outliers
    float hampel_filtered = hampel_filter_turbulence(&hampel_state_, turbulence);
    
    // Apply low-pass filter for noise reduction
    float filtered_turbulence = lowpass_filter_apply(&lowpass_state_, hampel_filtered);
    
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
