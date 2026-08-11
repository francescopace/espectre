/*
 * ESPectre - Base Detector Implementation
 *
 * Abstract base class for motion detection algorithms.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "base_detector.h"
#include "csi_format.h"
#include "utils.h"
#include <cstring>
#include <new>
#include "espectre_log.h"

namespace espectre {

static const char *TAG = "BaseDetector";

// ============================================================================
// CONSTRUCTOR / DESTRUCTOR
// ============================================================================

float* BaseDetector::alloc_zeroed_floats(uint16_t count) {
    if (count == 0) {
        return nullptr;
    }
    float* buffer = new (std::nothrow) float[count];
    if (buffer != nullptr) {
        std::memset(buffer, 0, count * sizeof(float));
    }
    return buffer;
}

BaseDetector::BaseDetector(uint16_t window_size)
    : turbulence_buffer_(nullptr)
    , ordered_turbulence_(nullptr)
    , buffer_index_(0)
    , buffer_count_(0)
    , window_size_(window_size)
    , state_(MotionState::IDLE)
    , current_metric_(0.0f)
    , total_packets_(0)
    , packet_index_(0)
    , packet_timestamp_us_(0U)
    , has_packet_timestamp_(false) {

    // Validate and clamp window size
    if (window_size_ < DETECTOR_MIN_WINDOW_SIZE) {
        window_size_ = DETECTOR_MIN_WINDOW_SIZE;
    } else if (window_size_ > DETECTOR_MAX_WINDOW_SIZE) {
        window_size_ = DETECTOR_MAX_WINDOW_SIZE;
    }

    // Allocate turbulence buffer and its chronological reorder scratch
    turbulence_buffer_ = alloc_zeroed_floats(window_size_);
    if (!turbulence_buffer_) {
        ESP_LOGE(TAG, "Failed to allocate turbulence buffer (%d elements)", window_size_);
    }
    ordered_turbulence_ = alloc_zeroed_floats(window_size_);
    if (!ordered_turbulence_) {
        ESP_LOGE(TAG, "Failed to allocate reorder buffer (%d elements)", window_size_);
    }

    // Initialize filters (disabled by default)
    lowpass_filter_init(&lowpass_state_, LOWPASS_CUTOFF_DEFAULT, LOWPASS_SAMPLE_RATE, false);
    hampel_turbulence_init(&hampel_state_, HAMPEL_TURBULENCE_WINDOW_DEFAULT, HAMPEL_TURBULENCE_THRESHOLD_DEFAULT, false);
}

BaseDetector::~BaseDetector() {
    delete[] turbulence_buffer_;
    turbulence_buffer_ = nullptr;
    delete[] ordered_turbulence_;
    ordered_turbulence_ = nullptr;
}

BaseDetector::BaseDetector(BaseDetector&& other) noexcept
    : turbulence_buffer_(other.turbulence_buffer_)
    , ordered_turbulence_(other.ordered_turbulence_)
    , buffer_index_(other.buffer_index_)
    , buffer_count_(other.buffer_count_)
    , window_size_(other.window_size_)
    , state_(other.state_)
    , current_metric_(other.current_metric_)
    , total_packets_(other.total_packets_)
    , packet_index_(other.packet_index_)
    , packet_timestamp_us_(other.packet_timestamp_us_)
    , has_packet_timestamp_(other.has_packet_timestamp_)
    , hampel_state_(other.hampel_state_)
    , lowpass_state_(other.lowpass_state_) {
    // Transfer ownership - null out source pointers
    other.turbulence_buffer_ = nullptr;
    other.ordered_turbulence_ = nullptr;
}

BaseDetector& BaseDetector::operator=(BaseDetector&& other) noexcept {
    if (this != &other) {
        // Free existing resources
        delete[] turbulence_buffer_;
        delete[] ordered_turbulence_;

        // Transfer all state
        turbulence_buffer_ = other.turbulence_buffer_;
        ordered_turbulence_ = other.ordered_turbulence_;
        buffer_index_ = other.buffer_index_;
        buffer_count_ = other.buffer_count_;
        window_size_ = other.window_size_;
        state_ = other.state_;
        current_metric_ = other.current_metric_;
        total_packets_ = other.total_packets_;
        packet_index_ = other.packet_index_;
        packet_timestamp_us_ = other.packet_timestamp_us_;
        has_packet_timestamp_ = other.has_packet_timestamp_;
        lowpass_state_ = other.lowpass_state_;
        hampel_state_ = other.hampel_state_;

        // Transfer ownership - null out source pointers
        other.turbulence_buffer_ = nullptr;
        other.ordered_turbulence_ = nullptr;
    }
    return *this;
}

const float* BaseDetector::ordered_turbulence(uint16_t& count) const {
    count = 0;
    if (turbulence_buffer_ == nullptr || buffer_count_ == 0) {
        return nullptr;
    }

    // Still filling: the ring has not wrapped, so it is already chronological.
    if (buffer_count_ < window_size_) {
        count = buffer_count_;
        return turbulence_buffer_;
    }

    if (ordered_turbulence_ == nullptr) {
        return nullptr;
    }

    // buffer_index_ points to the next write slot, i.e. the oldest sample. The
    // ring is full on this branch, so the split point is known and the copy is
    // two contiguous runs; the modulo-per-element form cost an integer division
    // per sample on every evaluation, and integer division is not single-cycle
    // on the Xtensa and RISC-V parts.
    const uint16_t tail = static_cast<uint16_t>(window_size_ - buffer_index_);
    std::memcpy(ordered_turbulence_, turbulence_buffer_ + buffer_index_,
                static_cast<size_t>(tail) * sizeof(float));
    std::memcpy(ordered_turbulence_ + tail, turbulence_buffer_,
                static_cast<size_t>(buffer_index_) * sizeof(float));
    count = buffer_count_;
    return ordered_turbulence_;
}

// ============================================================================
// VIRTUAL INTERFACE IMPLEMENTATION
// ============================================================================

void BaseDetector::process_packet(const int8_t* csi_data, size_t csi_len,
                                   const uint8_t* selected_subcarriers,
                                   uint8_t num_subcarriers,
                                   int8_t rssi_dbm) {
    if (!csi_data) {
        ESP_LOGE(TAG, "process_packet: null CSI data");
        return;
    }
    if (!turbulence_buffer_) {
        return;
    }
    (void) rssi_dbm;
    
    float amplitudes[HT20_SELECTED_BAND_SIZE];
    const uint8_t amplitude_count = extract_subcarrier_amplitudes(
        csi_data, csi_len, selected_subcarriers, num_subcarriers,
        amplitudes, HT20_SELECTED_BAND_SIZE);
    process_amplitudes(amplitudes, amplitude_count);
}

void BaseDetector::process_amplitudes(const float* amplitudes, uint8_t count) {
    add_turbulence_to_buffer(calculate_spatial_turbulence_from_amplitudes(amplitudes, count));
}

void BaseDetector::reset() {
    clear_evaluation_state_();
    packet_index_ = 0;
    total_packets_ = 0;
    has_packet_timestamp_ = false;

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
    has_packet_timestamp_ = false;
    clear_evaluation_state_();

    // Reset filters
    lowpass_filter_reset(&lowpass_state_);
    hampel_turbulence_init(&hampel_state_, hampel_state_.window_size, 
                           hampel_state_.threshold, hampel_state_.enabled);
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
