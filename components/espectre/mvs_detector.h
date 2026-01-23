/*
 * ESPectre - MVS Detector
 * 
 * Moving Variance Segmentation (MVS) motion detection algorithm.
 * 
 * Algorithm:
 * 1. Calculate spatial turbulence (std of subcarrier amplitudes) per packet
 * 2. Apply optional Hampel filter to remove outliers
 * 3. Apply optional low-pass filter for noise reduction
 * 4. Compute moving variance on turbulence signal
 * 5. Apply configurable threshold for motion segmentation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "detector_interface.h"
#include "utils.h"
#include "filters.h"
#include <cstdint>
#include <cstddef>

namespace esphome {
namespace espectre {

// MVS-specific constants
constexpr uint16_t MVS_DEFAULT_WINDOW_SIZE = 50;
constexpr uint16_t MVS_MIN_WINDOW_SIZE = 10;
constexpr uint16_t MVS_MAX_WINDOW_SIZE = 200;
constexpr float MVS_DEFAULT_THRESHOLD = 1.0f;
constexpr float MVS_MIN_THRESHOLD = 0.1f;
constexpr float MVS_MAX_THRESHOLD = 10.0f;

/**
 * MVS (Moving Variance Segmentation) Detector
 * 
 * Implements the default ESPectre motion detection algorithm.
 */
class MVSDetector : public IDetector {
public:
    /**
     * Constructor
     * 
     * @param window_size Moving variance window size (10-200 packets)
     * @param threshold Motion detection threshold (0.1-10.0)
     */
    MVSDetector(uint16_t window_size = MVS_DEFAULT_WINDOW_SIZE, 
                float threshold = MVS_DEFAULT_THRESHOLD);
    
    ~MVSDetector() override;
    
    // Move semantics (Rule of Five - we manage raw pointer)
    MVSDetector(MVSDetector&& other) noexcept;
    MVSDetector& operator=(MVSDetector&& other) noexcept;
    
    // Disable copy (raw pointer ownership)
    MVSDetector(const MVSDetector&) = delete;
    MVSDetector& operator=(const MVSDetector&) = delete;
    
    // IDetector interface
    void process_packet(const int8_t* csi_data, size_t csi_len,
                        const uint8_t* selected_subcarriers = nullptr,
                        uint8_t num_subcarriers = 0) override;
    
    void update_state() override;
    MotionState get_state() const override { return state_; }
    float get_motion_metric() const override { return current_moving_variance_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    void reset() override;
    bool is_ready() const override { return buffer_count_ >= window_size_; }
    uint32_t get_total_packets() const override { return total_packets_; }
    const char* get_name() const override { return "MVS"; }
    
    // MVS-specific configuration
    
    /**
     * Configure low-pass filter
     * 
     * @param enabled Whether to enable the filter
     * @param cutoff_hz Cutoff frequency (5.0-20.0 Hz)
     */
    void configure_lowpass(bool enabled, float cutoff_hz = LOWPASS_CUTOFF_DEFAULT);
    
    /**
     * Configure Hampel filter
     * 
     * @param enabled Whether to enable the filter
     * @param window_size Window size (3-11)
     * @param threshold MAD multiplier threshold
     */
    void configure_hampel(bool enabled, uint8_t window_size = HAMPEL_TURBULENCE_WINDOW_DEFAULT,
                          float threshold = HAMPEL_TURBULENCE_THRESHOLD_DEFAULT);
    
    /**
     * Get window size
     */
    uint16_t get_window_size() const { return window_size_; }
    
    /**
     * Get last turbulence value
     */
    float get_last_turbulence() const;
    
    /**
     * Clear turbulence buffer (cold restart)
     */
    void clear_buffer();
    
    /**
     * Check if low-pass filter is enabled
     */
    bool is_lowpass_enabled() const { return lowpass_state_.enabled; }
    
    /**
     * Check if Hampel filter is enabled
     */
    bool is_hampel_enabled() const { return hampel_state_.enabled; }

private:
    float calculate_moving_variance() const;
    void add_turbulence_to_buffer(float turbulence);
    
    // Configuration
    uint16_t window_size_;
    float threshold_;
    
    // State
    MotionState state_;
    float current_moving_variance_;
    uint32_t total_packets_;
    uint32_t packet_index_;
    
    // Turbulence buffer (circular)
    float* turbulence_buffer_;
    uint16_t buffer_index_;
    uint16_t buffer_count_;
    
    // Filters
    lowpass_filter_state_t lowpass_state_;
    hampel_filter_state_t hampel_state_;
};

}  // namespace espectre
}  // namespace esphome
