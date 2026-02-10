/*
 * ESPectre - Base Detector
 * 
 * Abstract base class for motion detection algorithms.
 * Provides shared turbulence buffer management and filtering.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cstddef>
#include "filters.h"
#include "utils.h"

namespace esphome {
namespace espectre {

// ============================================================================
// MOTION STATE
// ============================================================================

enum class MotionState {
    IDLE,       // No motion detected
    MOTION      // Motion in progress
};

// ============================================================================
// DETECTOR CONSTANTS
// ============================================================================

constexpr uint16_t DETECTOR_DEFAULT_WINDOW_SIZE = 75;
constexpr uint16_t DETECTOR_MIN_WINDOW_SIZE = 10;
constexpr uint16_t DETECTOR_MAX_WINDOW_SIZE = 200;

// ============================================================================
// BASE DETECTOR CLASS
// ============================================================================

/**
 * Abstract base class for motion detection algorithms
 * 
 * Provides shared functionality:
 * - Turbulence buffer management (circular buffer)
 * - Hampel and low-pass filtering
 * - CSI processing and spatial turbulence calculation
 * - Amplitude storage for feature extraction
 * 
 * Subclasses must implement:
 * - update_state(): detection algorithm logic
 * - get_motion_metric(): primary detection metric
 * - get_threshold() / set_threshold(): threshold management
 * - get_name(): detector name for logging
 */
class BaseDetector {
public:
    /**
     * Constructor
     * 
     * @param window_size Buffer window size (10-200 packets)
     */
    explicit BaseDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE);
    
    virtual ~BaseDetector();
    
    // Move semantics (Rule of Five - we manage raw pointer)
    BaseDetector(BaseDetector&& other) noexcept;
    BaseDetector& operator=(BaseDetector&& other) noexcept;
    
    // Disable copy (raw pointer ownership)
    BaseDetector(const BaseDetector&) = delete;
    BaseDetector& operator=(const BaseDetector&) = delete;
    
    // ========================================================================
    // VIRTUAL INTERFACE (implemented in base)
    // ========================================================================
    
    /**
     * Process a CSI packet and update internal state
     * 
     * Calculates spatial turbulence from CSI data, applies filtering,
     * and stores in circular buffer. Also stores amplitudes for feature
     * extraction by ML detector.
     * 
     * @param csi_data Raw CSI data (I/Q interleaved)
     * @param csi_len Length of CSI data
     * @param selected_subcarriers Array of subcarrier indices
     * @param num_subcarriers Number of selected subcarriers
     */
    virtual void process_packet(const int8_t* csi_data, size_t csi_len,
                                const uint8_t* selected_subcarriers = nullptr,
                                uint8_t num_subcarriers = 0);
    
    /**
     * Reset detector state
     * 
     * Resets state machine but preserves buffer ("warm" restart).
     */
    virtual void reset();
    
    /**
     * Get current motion state
     */
    virtual MotionState get_state() const { return state_; }
    
    /**
     * Check if detector is ready (buffer filled)
     */
    virtual bool is_ready() const { return buffer_count_ >= window_size_; }
    
    /**
     * Get total packets processed
     */
    virtual uint32_t get_total_packets() const { return total_packets_; }
    
    // ========================================================================
    // PURE VIRTUAL INTERFACE (must be implemented by subclasses)
    // ========================================================================
    
    /**
     * Update state machine (call at publish interval)
     * 
     * Subclasses implement their detection algorithm here.
     */
    virtual void update_state() = 0;
    
    /**
     * Get current motion metric value
     * 
     * @return Primary metric (moving variance for MVS, probability for ML)
     */
    virtual float get_motion_metric() const = 0;
    
    /**
     * Set detection threshold
     * 
     * @param threshold New threshold value
     * @return true if value was accepted
     */
    virtual bool set_threshold(float threshold) = 0;
    
    /**
     * Get current threshold
     */
    virtual float get_threshold() const = 0;
    
    /**
     * Get detector name for logging
     */
    virtual const char* get_name() const = 0;
    
    // ========================================================================
    // FILTER CONFIGURATION
    // ========================================================================
    
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
     * Configure CV normalization mode
     * 
     * CV normalization (std/mean) makes turbulence gain-invariant but reduces
     * sensitivity for contiguous subcarrier bands (P95). When gain is locked,
     * raw std is preferred as amplitudes are already stable.
     * 
     * @param enabled true = CV normalization (std/mean), false = raw std
     */
    void set_cv_normalization(bool enabled);
    
    /**
     * Check if CV normalization is enabled
     */
    bool is_cv_normalization_enabled() const { return use_cv_normalization_; }
    
    /**
     * Clear turbulence buffer (cold restart)
     */
    void clear_buffer();
    
    // ========================================================================
    // BUFFER ACCESSORS (for subclasses and feature extraction)
    // ========================================================================
    
    /**
     * Get turbulence buffer pointer
     */
    const float* get_turbulence_buffer() const { return turbulence_buffer_; }
    
    /**
     * Get number of valid samples in buffer
     */
    uint16_t get_buffer_count() const { return buffer_count_; }
    
    /**
     * Get configured window size
     */
    uint16_t get_window_size() const { return window_size_; }
    
    /**
     * Get last packet amplitudes (for feature extraction)
     */
    const float* get_last_amplitudes() const { return amplitude_buffer_; }
    
    /**
     * Get number of amplitudes stored
     */
    uint8_t get_num_amplitudes() const { return num_amplitudes_; }
    
    /**
     * Get last turbulence value
     */
    float get_last_turbulence() const;
    
    /**
     * Check if low-pass filter is enabled
     */
    bool is_lowpass_enabled() const { return lowpass_state_.enabled; }
    
    /**
     * Check if Hampel filter is enabled
     */
    bool is_hampel_enabled() const { return hampel_state_.enabled; }

protected:
    /**
     * Add turbulence value to buffer (with filtering)
     */
    void add_turbulence_to_buffer(float turbulence);
    
    // Buffer state
    float* turbulence_buffer_;
    float amplitude_buffer_[HT20_SELECTED_BAND_SIZE];  // Last packet amplitudes
    uint8_t num_amplitudes_;
    uint16_t buffer_index_;
    uint16_t buffer_count_;
    uint16_t window_size_;
    
    // Motion state
    MotionState state_;
    uint32_t total_packets_;
    uint32_t packet_index_;
    
    // Filters
    hampel_filter_state_t hampel_state_;
    lowpass_filter_state_t lowpass_state_;
    
    // CV normalization: true = std/mean (gain-invariant), false = raw std
    // Default false: raw std is more sensitive and matches ML model training
    // Set true only for chips without gain lock (e.g., ESP32)
    bool use_cv_normalization_{false};
};

}  // namespace espectre
}  // namespace esphome
