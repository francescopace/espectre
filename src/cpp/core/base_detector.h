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
#include "detector_limits.h"
#include "filters.h"
#include "utils.h"

namespace espectre {

// ============================================================================
// MOTION STATE
// ============================================================================

enum class MotionState {
    IDLE,       // No motion detected
    MOTION      // Motion in progress
};

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
     * and stores in circular buffer.
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
     * @return Primary metric (classic motion metric, ML probability, etc.)
     */
    virtual float get_motion_metric() const = 0;
    
    /**
     * Set detection threshold
     * 
     * @param threshold New threshold value
     * @return true if value was accepted
     */
    virtual bool set_threshold(float threshold) = 0;

    /** Apply a detector-specific startup-calibrated threshold. */
    virtual bool set_adaptive_threshold(float threshold) { return set_threshold(threshold); }
    
    /**
     * Get current threshold
     */
    virtual float get_threshold() const = 0;
    
    /**
     * Get detector name for logging
     */
    virtual const char* get_name() const = 0;

    /**
     * Get the detector-specific startup multiplier for AUTO threshold mode
     *
     * threshold = threshold_metric x factor. Matches the Python
     * runtime's detector STARTUP_THRESHOLD_FACTOR convention, where
     * `threshold_metric` comes from the shared startup calibrator.
     */
    virtual float get_startup_threshold_factor() const { return 1.3f; }

    /**
     * Whether startup calibration uses the consistency gate (threshold.h)
     *
     * Enabled only for detectors with a tight quiet floor (l1_delta).
     * Matches the Python runtime's detector STARTUP_GATE convention.
     */
    virtual bool startup_gate_enabled() const { return false; }

    /** Hook called immediately before startup calibration begins. */
    virtual void on_startup_calibration_begin() {}

    /**
     * Hook called when startup calibration completes successfully.
     *
     * Detectors can freeze session-specific state here before the runtime
     * performs its warm clear between calibration and steady-state detection.
     */
    virtual void on_startup_calibration_complete() {}

    /**
     * Auxiliary startup metric used to build detector-specific frozen state.
     *
     * Classic uses the moving-variance metric to build its startup floor.
     * Detectors that do not need a startup floor can keep the default 0.0f.
     */
    virtual float get_startup_floor_metric() const { return 0.0f; }

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
    virtual void configure_hampel(
        bool enabled, uint8_t window_size = HAMPEL_TURBULENCE_WINDOW_DEFAULT,
        float threshold = HAMPEL_TURBULENCE_THRESHOLD_DEFAULT);
    
    /**
     * Clear turbulence buffer (cold restart)
     *
     * Virtual so detectors with additional state (e.g. L1-Delta profile
     * rings) can extend the cold clear.
     */
    virtual void clear_buffer();
    
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
    void process_amplitudes(const float* amplitudes, uint8_t count);

    /**
     * Add turbulence value to buffer (with filtering)
     */
    void add_turbulence_to_buffer(float turbulence);
    
    // Buffer state
    float* turbulence_buffer_;
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
    
};

}  // namespace espectre
