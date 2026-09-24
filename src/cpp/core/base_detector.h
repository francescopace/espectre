/*
 * ESPectre - Base Detector
 *
 * Abstract base class for motion detection algorithms.
 * Provides shared turbulence buffer management and filtering.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <limits>
#include "detector_types.h"
#include "detector_limits.h"
#include "filters.h"
#include "utils.h"

namespace espectre {

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
 * - update_state(): detection algorithm logic; it assigns `current_metric_`,
 *   which get_motion_metric() returns
 * - get_threshold() / set_threshold(): threshold management
 * - get_name(): detector name for logging
 */
class BaseDetector {
public:
    /**
     * Constructor
     * 
     * @param window_size Buffer window size in the inclusive range defined by
     *        DETECTOR_MIN_WINDOW_SIZE and DETECTOR_MAX_WINDOW_SIZE
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
     * Process one normalized, temporally admitted CSI packet.
     * 
     * Calculates spatial turbulence from CSI data, applies filtering,
     * and stores in circular buffer.
     * 
     * @param csi_data Normalized CSI data (I/Q interleaved)
     * @param csi_len Length of CSI data
     * @param selected_subcarriers Array of subcarrier indices
     * @param num_subcarriers Number of selected subcarriers
     * @param rssi_dbm Link RSSI for this packet, or INT8_MIN when unknown
     */
    virtual void process_packet(const int8_t* csi_data, size_t csi_len,
                                const uint8_t* selected_subcarriers = nullptr,
                                uint8_t num_subcarriers = 0,
                                int8_t rssi_dbm = INT8_MIN);

    /** Supply the monotonic arrival timestamp consumed by time-binned features. */
    void set_packet_timestamp_us(uint64_t timestamp_us) {
        packet_timestamp_us_ = timestamp_us;
        has_packet_timestamp_ = true;
    }
    
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

    /** Return whether all base detector working storage was allocated. */
    virtual bool is_valid() const {
        return turbulence_buffer_ != nullptr && ordered_turbulence_ != nullptr;
    }
    
    /**
     * Check if detector is ready
     *
     * Ready once the window has filled and its valid slots reach the floor set
     * by set_minimum_valid_samples().
     */
    virtual bool is_ready() const {
        return buffer_count_ >= window_size_ &&
               valid_buffer_count_ >= minimum_valid_samples_;
    }

    /** Advance packet-indexed feature rings for absent temporal slots. */
    virtual void advance_missing_slots(uint32_t count);

    /** Set the valid-slot occupancy floor used by `is_ready()`. */
    void set_minimum_valid_samples(uint16_t count) {
        minimum_valid_samples_ = std::max<uint16_t>(
            1U, std::min<uint16_t>(count, window_size_));
    }
    
    /**
     * Get total packets processed
     */
    virtual uint32_t get_total_packets() const { return total_packets_; }
    
    // ========================================================================
    // PURE VIRTUAL INTERFACE (must be implemented by subclasses)
    // ========================================================================
    
    /**
     * Update state machine (call at the detector evaluation interval)
     * 
     * Subclasses implement their detection algorithm here.
     */
    virtual void update_state() = 0;
    
    /**
     * Get current motion metric value
     *
     * Subclasses assign `current_metric_` at the end of their `update_state()`.
     *
     * @return Primary metric, on the detector's 0..1 probability scale
     */
    float get_motion_metric() const { return current_metric_; }

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
     * Get the detector-specific automatic startup multiplier.
     *
     * threshold = threshold_metric x factor, where `threshold_metric` comes
     * from the shared startup calibrator.
     */
    virtual float get_startup_threshold_factor() const { return 1.3f; }

    /**
     * Whether startup calibration uses the calibrator's consistency gate.
     *
     * The gate can end a calibration early on a quiet, motion, quiet pattern.
     * No shipped detector enables it.
     */
    virtual bool startup_gate_enabled() const { return false; }

    /**
     * Whether the calibration evidence collected so far can set a threshold.
     *
     * The runtime asks each time the calibration budget is spent. While the
     * answer is false, the calibration continues in steps of half its initial
     * budget, up to three times that budget. The default always concludes.
     */
    virtual bool startup_calibration_conclusive() const { return true; }

    /**
     * Motion metric above which a calibration evaluation counts as motion.
     *
     * An evaluation above it restarts the calibration window. A recalibration
     * under the setup of the last successful calibration also restarts above
     * the live threshold when that is lower. The default, infinity, leaves
     * calibration without an absolute reference.
     */
    virtual float calibration_motion_ceiling() const {
        return std::numeric_limits<float>::infinity();
    }

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
     * Hook called when a calibration ends without a result.
     *
     * The runtime keeps the threshold in force before the calibration began.
     * Detectors discard the evidence collected since
     * on_startup_calibration_begin() and resume the adaptation they had, or
     * start adapting the threshold in force if no calibration has completed.
     */
    virtual void on_startup_calibration_abandoned() {}

    // ========================================================================
    // FILTER CONFIGURATION
    // ========================================================================
    
    /**
     * Configure low-pass filter
     * 
     * @param enabled Whether to enable the filter
     * @param cutoff_hz Cutoff frequency (5.0-20.0 Hz)
     */
    virtual void configure_lowpass(
        bool enabled, float cutoff_hz = LOWPASS_CUTOFF_DEFAULT);
    
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
     * Get number of window slots filled, including missing slots
     */
    uint16_t get_buffer_count() const { return buffer_count_; }
    /** Get number of filled window slots that hold a measured sample. */
    uint16_t get_valid_buffer_count() const { return valid_buffer_count_; }
    
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
    /**
     * Drop the last evaluation result.
     *
     * Anything that invalidates the window must also invalidate what was
     * derived from it, or the next publish ships a metric computed from
     * samples the detector no longer holds.
     */
    void clear_evaluation_state_() {
        current_metric_ = 0.0f;
        state_ = MotionState::IDLE;
    }

    /** Add the spatial turbulence of one amplitude frame to the ring. */
    void process_amplitudes(const float* amplitudes, uint8_t count);

    /** Timestamp set by set_packet_timestamp_us() for this packet, or `fallback`. */
    uint64_t packet_timestamp_us_or(uint64_t fallback) const {
        return has_packet_timestamp_ ? packet_timestamp_us_ : fallback;
    }

    /**
     * Add turbulence value to buffer (with filtering)
     */
    void add_turbulence_to_buffer(float turbulence);

    /**
     * Allocate a zeroed float buffer on the heap.
     *
     * Shared by the detectors so no feature helper puts a window-sized
     * array on the CSI callback stack.
     *
     * @return nullptr when count is 0 or the allocation fails
     */
    static float* alloc_zeroed_floats(uint16_t count);

    /**
     * View the turbulence ring in chronological order.
     *
     * Returns the ring itself while it is still filling (already in order),
     * and the base-owned reorder buffer once it wraps. Returns nullptr when
     * there is nothing to read or the reorder buffer could not be allocated.
     *
     * @param count Receives the number of valid samples
     */
    const float* ordered_turbulence(uint16_t& count) const;

    /** @name Turbulence ring */
    /** @{ */
    /** Circular buffer of filtered turbulence, `window_size_` entries. */
    float* turbulence_buffer_;
    /** Scratch buffer behind ordered_turbulence(). */
    float* ordered_turbulence_;
    /** Next write position in `turbulence_buffer_`. */
    uint16_t buffer_index_;
    /** See get_buffer_count(). */
    uint16_t buffer_count_;
    /** See get_valid_buffer_count(). */
    uint16_t valid_buffer_count_;
    /** See set_minimum_valid_samples(). */
    uint16_t minimum_valid_samples_;
    uint16_t window_size_;
    /** @} */

    /** @name Evaluation state */
    /** @{ */
    /** State reported by get_state(); clear_evaluation_state_() resets it. */
    MotionState state_;
    /**
     * Metric reported by get_motion_metric() and published by the runtime.
     * Subclasses assign it at the end of update_state().
     */
    float current_metric_;
    /** Packets processed since the last reset(). */
    uint32_t total_packets_;
    /** Position of the current packet since the last reset(). */
    uint32_t packet_index_;
    /** See packet_timestamp_us_or(). */
    uint64_t packet_timestamp_us_;
    bool has_packet_timestamp_;
    /** @} */

    /** Hampel filter applied before the ring. */
    hampel_filter_state_t hampel_state_;
    /** Low-pass filter applied before the ring. */
    lowpass_filter_state_t lowpass_state_;
    
};

}  // namespace espectre
