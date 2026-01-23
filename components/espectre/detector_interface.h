/*
 * ESPectre - Detector Interface
 * 
 * Abstract interface for motion detection algorithms.
 * Allows polymorphic use of different detection strategies (MVS, PCA).
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstdint>
#include <cstddef>

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
// DETECTOR INTERFACE
// ============================================================================

/**
 * Abstract interface for motion detection algorithms
 * 
 * All detectors must implement this interface to be used
 * interchangeably in the ESPectre component.
 */
class IDetector {
public:
    virtual ~IDetector() = default;
    
    /**
     * Process a CSI packet and update internal state
     * 
     * @param csi_data Raw CSI data (I/Q interleaved)
     * @param csi_len Length of CSI data
     * @param selected_subcarriers Array of subcarrier indices (may be ignored by some detectors)
     * @param num_subcarriers Number of selected subcarriers
     */
    virtual void process_packet(const int8_t* csi_data, size_t csi_len,
                                const uint8_t* selected_subcarriers = nullptr,
                                uint8_t num_subcarriers = 0) = 0;
    
    /**
     * Update state machine (call at publish interval for lazy evaluation)
     * 
     * Some detectors may calculate metrics lazily for performance.
     * Call this before reading motion state.
     */
    virtual void update_state() = 0;
    
    /**
     * Get current motion state
     * 
     * @return Current motion state (IDLE or MOTION)
     */
    virtual MotionState get_state() const = 0;
    
    /**
     * Get current motion metric value
     * 
     * Returns the primary metric used for motion detection.
     * For MVS: moving variance
     * For PCA: jitter value
     * 
     * @return Current motion metric
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
     * 
     * @return Current threshold value
     */
    virtual float get_threshold() const = 0;
    
    /**
     * Reset detector state
     * 
     * Clears internal buffers and resets state machine.
     */
    virtual void reset() = 0;
    
    /**
     * Check if detector is calibrated/ready
     * 
     * Some detectors need initial calibration period.
     * 
     * @return true if ready for detection
     */
    virtual bool is_ready() const = 0;
    
    /**
     * Get total packets processed
     * 
     * @return Total packet count since initialization
     */
    virtual uint32_t get_total_packets() const = 0;
    
    /**
     * Get detector name for logging
     * 
     * @return Detector name string
     */
    virtual const char* get_name() const = 0;
};

}  // namespace espectre
}  // namespace esphome
