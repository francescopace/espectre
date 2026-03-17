#ifndef MVS_DETECTOR_H
#define MVS_DETECTOR_H

#include <Arduino.h>
#include <vector>

/**
 * Motion state enum
 */
enum MotionState {
    IDLE = 0,
    MOTION = 1
};

/**
 * MVS Detector - Moving Variance Segmentation algorithm
 * Adapted from ESPectre components/espectre/mvs_detector.h
 *
 * Detects motion through WiFi CSI multipath interference patterns
 * Accuracy: ~97% in optimal conditions
 */
class MVSDetector {
public:
    /**
     * Constructor
     * @param window_size Number of packets for moving variance calculation (default: 50)
     */
    MVSDetector(size_t window_size = 50);

    /**
     * Process a single CSI packet
     * @param csi_data Raw CSI data (128 bytes: 64 subcarriers × 2 bytes I/Q)
     * @param selected_scs Selected subcarrier indices (12 optimal subcarriers)
     */
    void processPacket(const int8_t* csi_data, const std::vector<uint8_t>& selected_scs);

    /**
     * Update motion state based on current turbulence buffer
     * Call after processPacket()
     */
    void updateState();

    /**
     * Get current motion state
     */
    MotionState getState() const { return state_; }

    /**
     * Get current motion metric (moving variance)
     */
    float getMotionMetric() const { return motion_metric_; }

    /**
     * Get current threshold
     */
    float getThreshold() const { return threshold_; }

    /**
     * Set detection threshold
     * @param threshold Variance threshold (typical range: 0.5-2.0)
     */
    void setThreshold(float threshold) { threshold_ = threshold; }

    /**
     * Check if detector has enough samples to produce valid results
     */
    bool isReady() const { return turbulence_buffer_.size() >= window_size_; }

    /**
     * Get total packets processed
     */
    uint32_t getTotalPackets() const { return total_packets_; }

    /**
     * Get window size
     */
    size_t getWindowSize() const { return window_size_; }

    /**
     * Reset detector state
     */
    void reset();

private:
    /**
     * Calculate spatial turbulence = std(amplitudes) across selected subcarriers
     */
    float calculateTurbulence(const int8_t* csi_data, const std::vector<uint8_t>& selected_scs);

    /**
     * Calculate moving variance of turbulence buffer
     */
    float calculateMovingVariance();

    size_t window_size_;
    float threshold_;
    std::vector<float> turbulence_buffer_;
    MotionState state_;
    float motion_metric_;
    uint32_t total_packets_;
};

#endif
