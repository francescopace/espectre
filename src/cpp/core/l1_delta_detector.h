/*
 * ESPectre - L1-Delta Detector
 *
 * Normalized amplitude-profile displacement motion detection algorithm.
 *
 * Algorithm:
 * 1. Extract subcarrier amplitudes per packet and normalize by their mean
 *    (per-packet gain invariance, same rationale as the CV turbulence path)
 * 2. d = mean absolute difference vs the profile L1_DELTA_LAG packets earlier
 * 3. Motion metric = mean of d over the sliding window
 * 4. Apply configurable threshold for motion segmentation
 *
 * Motion decorrelates the multipath profile coherently across subcarriers,
 * while receiver noise keeps d on a stable floor. Offline benchmarks show the
 * quiet level of this metric varies <=1.3x across sessions (vs up to 14.5x
 * for the MVS moving variance); see docs/ALGORITHMS.md and docs/EXPERIMENTS.md.
 *
 * Aligned with the Python reference implementation in
 * src/python/micro_espectre/l1_delta_detector.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "base_detector.h"
#include "utils.h"
#include <cstdint>
#include <cstddef>

namespace esphome {
namespace espectre {

// L1-Delta-specific constants
constexpr float L1_DELTA_DEFAULT_THRESHOLD = 1.0f;
constexpr float L1_DELTA_MIN_THRESHOLD = 0.0f;
constexpr float L1_DELTA_MAX_THRESHOLD = 10.0f;

// Profile comparison lag in packets (~100 ms at 100 pps): long enough for
// body motion to displace the multipath profile, short enough to track it.
constexpr uint8_t L1_DELTA_LAG = 10;

// Startup calibration multiplier (benchmark-tuned for this metric;
// MVS keeps the shared 1.3 default).
constexpr float L1_DELTA_STARTUP_THRESHOLD_FACTOR = 1.1f;

/**
 * L1-Delta (Normalized Profile Displacement) Detector
 *
 * Keeps the shared BaseDetector turbulence pipeline for telemetry parity and
 * maintains its own allocation-free profile/delta rings for the L1 metric.
 */
class L1DeltaDetector : public BaseDetector {
public:
    /**
     * Constructor
     *
     * @param window_size Metric averaging window size (10-200 packets)
     * @param threshold Motion detection threshold (0.0-10.0)
     */
    L1DeltaDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE,
                    float threshold = L1_DELTA_DEFAULT_THRESHOLD);

    ~L1DeltaDetector() override = default;

    // Move semantics inherited from BaseDetector (all own state is in-place arrays)
    L1DeltaDetector(L1DeltaDetector&& other) noexcept = default;
    L1DeltaDetector& operator=(L1DeltaDetector&& other) noexcept = default;

    // Disable copy
    L1DeltaDetector(const L1DeltaDetector&) = delete;
    L1DeltaDetector& operator=(const L1DeltaDetector&) = delete;

    // ========================================================================
    // BaseDetector interface implementation
    // ========================================================================

    void process_packet(const int8_t* csi_data, size_t csi_len,
                        const uint8_t* selected_subcarriers = nullptr,
                        uint8_t num_subcarriers = 0) override;
    void update_state() override;
    void reset() override;
    void clear_buffer() override;
    bool is_ready() const override { return delta_count_ >= window_size_; }
    float get_motion_metric() const override { return current_metric_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    const char* get_name() const override { return "L1D"; }
    float get_startup_threshold_factor() const override { return L1_DELTA_STARTUP_THRESHOLD_FACTOR; }
    // The tight quiet floor of this metric enables the startup consistency
    // gate (threshold.h); validated for this metric only, not for MVS.
    bool startup_gate_enabled() const override { return true; }

private:
    void clear_l1_state_();

    float threshold_;
    float current_metric_;

    // Ring of the last L1_DELTA_LAG normalized profiles (0 length = invalid).
    float profile_ring_[L1_DELTA_LAG][HT20_SELECTED_BAND_SIZE];
    uint8_t profile_len_[L1_DELTA_LAG];

    // Ring of the last window_size d values (fixed capacity, no allocation).
    float delta_ring_[DETECTOR_MAX_WINDOW_SIZE];
    uint16_t delta_index_;
    uint16_t delta_count_;

    uint32_t l1_packet_count_;
};

}  // namespace espectre
}  // namespace esphome
