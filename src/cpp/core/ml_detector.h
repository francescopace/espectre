/*
 * ESPectre - ML Detector
 * 
 * Neural network-based motion detection algorithm.
 * 
 * Algorithm:
 * 1. Calculate spatial turbulence per packet using CV normalization
 *    (`std/mean`)
 * 2. Apply optional Hampel filter to remove outliers
 * 3. Apply optional low-pass filter for noise reduction
 * 4. Extract statistical features from turbulence buffer
 * 5. Run MLP inference using exported architecture metadata
 * 6. Compare probability to threshold for motion detection
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "base_detector.h"
#include "features.h"
#include <cstdint>
#include <cstddef>

namespace esphome {
namespace espectre {

// ML-specific constants
constexpr float ML_DEFAULT_THRESHOLD = 5.0f;
constexpr float ML_MIN_THRESHOLD = 0.0f;
constexpr float ML_MAX_THRESHOLD = 10.0f;
constexpr float ML_METRIC_SCALE = 10.0f;
constexpr float ML_TEMPERATURE = 5.0f;

/**
 * ML (Machine Learning) Detector
 * 
 * Neural network-based motion detector using MLP inference.
 * Inherits buffer management from BaseDetector.
 */
class MLDetector : public BaseDetector {
public:
    /**
     * Constructor
     * 
     * @param window_size Feature extraction window size (10-200 packets)
     * @param threshold Motion detection threshold (0.0-10.0 on the shared runtime scale)
     */
    MLDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE, 
               float threshold = ML_DEFAULT_THRESHOLD);
    
    ~MLDetector() override = default;
    
    // Move semantics inherited from BaseDetector
    MLDetector(MLDetector&& other) noexcept;
    MLDetector& operator=(MLDetector&& other) noexcept;
    
    // Disable copy
    MLDetector(const MLDetector&) = delete;
    MLDetector& operator=(const MLDetector&) = delete;
    
    // ========================================================================
    // BaseDetector interface implementation
    // ========================================================================

    void process_packet(const int8_t* csi_data, size_t csi_len,
                        const uint8_t* selected_subcarriers = nullptr,
                        uint8_t num_subcarriers = 0) override;
    void update_state() override;
    void clear_buffer() override;
    float get_motion_metric() const override { return current_probability_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    const char* get_name() const override { return "ML"; }

private:
    /**
     * Extract ML features from the turbulence buffer and L1-delta series
     */
    void extract_features(float* features_out);

    /**
     * Reconstruct the L1-delta series in chronological order.
     *
     * @param out Destination buffer (at least DETECTOR_MAX_WINDOW_SIZE)
     * @return Number of valid delta samples written
     */
    uint16_t build_delta_series(float* out) const;

    /**
     * Reset the L1-delta profile and delta rings (cold clear).
     */
    void clear_l1_state_();

    /**
     * L1-delta ring capacity for this window: window_size - lag (0 if window
     * is not larger than the lag). Sized so the series matches Python
     * features.l1_delta_series (window_size profiles -> window_size - lag deltas).
     */
    uint16_t l1_delta_capacity_() const;
    
    /**
     * Run MLP inference on features.
     *
     * The hidden-layer layout is defined by the auto-generated
     * `ml_weights.h` metadata rather than hardcoded in this class.
     *
     * @param features Feature vector expected by the exported model
     * @return Scaled motion metric (0.0-10.0 on the shared runtime scale)
     */
    float predict(const float* features);

    float threshold_;
    float current_probability_;

    // L1-delta profile-displacement state, maintained only when the exported
    // model actually uses L1-delta features (checked against ML_FEATURE_IDS).
    // Mirrors the shared L1-delta tracker rings; keep aligned with the Python
    // features.l1_delta_series reference.
    bool uses_l1_features_;
    float profile_ring_[L1_DELTA_LAG][HT20_SELECTED_BAND_SIZE];
    uint8_t profile_len_[L1_DELTA_LAG];
    float delta_ring_[DETECTOR_MAX_WINDOW_SIZE];
    uint16_t delta_index_;
    uint16_t delta_count_;
    uint32_t l1_packet_count_;
};

}  // namespace espectre
}  // namespace esphome
