/*
 * ESPectre - Classic Detector
 *
 * Non-ML fusion detector: L1-Delta primary with a variance recovery vote.
 *
 * This mirrors src/python/micro_espectre/classic_detector.py:
 * - L1-Delta remains the primary metric and startup gate owner
 * - Moving variance is used only in the ambiguous band below threshold
 * - The variance floor is collected during startup and frozen after calibration
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "base_detector.h"
#include "csi_format.h"
#include "features.h"
#include "l1_delta_tracker.h"
#include "utils.h"
#include <cstddef>
#include <cstdint>

namespace espectre {

constexpr float CLASSIC_DEFAULT_THRESHOLD = 1.0f;
constexpr float CLASSIC_MIN_THRESHOLD = 0.0f;
constexpr float CLASSIC_MAX_THRESHOLD = 10.0f;
constexpr float CLASSIC_STARTUP_THRESHOLD_FACTOR = L1_DELTA_STARTUP_THRESHOLD_FACTOR;

constexpr float CLASSIC_BAND_ALPHA = 0.6f;
constexpr float CLASSIC_RECOVERY_VOTE_RATIO = 3.0f;
constexpr float CLASSIC_RECOVERY_DISPERSION_CUT = 4.0f;

constexpr uint8_t CLASSIC_L1_LAG = L1_DELTA_LAG;
constexpr uint16_t CLASSIC_VARIANCE_FLOOR_MIN = 300;

class ClassicDetector : public BaseDetector {
public:
    ClassicDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE,
                    float threshold = CLASSIC_DEFAULT_THRESHOLD,
                    bool recovery_vote_enabled = true);

    ~ClassicDetector() override = default;

    ClassicDetector(ClassicDetector&& other) noexcept = default;
    ClassicDetector& operator=(ClassicDetector&& other) noexcept = default;

    ClassicDetector(const ClassicDetector&) = delete;
    ClassicDetector& operator=(const ClassicDetector&) = delete;

    void process_packet(const int8_t* csi_data, size_t csi_len,
                        const uint8_t* selected_subcarriers = nullptr,
                        uint8_t num_subcarriers = 0) override;
    void update_state() override;
    void reset() override;
    void clear_buffer() override;
    bool is_ready() const override { return l1_tracker_.count() >= window_size_; }
    float get_motion_metric() const override { return current_l1_metric_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    const char* get_name() const override { return "Classic"; }
    float get_startup_threshold_factor() const override { return CLASSIC_STARTUP_THRESHOLD_FACTOR; }
    bool startup_gate_enabled() const override { return true; }
    void on_startup_calibration_complete() override;
    float get_startup_floor_metric() const override {
        return recovery_vote_configured_ ? current_moving_variance_ : 0.0f;
    }
    void apply_startup_floor(float variance_floor, bool recovery_vote_enabled,
                             uint16_t sample_count) override;

    float get_variance_floor() const { return variance_floor_; }
    bool recovery_vote_enabled() const { return recovery_vote_enabled_; }
    bool recovery_vote_configured() const { return recovery_vote_configured_; }
    float get_last_moving_variance() const { return current_moving_variance_; }

private:
    void clear_l1_state_();
    float calculate_moving_variance_() const;

    float threshold_;
    float current_l1_metric_;
    float current_moving_variance_;

    L1DeltaTracker l1_tracker_;

    uint16_t floor_count_;
    float variance_floor_;
    bool recovery_vote_configured_;
    bool recovery_vote_enabled_;
    bool floor_frozen_;
};

}  // namespace espectre
