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
#include "features.h"
#include "utils.h"
#include <array>
#include <cstddef>
#include <cstdint>

namespace esphome {
namespace espectre {

constexpr float CLASSIC_DEFAULT_THRESHOLD = 1.0f;
constexpr float CLASSIC_MIN_THRESHOLD = 0.0f;
constexpr float CLASSIC_MAX_THRESHOLD = 10.0f;
constexpr float CLASSIC_STARTUP_THRESHOLD_FACTOR = L1_DELTA_STARTUP_THRESHOLD_FACTOR;

constexpr float CLASSIC_BAND_ALPHA = 0.6f;
constexpr float CLASSIC_RECOVERY_VOTE_RATIO = 3.0f;
constexpr float CLASSIC_RECOVERY_DISPERSION_CUT = 4.0f;

constexpr uint8_t CLASSIC_L1_LAG = L1_DELTA_LAG;
constexpr uint16_t CLASSIC_VARIANCE_FLOOR_SIZE = 1000;
constexpr uint16_t CLASSIC_VARIANCE_FLOOR_MIN = 300;
constexpr uint16_t CLASSIC_VARIANCE_FLOOR_REFRESH = 100;

class ClassicDetector : public BaseDetector {
public:
    ClassicDetector(uint16_t window_size = DETECTOR_DEFAULT_WINDOW_SIZE,
                    float threshold = CLASSIC_DEFAULT_THRESHOLD);

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
    bool is_ready() const override { return delta_count_ >= window_size_; }
    float get_motion_metric() const override { return current_l1_metric_; }
    bool set_threshold(float threshold) override;
    float get_threshold() const override { return threshold_; }
    const char* get_name() const override { return "Classic"; }
    float get_startup_threshold_factor() const override { return CLASSIC_STARTUP_THRESHOLD_FACTOR; }
    bool startup_gate_enabled() const override { return true; }
    void on_startup_calibration_complete() override;
    float get_startup_floor_metric() const override { return current_moving_variance_; }
    void apply_startup_floor(float variance_floor, bool recovery_vote_enabled,
                             uint16_t sample_count) override;

    float get_variance_floor() const { return variance_floor_; }
    bool recovery_vote_enabled() const { return recovery_vote_enabled_; }
    float get_last_moving_variance() const { return current_moving_variance_; }

private:
    void clear_l1_state_();
    float calculate_moving_variance_() const;
    void push_variance_floor_(float value);
    void refresh_variance_floor_();

    float threshold_;
    float current_l1_metric_;
    float current_moving_variance_;

    float profile_ring_[CLASSIC_L1_LAG][HT20_SELECTED_BAND_SIZE];
    uint8_t profile_len_[CLASSIC_L1_LAG];

    float delta_ring_[DETECTOR_MAX_WINDOW_SIZE];
    uint16_t delta_index_;
    uint16_t delta_count_;
    uint32_t l1_packet_count_;

    float variance_floor_ring_[CLASSIC_VARIANCE_FLOOR_SIZE];
    uint16_t floor_idx_;
    uint16_t floor_count_;
    uint16_t since_refresh_;
    float variance_floor_;
    bool recovery_vote_enabled_;
    bool floor_frozen_;
};

}  // namespace espectre
}  // namespace esphome
