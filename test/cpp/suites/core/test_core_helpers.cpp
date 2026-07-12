/*
 * ESPectre - Core Helper Unit Tests
 *
 * Targets uncovered helper paths in the reusable core layer.
 */

#include "test_harness.h"

#include <cmath>
#include <utility>
#include <vector>

#include "classic_detector.h"
#include "ml_detector.h"
#include "features.h"
#include "threshold.h"
#include "csi_format.h"
#include "utils.h"

using namespace espectre;

namespace {

std::vector<int8_t> make_constant_packet(int8_t i_value, int8_t q_value) {
    std::vector<int8_t> packet(HT20_CSI_LEN);
    for (uint16_t sc = 0; sc < HT20_NUM_SUBCARRIERS; ++sc) {
        packet[sc * 2] = q_value;
        packet[sc * 2 + 1] = i_value;
    }
    return packet;
}

}  // namespace

void test_utils_statistical_helpers_cover_edge_cases(void) {
    float no_values = calculate_mean(nullptr, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, no_values);

    float float_values_even[] = {9.0f, 1.0f, 5.0f, 3.0f};
    TEST_ASSERT_EQUAL_FLOAT(4.0f, calculate_median_float(float_values_even, 4));

    float float_values_odd[] = {7.0f, 1.0f, 3.0f};
    TEST_ASSERT_EQUAL_FLOAT(3.0f, calculate_median_float(float_values_odd, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_median_float(nullptr, 0));

    uint8_t u8_values[] = {9, 1, 5, 3};
    TEST_ASSERT_EQUAL_UINT8(4, calculate_median_u8(u8_values, 4));
    TEST_ASSERT_EQUAL_UINT8(0, calculate_median_u8(nullptr, 0));

    int8_t i8_values[] = {9, -3, 5, 1};
    TEST_ASSERT_EQUAL_INT8(3, calculate_median_i8(i8_values, 4));
    TEST_ASSERT_EQUAL_INT8(0, calculate_median_i8(nullptr, 0));

    TEST_ASSERT_EQUAL_FLOAT(3.0f, apply_cv_normalization(6.0f, 2.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, apply_cv_normalization(6.0f, 0.0f));

    float mean_values[] = {2.0f, 4.0f, 6.0f, 8.0f};
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.4472136f,
                             calculate_turbulence_from_variance(5.0f, mean_values, 4));
}

void test_utils_spatial_turbulence_handles_invalid_inputs(void) {
    float magnitudes[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t invalid_band[] = {0, 7, 9};

    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(nullptr, invalid_band, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, nullptr, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, invalid_band, 0));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, invalid_band, 3, 4));

    float valid = calculate_spatial_turbulence(magnitudes, invalid_band, 3, 10);
    TEST_ASSERT_TRUE(valid >= 0.0f);

    auto packet = make_constant_packet(3, 4);
    uint8_t sparse_band[] = {0, 31, 63, 90};
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(nullptr, packet.size(), sparse_band, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(packet.data(), 1, sparse_band, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(packet.data(), packet.size(), nullptr, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(packet.data(), packet.size(), sparse_band, 0));

    float from_csi = calculate_spatial_turbulence_from_csi(packet.data(), packet.size(), sparse_band, 4);
    TEST_ASSERT_TRUE(from_csi >= 0.0f);

    float negative = -3.0f;
    float positive = 2.0f;
    TEST_ASSERT_TRUE(compare_float_abs(&negative, &positive) > 0);
    TEST_ASSERT_TRUE(compare_float(&negative, &positive) < 0);

    int8_t low = -5;
    int8_t high = 6;
    TEST_ASSERT_TRUE(compare_int8(&low, &high) < 0);
}

void test_threshold_helpers_cover_modes_and_ranges(void) {
    TEST_ASSERT_EQUAL_FLOAT(DEFAULT_ADAPTIVE_FACTOR, get_threshold_factor(ThresholdMode::AUTO));
    TEST_ASSERT_EQUAL_FLOAT(1.0f, get_threshold_factor(ThresholdMode::MIN));

    std::vector<float> empty;
    TEST_ASSERT_EQUAL_FLOAT(1.0f, calculate_max_value(empty));

    std::vector<float> values = {1.0f, 2.0f, 3.0f, 9.0f};
    TEST_ASSERT_EQUAL_FLOAT(9.0f, calculate_max_value(values));

    TEST_ASSERT_TRUE(is_valid_threshold(2.0f, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(NAN, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(0.0f, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(4.0f, 0.1f, 3.0f));

    TEST_ASSERT_EQUAL_FLOAT(0.1f, clamp_threshold(NAN, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.1f, clamp_threshold(-1.0f, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, clamp_threshold(9.0f, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, clamp_threshold(2.0f, 0.1f, 3.0f));

    float adaptive_threshold = 0.0f;
    float factor = 0.0f;
    calculate_adaptive_threshold(values, ThresholdMode::AUTO, adaptive_threshold, factor);
    TEST_ASSERT_EQUAL_FLOAT(DEFAULT_ADAPTIVE_FACTOR, factor);
    TEST_ASSERT_TRUE(adaptive_threshold > 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(calculate_max_value(values) * 1.1f, calculate_adaptive_threshold(values, 1.1f));
}

void test_startup_threshold_calibrator_gate_disabled_matches_max(void) {
    StartupThresholdCalibrator calibrator;
    calibrator.begin(3, false);
    calibrator.observe(true, 0.02f);
    calibrator.observe(true, 0.05f);
    calibrator.observe(true, 0.03f);

    TEST_ASSERT_TRUE(calibrator.is_complete());
    TEST_ASSERT_TRUE(calibrator.is_successful());
    TEST_ASSERT_FALSE(calibrator.is_extending());
    TEST_ASSERT_EQUAL_FLOAT(0.05f, calibrator.threshold_metric());
    TEST_ASSERT_EQUAL_STRING("max", calibrator.statistic_name());
}

void test_startup_threshold_calibrator_gate_accepts_clean_startup(void) {
    StartupThresholdCalibrator calibrator;
    calibrator.begin(60, true);
    for (int i = 0; i < 60; ++i) {
        calibrator.observe(true, (i % 2 == 0) ? 0.05f : 0.048f);
    }

    TEST_ASSERT_TRUE(calibrator.is_complete());
    TEST_ASSERT_TRUE(calibrator.gate_accepted());
    TEST_ASSERT_EQUAL_FLOAT(0.05f, calibrator.threshold_metric());
    TEST_ASSERT_EQUAL_STRING("gated max", calibrator.statistic_name());
}

void test_motion_first_calibrator_accepts_quiet_motion_quiet_before_budget(void) {
    StartupThresholdCalibrator calibrator;
    calibrator.begin(200, true);
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.05f);
    }
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.12f);
    }
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.05f);
    }
    TEST_ASSERT_TRUE(calibrator.is_complete());
    TEST_ASSERT_EQUAL_INT(150, static_cast<int>(calibrator.packet_count()));
    TEST_ASSERT_EQUAL_FLOAT(0.085f, calibrator.threshold_metric());
    TEST_ASSERT_EQUAL_STRING("motion gap midpoint", calibrator.statistic_name());
}

void test_motion_first_short_spike_falls_back_to_quiet_first(void) {
    StartupThresholdCalibrator calibrator;
    calibrator.begin(100, true);
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.05f);
    }
    for (int i = 0; i < 25; ++i) {
        calibrator.observe(true, 0.12f);
    }
    for (int i = 0; i < 25; ++i) {
        calibrator.observe(true, 0.05f);
    }
    TEST_ASSERT_TRUE(calibrator.is_complete());
    TEST_ASSERT_EQUAL_FLOAT(0.05f, calibrator.threshold_metric());
    TEST_ASSERT_EQUAL_STRING("quiet anchor", calibrator.statistic_name());
}

void test_motion_without_return_uses_fallback_inside_budget(void) {
    StartupThresholdCalibrator calibrator;
    calibrator.begin(100, true);
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.05f);
    }
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.12f);
    }
    TEST_ASSERT_TRUE(calibrator.is_complete());
    TEST_ASSERT_EQUAL_FLOAT(0.075f, calibrator.threshold_metric());
    TEST_ASSERT_EQUAL_STRING("quiet anchor", calibrator.statistic_name());
}

void test_motion_without_return_is_stable_at_budget_boundary(void) {
    for (uint16_t target_packets = 100; target_packets <= 102; ++target_packets) {
        StartupThresholdCalibrator calibrator;
        calibrator.begin(target_packets, true);
        for (int i = 0; i < 50; ++i) {
            calibrator.observe(true, 0.05f);
        }
        for (int i = 50; i < target_packets; ++i) {
            calibrator.observe(true, 0.12f);
        }
        TEST_ASSERT_TRUE(calibrator.is_complete());
        TEST_ASSERT_EQUAL_FLOAT(0.075f, calibrator.threshold_metric());
    }
}

void test_motion_first_preserves_validated_quiet_floor_samples(void) {
    StartupThresholdCalibrator calibrator;
    calibrator.begin(500, true);
    for (int i = 0; i < 300; ++i) {
        calibrator.observe(true, 0.05f, 0.01f);
    }
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.12f, 0.50f);
    }
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.05f, 0.01f);
    }

    float floor = 0.0f;
    bool vote_enabled = false;
    uint16_t sample_count = 0;
    calibrator.floor_snapshot(floor, vote_enabled, sample_count);
    TEST_ASSERT_TRUE(calibrator.is_complete());
    TEST_ASSERT_EQUAL_FLOAT(0.01f, floor);
    TEST_ASSERT_TRUE(vote_enabled);
    TEST_ASSERT_TRUE(sample_count >= STARTUP_FLOOR_MIN);
}

void test_detector_startup_gate_traits(void) {
    ClassicDetector classic;
    MLDetector ml;
    TEST_ASSERT_TRUE(classic.startup_gate_enabled());
    TEST_ASSERT_FALSE(ml.startup_gate_enabled());
}

void test_ml_feature_helpers_cover_guard_paths(void) {
    float sample[] = {1.0f, 3.0f, 5.0f, 7.0f};
    float sorted[] = {1.0f, 3.0f, 5.0f, 7.0f};

    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_skewness(sample, 2, 2.0f, 1.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_skewness(sample, 4, 4.0f, 0.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, median_from_sorted(nullptr, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, median_from_sorted(sorted, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 2, 2.0f, 1.0f, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 4, 4.0f, 0.0f, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_mad(sample, ML_MAX_SORT_SIZE + 1));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, calc_mad(sample, 4, sorted));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_waveform_length(nullptr, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_waveform_length(sample, 1));
    TEST_ASSERT_EQUAL_FLOAT(6.0f, calc_waveform_length(sample, 4));
}

void test_classic_detector_move_semantics_and_base_accessors(void) {
    auto packet = make_constant_packet(3, 4);

    ClassicDetector source(5, 2.5f);
    source.configure_lowpass(true, 2.0f);
    source.configure_hampel(true, 5, 2.5f);
    source.process_packet(nullptr, packet.size(), DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);
    source.process_packet(packet.data(), packet.size(), DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);

    TEST_ASSERT_NOT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_EQUAL(1, source.get_buffer_count());

    ClassicDetector moved(std::move(source));
    TEST_ASSERT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_TRUE(moved.is_lowpass_enabled());
    TEST_ASSERT_TRUE(moved.is_hampel_enabled());
    TEST_ASSERT_EQUAL(1, moved.get_total_packets());
    TEST_ASSERT_EQUAL(1, moved.get_buffer_count());
    TEST_ASSERT_EQUAL_FLOAT(2.5f, moved.get_threshold());

    ClassicDetector assigned(7, 4.0f);
    assigned = std::move(moved);
    TEST_ASSERT_NULL(moved.get_turbulence_buffer());
    TEST_ASSERT_TRUE(assigned.is_lowpass_enabled());
    TEST_ASSERT_TRUE(assigned.is_hampel_enabled());
    TEST_ASSERT_EQUAL(1, assigned.get_total_packets());
    TEST_ASSERT_EQUAL(1, assigned.get_buffer_count());
}

void test_ml_detector_move_semantics_and_cv_state(void) {
    auto packet = make_constant_packet(3, 4);

    MLDetector source(6, 6.0f);
    source.process_packet(packet.data(), packet.size(), DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);

    MLDetector moved(std::move(source));
    TEST_ASSERT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_EQUAL_FLOAT(1.0f, moved.get_threshold());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, moved.get_motion_metric());

    MLDetector assigned(10, 7.0f);
    assigned = std::move(moved);
    TEST_ASSERT_NULL(moved.get_turbulence_buffer());
    TEST_ASSERT_EQUAL_FLOAT(1.0f, assigned.get_threshold());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, assigned.get_motion_metric());
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_utils_statistical_helpers_cover_edge_cases);
    RUN_TEST(test_utils_spatial_turbulence_handles_invalid_inputs);
    RUN_TEST(test_threshold_helpers_cover_modes_and_ranges);
    RUN_TEST(test_startup_threshold_calibrator_gate_disabled_matches_max);
    RUN_TEST(test_startup_threshold_calibrator_gate_accepts_clean_startup);
    RUN_TEST(test_motion_first_calibrator_accepts_quiet_motion_quiet_before_budget);
    RUN_TEST(test_motion_first_short_spike_falls_back_to_quiet_first);
    RUN_TEST(test_motion_without_return_uses_fallback_inside_budget);
    RUN_TEST(test_motion_without_return_is_stable_at_budget_boundary);
    RUN_TEST(test_motion_first_preserves_validated_quiet_floor_samples);
    RUN_TEST(test_detector_startup_gate_traits);
    RUN_TEST(test_ml_feature_helpers_cover_guard_paths);
    RUN_TEST(test_classic_detector_move_semantics_and_base_accessors);
    RUN_TEST(test_ml_detector_move_semantics_and_cv_state);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
