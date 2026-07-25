/*
 * ESPectre - Core Helper Unit Tests
 *
 * Targets uncovered helper paths in the reusable core layer.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "test_harness.h"

#include <cmath>
#include <utility>
#include <vector>

#include "classic_detector.h"
#include "ml_detector.h"
#include "csi_features.h"
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

// Feed enough varying packets to wrap the turbulence ring, so the detector
// exercises the chronological reorder buffer and the feature scratch rather
// than the still-filling shortcut.
template <typename Detector>
void fill_past_window(Detector& detector, uint16_t packets) {
    for (uint16_t p = 0; p < packets; ++p) {
        const int8_t magnitude = static_cast<int8_t>(4 + (p % 7));
        auto packet = make_constant_packet(magnitude, static_cast<int8_t>(magnitude + 1));
        detector.process_packet(packet.data(), packet.size(), DEFAULT_SUBCARRIERS,
                                HT20_SELECTED_BAND_SIZE);
    }
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
    // max_subcarrier is the length of magnitudes: indices at or above it are
    // dropped, so the array must be at least that long for the ones below it.
    float magnitudes[10] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f};
    uint8_t invalid_band[] = {0, 7, 9};

    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(nullptr, invalid_band, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, nullptr, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, invalid_band, 0));
    // Only index 0 is below the band limit, and one sample has no spread.
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

namespace {

// Build a payload that is populated everywhere except the guard bins of one
// layout, matching what the radio actually reports.
std::vector<int8_t> make_layout_packet(const uint8_t* null_bins, uint8_t null_count) {
    std::vector<int8_t> packet(HT20_CSI_LEN);
    for (uint16_t bin = 0; bin < HT20_NUM_SUBCARRIERS; ++bin) {
        packet[bin * 2] = static_cast<int8_t>(7 + (bin % 5));
        packet[bin * 2 + 1] = static_cast<int8_t>(-3 - (bin % 4));
    }
    for (uint8_t i = 0; i < null_count; ++i) {
        packet[null_bins[i] * 2] = 0;
        packet[null_bins[i] * 2 + 1] = 0;
    }
    return packet;
}

}  // namespace

void test_ht20_bin_layout_detection_requires_positive_evidence(void) {
    constexpr uint8_t kNullCount = static_cast<uint8_t>(sizeof(HT20_CLASSIC_ONLY_NULL_BINS));

    auto classic = make_layout_packet(HT20_CLASSIC_ONLY_NULL_BINS, kNullCount);
    auto centered = make_layout_packet(HT20_CENTERED_ONLY_NULL_BINS, kNullCount);
    TEST_ASSERT_TRUE(detect_ht20_bin_layout(classic.data(), classic.size()) ==
                     Ht20BinLayout::CLASSIC);
    TEST_ASSERT_TRUE(detect_ht20_bin_layout(centered.data(), centered.size()) ==
                     Ht20BinLayout::CENTERED);

    // An all-zero payload is null under both conventions, so neither wins.
    std::vector<int8_t> empty(HT20_CSI_LEN, 0);
    TEST_ASSERT_TRUE(detect_ht20_bin_layout(empty.data(), empty.size()) ==
                     Ht20BinLayout::UNKNOWN);

    // A single faded guard-adjacent tone withdraws the positive evidence rather
    // than guessing; the capture service latches the previous answer instead.
    auto faded = classic;
    faded[HT20_CENTERED_ONLY_NULL_BINS[0] * 2] = 0;
    faded[HT20_CENTERED_ONLY_NULL_BINS[0] * 2 + 1] = 0;
    TEST_ASSERT_TRUE(detect_ht20_bin_layout(faded.data(), faded.size()) ==
                     Ht20BinLayout::UNKNOWN);

    TEST_ASSERT_TRUE(detect_ht20_bin_layout(nullptr, HT20_CSI_LEN) == Ht20BinLayout::UNKNOWN);
    TEST_ASSERT_TRUE(detect_ht20_bin_layout(classic.data(), HT20_CSI_LEN_SHORT) ==
                     Ht20BinLayout::UNKNOWN);
}

void test_ht20_rotation_maps_classic_onto_centered(void) {
    constexpr uint8_t kNullCount = static_cast<uint8_t>(sizeof(HT20_CLASSIC_ONLY_NULL_BINS));
    auto classic = make_layout_packet(HT20_CLASSIC_ONLY_NULL_BINS, kNullCount);

    std::vector<int8_t> rotated(HT20_CSI_LEN);
    rotate_ht20_classic_to_centered(classic.data(), rotated.data());
    TEST_ASSERT_TRUE(detect_ht20_bin_layout(rotated.data(), rotated.size()) ==
                     Ht20BinLayout::CENTERED);

    // DC moves from bin 0 to HT20_DC_SUBCARRIER, which is what the band assumes.
    for (uint16_t bin = 0; bin < HT20_NUM_SUBCARRIERS; ++bin) {
        const uint16_t source = static_cast<uint16_t>((bin + HT20_DC_SUBCARRIER) % HT20_NUM_SUBCARRIERS);
        TEST_ASSERT_EQUAL_INT8(classic[source * 2], rotated[bin * 2]);
        TEST_ASSERT_EQUAL_INT8(classic[source * 2 + 1], rotated[bin * 2 + 1]);
    }

    // Rotating by half the FFT size is its own inverse.
    std::vector<int8_t> round_trip(HT20_CSI_LEN);
    rotate_ht20_classic_to_centered(rotated.data(), round_trip.data());
    TEST_ASSERT_TRUE(round_trip == classic);
}

void test_threshold_helpers_cover_ranges(void) {
    TEST_ASSERT_TRUE(is_valid_threshold(2.0f, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(NAN, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(0.0f, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(4.0f, 0.1f, 3.0f));

    TEST_ASSERT_EQUAL_FLOAT(0.1f, clamp_threshold(NAN, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.1f, clamp_threshold(-1.0f, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, clamp_threshold(9.0f, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, clamp_threshold(2.0f, 0.1f, 3.0f));

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

void test_startup_threshold_calibrator_weighted_observation_matches_repeated(void) {
    StartupThresholdCalibrator weighted;
    StartupThresholdCalibrator repeated;
    weighted.begin(200, true);
    repeated.begin(200, true);
    const float metrics[] = {0.05f, 0.05f, 0.12f, 0.12f, 0.05f, 0.05f};
    const float floors[] = {0.01f, 0.01f, 0.50f, 0.50f, 0.01f, 0.01f};
    for (size_t i = 0; i < sizeof(metrics) / sizeof(metrics[0]); ++i) {
        weighted.observe(true, metrics[i], floors[i], 25U);
        for (uint16_t packet = 0; packet < 25U; ++packet) {
            repeated.observe(true, metrics[i], floors[i]);
        }
    }

    TEST_ASSERT_EQUAL_INT(static_cast<int>(repeated.packet_count()),
                          static_cast<int>(weighted.packet_count()));
    TEST_ASSERT_EQUAL_INT(static_cast<int>(repeated.ready_packet_count()),
                          static_cast<int>(weighted.ready_packet_count()));
    TEST_ASSERT_EQUAL_FLOAT(repeated.threshold_metric(), weighted.threshold_metric());
    TEST_ASSERT_EQUAL_STRING(repeated.statistic_name(), weighted.statistic_name());
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
    float abs_devs[4];

    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_skewness(sample, 2, 2.0f, 1.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_skewness(sample, 4, 4.0f, 0.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, median_from_sorted(nullptr, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, median_from_sorted(sorted, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 2, 2.0f, 1.0f, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 4, 4.0f, 0.0f, 1));
    // MAD requires a caller-owned sorted view and a scratch that fits.
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_mad(sample, 4, nullptr, abs_devs, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_mad(sample, 4, sorted, nullptr, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_mad(sample, 4, sorted, abs_devs, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_mad(sample, 1, sorted, abs_devs, 4));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, calc_mad(sample, 4, sorted, abs_devs, 4));

    // A scratch that cannot back the sort leaves mad and zcr at zero rather
    // than reading past its end.
    MLStatNeeds needs;
    needs.sorted = true;
    MLSeriesStats stats;
    compute_ml_series_stats(sample, 4, &stats, needs, MLSeriesScratch{});
    TEST_ASSERT_EQUAL_FLOAT(0.0f, stats.mad);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, stats.zcr);

    float sort_scratch[4];
    const MLSeriesScratch scratch{sort_scratch, abs_devs, 4U};
    compute_ml_series_stats(sample, 4, &stats, needs, scratch);
    TEST_ASSERT_EQUAL_FLOAT(2.0f, stats.mad);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_waveform_length(nullptr, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_waveform_length(sample, 1));
    TEST_ASSERT_EQUAL_FLOAT(6.0f, calc_waveform_length(sample, 4));
}

void test_classic_detector_move_semantics_and_base_accessors(void) {
    auto packet = make_constant_packet(3, 4);

    ClassicDetector source(5, 0.75f);
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
    TEST_ASSERT_EQUAL_FLOAT(0.75f, moved.get_threshold());

    // Keep detecting after the move: this drives the reorder buffer and the
    // L1-delta ring, which the move had to carry over.
    const uint16_t window = moved.get_window_size();
    fill_past_window(moved, static_cast<uint16_t>(window + 20));
    moved.update_state();
    TEST_ASSERT_EQUAL(window, moved.get_buffer_count());
    TEST_ASSERT_TRUE(moved.is_ready());
    TEST_ASSERT_FALSE(std::isnan(moved.get_motion_metric()));

    ClassicDetector assigned(7, 4.0f);
    assigned = std::move(moved);
    TEST_ASSERT_NULL(moved.get_turbulence_buffer());
    TEST_ASSERT_TRUE(assigned.is_lowpass_enabled());
    TEST_ASSERT_TRUE(assigned.is_hampel_enabled());
    TEST_ASSERT_EQUAL(window, assigned.get_buffer_count());

    fill_past_window(assigned, 30);
    assigned.update_state();
    TEST_ASSERT_TRUE(assigned.is_ready());
    TEST_ASSERT_FALSE(std::isnan(assigned.get_motion_metric()));
}

void test_ml_detector_move_semantics_and_cv_state(void) {
    auto packet = make_constant_packet(3, 4);

    MLDetector source(6, 6.0f);
    source.process_packet(packet.data(), packet.size(), DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);

    MLDetector moved(std::move(source));
    TEST_ASSERT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_EQUAL_FLOAT(1.0f, moved.get_threshold());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, moved.get_motion_metric());

    // Run inference after the move: this drives the feature scratch block and
    // the reorder buffer, which the move had to carry over.
    const uint16_t window = moved.get_window_size();
    fill_past_window(moved, static_cast<uint16_t>(window + 20));
    moved.update_state();
    TEST_ASSERT_EQUAL(window, moved.get_buffer_count());
    TEST_ASSERT_FALSE(std::isnan(moved.get_motion_metric()));

    MLDetector assigned(10, 7.0f);
    assigned = std::move(moved);
    TEST_ASSERT_NULL(moved.get_turbulence_buffer());
    TEST_ASSERT_EQUAL_FLOAT(1.0f, assigned.get_threshold());
    TEST_ASSERT_EQUAL(window, assigned.get_buffer_count());

    fill_past_window(assigned, 30);
    assigned.update_state();
    TEST_ASSERT_FALSE(std::isnan(assigned.get_motion_metric()));
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_utils_statistical_helpers_cover_edge_cases);
    RUN_TEST(test_utils_spatial_turbulence_handles_invalid_inputs);
    RUN_TEST(test_ht20_bin_layout_detection_requires_positive_evidence);
    RUN_TEST(test_ht20_rotation_maps_classic_onto_centered);
    RUN_TEST(test_threshold_helpers_cover_ranges);
    RUN_TEST(test_startup_threshold_calibrator_gate_disabled_matches_max);
    RUN_TEST(test_startup_threshold_calibrator_gate_accepts_clean_startup);
    RUN_TEST(test_startup_threshold_calibrator_weighted_observation_matches_repeated);
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
