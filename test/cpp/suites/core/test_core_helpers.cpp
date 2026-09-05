/*
 * ESPectre - Core Helper Unit Tests
 *
 * Targets uncovered helper paths in the reusable core layer.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <cmath>
#include <utility>
#include <vector>

#include "lightweight_detector.h"
#include "high_accuracy_detector.h"
#include "csi_features.h"
#include "ml_feature_trackers.h"
#include "threshold.h"
#include "temporal_csi_sampler.h"
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

std::vector<int8_t> make_trajectory_packet(uint8_t step, int gain) {
    std::vector<int8_t> packet(HT20_CSI_LEN, 0);
    const float angle = 1.5f * 3.14159265358979323846f *
                        static_cast<float>(step) / 11.0f;
    for (uint8_t subband = 0U; subband < CHANNEL_SHAPE_SUBBAND_COUNT; subband++) {
        const float position = 3.14159265358979323846f *
                               (static_cast<float>(subband) + 0.5f) / 8.0f;
        const float value = 40.0f + 18.0f *
            (std::cos(angle) * std::cos(position) +
             std::sin(angle) * std::cos(2.0f * position));
        const int8_t amplitude = static_cast<int8_t>(
            std::lround(value) * gain);
        for (uint8_t offset = 0U; offset < CHANNEL_SHAPE_SUBBAND_SIZE; offset++) {
            const uint8_t live_index = static_cast<uint8_t>(
                subband * CHANNEL_SHAPE_SUBBAND_SIZE + offset);
            packet[HT20_LIVE_BINS[live_index] * 2U + 1U] = amplitude;
        }
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

void test_required_amplitudes_preserve_selected_and_aggregated_bands(void) {
    auto packet = make_trajectory_packet(7U, 1);
    const uint8_t custom[] = {255U, 0U, 32U, 63U, 60U, 4U, 7U, 8U,
                             9U, 10U, 11U, 12U, 13U, 14U, 15U};
    for (const bool aggregated : {false, true}) {
        for (const size_t length : {size_t{18U}, size_t{114U}, size_t{HT20_CSI_LEN}}) {
            for (const bool use_default : {false, true}) {
                const uint8_t *band = use_default ? DEFAULT_SUBCARRIERS : custom;
                const uint8_t band_count = use_default ? HT20_SELECTED_BAND_SIZE : sizeof(custom);
                float reference[HT20_NUM_SUBCARRIERS]{};
                float optimized[HT20_NUM_SUBCARRIERS]{};
                const uint8_t count = extract_packet_subcarrier_amplitudes(
                    packet.data(), length, reference, HT20_NUM_SUBCARRIERS);
                fill_packet_subcarrier_energies(packet.data(), length, optimized, HT20_NUM_SUBCARRIERS);
                detail::required_energies_to_amplitudes<TURB_IQR_AGGREGATION_WIDTH>(
                    optimized, count, band, band_count, aggregated);
                float expected[HT20_SELECTED_BAND_SIZE]{};
                float actual[HT20_SELECTED_BAND_SIZE]{};
                const uint8_t selected = select_subcarrier_amplitudes(
                    reference, count, band, band_count, expected, HT20_SELECTED_BAND_SIZE);
                TEST_ASSERT_EQUAL(selected, select_subcarrier_amplitudes(
                    optimized, count, band, band_count, actual, HT20_SELECTED_BAND_SIZE));
                for (uint8_t i = 0U; i < selected; ++i) TEST_ASSERT_EQUAL_FLOAT(expected[i], actual[i]);
                if (aggregated) {
                    const uint8_t adjacent = select_adjacent_aggregated_subcarrier_amplitudes(
                        reference, count, band, band_count, TURB_IQR_AGGREGATION_WIDTH,
                        expected, HT20_SELECTED_BAND_SIZE);
                    TEST_ASSERT_EQUAL(adjacent, select_adjacent_aggregated_subcarrier_amplitudes(
                        optimized, count, band, band_count, TURB_IQR_AGGREGATION_WIDTH,
                        actual, HT20_SELECTED_BAND_SIZE));
                    for (uint8_t i = 0U; i < adjacent; ++i) TEST_ASSERT_EQUAL_FLOAT(expected[i], actual[i]);
                }
            }
        }
    }
}

void test_l1_reconfiguration_and_moves_preserve_profile_history(void) {
    L1DeltaTracker reused;
    for (const uint16_t lag : {uint16_t{1U}, L1_DELTA_LAG_MAX, uint16_t{L1_DELTA_LAG}}) {
        reused.configure(20U, lag);
        L1DeltaTracker fresh;
        fresh.configure(20U, lag);
        for (uint16_t packet = 0U; packet < 2U * lag + 20U; ++packet) {
            float amplitudes[HT20_SELECTED_BAND_SIZE];
            for (uint8_t tone = 0U; tone < HT20_SELECTED_BAND_SIZE; ++tone) {
                amplitudes[tone] = 1U + (packet * (tone + 1U)) % 23U;
            }
            if (packet % 7U == 0U) {
                reused.advance_missing_slots(2U);
                fresh.advance_missing_slots(2U);
            }
            reused.process(amplitudes, HT20_SELECTED_BAND_SIZE);
            fresh.process(amplitudes, HT20_SELECTED_BAND_SIZE);
            L1DeltaTracker moved(std::move(reused));
            reused = std::move(moved);
            TEST_ASSERT_EQUAL(fresh.count(), reused.count());
            TEST_ASSERT_EQUAL_FLOAT(fresh.delta_lag_ratio(), reused.delta_lag_ratio());
        }
        reused.configure(0U, lag);
        reused.advance_missing_slots(3U);
        TEST_ASSERT_EQUAL(0U, reused.count());
    }
}

void test_channel_shape_trajectory_is_gain_and_stutter_invariant(void) {
    ChannelShapeTrajectoryTracker baseline;
    ChannelShapeTrajectoryTracker gained;
    baseline.configure(true);
    gained.configure(true);
    for (uint8_t step = 0U; step < 12U; step++) {
        const auto base_packet = make_trajectory_packet(step, 1);
        const auto gained_packet = make_trajectory_packet(step, 2);
        const uint64_t timestamp = static_cast<uint64_t>(step) * CHANNEL_SHAPE_BIN_US;
        baseline.process_packet(base_packet.data(), base_packet.size(), timestamp);
        gained.process_packet(gained_packet.data(), gained_packet.size(), timestamp);
        gained.process_packet(gained_packet.data(), gained_packet.size(), timestamp + 20000U);
    }
    TEST_ASSERT_TRUE(baseline.coherent_innovation_energy() > 0.0f);
    TEST_ASSERT_TRUE(baseline.excess_path() > 0.0f);
    TEST_ASSERT_TRUE(baseline.subband_kendall_lag_excess() > 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, baseline.coherent_innovation_energy(),
        gained.coherent_innovation_energy());
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, baseline.excess_path(), gained.excess_path());
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, baseline.subband_kendall_lag_excess(),
        gained.subband_kendall_lag_excess());
}

void test_trajectory_duplicate_packets_expire_old_motion(void) {
    ChannelShapeTrajectoryTracker tracker;
    tracker.configure(true);
    for (uint8_t step = 0U; step < 12U; ++step) {
        const auto packet = make_trajectory_packet(step, 1);
        tracker.process_packet(packet.data(), packet.size(),
                               static_cast<uint64_t>(step) * CHANNEL_SHAPE_BIN_US);
    }
    TEST_ASSERT_TRUE(tracker.shape_spread_subband() > 0.0f);
    const auto repeated = make_trajectory_packet(11U, 1);
    for (uint8_t step = 12U; step < 40U; ++step) {
        tracker.process_packet(repeated.data(), repeated.size(),
                               static_cast<uint64_t>(step) * CHANNEL_SHAPE_BIN_US);
    }
    TEST_ASSERT_EQUAL_FLOAT(0.0f, tracker.coherent_innovation_energy());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, tracker.excess_path());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, tracker.shape_spread_subband());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, tracker.subband_kendall_lag_excess());

    for (uint8_t step = 40U; step < 52U; ++step) {
        const auto packet = make_trajectory_packet(step - 40U, 1);
        tracker.process_packet(packet.data(), packet.size(),
                               static_cast<uint64_t>(step) * CHANNEL_SHAPE_BIN_US);
    }
    TEST_ASSERT_TRUE(tracker.shape_spread_subband() > 0.0f);
}

void test_shared_packet_frame_matches_direct_trajectory_tracker(void) {
    ChannelShapeTrajectoryTracker direct_trajectory;
    ChannelShapeTrajectoryTracker shared_trajectory;
    direct_trajectory.configure(true);
    shared_trajectory.configure(true);

    for (uint8_t step = 0U; step < 12U; step++) {
        const auto packet = make_trajectory_packet(step, 1);
        float packet_values[HT20_NUM_SUBCARRIERS]{};
        for (uint8_t subcarrier = 0U;
             subcarrier < HT20_NUM_SUBCARRIERS; subcarrier++) {
            const float imag = static_cast<float>(packet[subcarrier * 2U]);
            const float real = static_cast<float>(packet[subcarrier * 2U + 1U]);
            packet_values[subcarrier] = real * real + imag * imag;
        }
        const uint64_t timestamp =
            static_cast<uint64_t>(step) * CHANNEL_SHAPE_BIN_US;
        direct_trajectory.process_packet(
            packet.data(), packet.size(), timestamp);
        shared_trajectory.process_packet(
            packet.data(), packet.size(), timestamp,
            packet_values, HT20_NUM_SUBCARRIERS);
    }

    float direct_innovation = 0.0f;
    float direct_excess = 0.0f;
    float direct_spread = 0.0f;
    float shared_innovation = 0.0f;
    float shared_excess = 0.0f;
    float shared_spread = 0.0f;
    direct_trajectory.trajectory_features(
        direct_innovation, direct_excess, direct_spread);
    shared_trajectory.trajectory_features(
        shared_innovation, shared_excess, shared_spread);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, direct_innovation, shared_innovation);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, direct_excess, shared_excess);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, direct_spread, shared_spread);
    TEST_ASSERT_TRUE(shared_spread > 0.0f);
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, shared_innovation,
        shared_trajectory.coherent_innovation_energy());
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, shared_excess, shared_trajectory.excess_path());
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, shared_spread, shared_trajectory.shape_spread_subband());
    TEST_ASSERT_FLOAT_WITHIN(
        1e-6f, shared_trajectory.subband_kendall_lag_excess(),
        direct_trajectory.subband_kendall_lag_excess());
}

void test_utils_statistical_helpers_cover_edge_cases(void) {
    float no_values = calculate_mean(nullptr, 0);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, no_values);

    float float_values_even[] = {9.0f, 1.0f, 5.0f, 3.0f};
    TEST_ASSERT_EQUAL_FLOAT(4.0f, calculate_median_float(float_values_even, 4));

    float float_values_odd[] = {7.0f, 1.0f, 3.0f};
    TEST_ASSERT_EQUAL_FLOAT(3.0f, calculate_median_float(float_values_odd, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_median_float(nullptr, 0));

    TEST_ASSERT_EQUAL_FLOAT(3.0f, apply_cv_normalization(6.0f, 2.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, apply_cv_normalization(6.0f, 0.0f));

    float mean_values[] = {2.0f, 4.0f, 6.0f, 8.0f};
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.4472136f,
                             calculate_turbulence_from_variance(5.0f, mean_values, 4));
}

void test_temporal_csi_sampler_matches_fixed_slot_contract(void) {
    TEST_ASSERT_EQUAL(7U, TEMPORAL_CSI_MINIMUM_COVERAGE_NUMERATOR);
    TEST_ASSERT_EQUAL(10U, TEMPORAL_CSI_MINIMUM_COVERAGE_DENOMINATOR);
    TEST_ASSERT_EQUAL(100U, temporal_window_slots(100U, 1000U));
    TEST_ASSERT_EQUAL(141U, temporal_window_slots(94U, 1500U));
    TEST_ASSERT_EQUAL(70U, temporal_minimum_valid_slots(100U));
    TEST_ASSERT_EQUAL(99U, temporal_minimum_valid_slots(141U));
    TEST_ASSERT_EQUAL(5000U, temporal_minimum_sample_spacing_us(100U));

    TemporalCsiSampler sampler(10U, 1000U);
    for (uint32_t slot = 0U; slot < 10U; ++slot) {
        if (slot == 3U || slot == 7U || slot == 8U) continue;
        sampler.admit(slot * 100000U);
    }
    TEST_ASSERT_TRUE(sampler.flush());
    TEST_ASSERT_EQUAL(9U, sampler.current_slot());
    TEST_ASSERT_EQUAL(7U, sampler.occupancy_slots());
    TEST_ASSERT_EQUAL(3U, sampler.missing_slots());
    TEST_ASSERT_EQUAL(7U, sampler.minimum_valid_slots());
    TEST_ASSERT_TRUE(sampler.is_ready());
}

void test_temporal_csi_sampler_rejects_bursts_bad_order_and_stale_packets(void) {
    TemporalCsiSampler sampler(100U, 1000U);
    TEST_ASSERT_FALSE(sampler.admit(1000000U));
    TEST_ASSERT_FALSE(sampler.admit(1000100U));
    TEST_ASSERT_FALSE(sampler.admit(1000500U));
    TEST_ASSERT_TRUE(sampler.admit(1009999U));
    TEST_ASSERT_FALSE(sampler.admit(1010000U));
    TEST_ASSERT_TRUE(sampler.flush());
    TEST_ASSERT_EQUAL(3U, sampler.excess_packets());

    TEST_ASSERT_FALSE(sampler.admit(1010000U));
    TEST_ASSERT_FALSE(sampler.admit(1009999U));
    TEST_ASSERT_FALSE(sampler.admit(1020000U, true, 2020000U, true));
    TEST_ASSERT_EQUAL(1U, sampler.duplicate_packets());
    TEST_ASSERT_EQUAL(1U, sampler.out_of_order_packets());
    TEST_ASSERT_EQUAL(1U, sampler.stale_packets());
}

void test_temporal_csi_sampler_tolerates_alternating_scheduler_jitter(void) {
    TemporalCsiSampler sampler(100U, 1000U);
    TEST_ASSERT_FALSE(sampler.admit(0U));
    for (uint32_t pair = 1U; pair <= 50U; ++pair) {
        TEST_ASSERT_TRUE(sampler.admit(pair * 20000U - 11000U));
        TEST_ASSERT_TRUE(sampler.admit(pair * 20000U));
    }
    TEST_ASSERT_TRUE(sampler.flush());
    TEST_ASSERT_EQUAL(101U, sampler.accepted_packets());
    TEST_ASSERT_EQUAL(0U, sampler.excess_packets());
    TEST_ASSERT_EQUAL(0U, sampler.missing_slots());
    TEST_ASSERT_EQUAL(100U, sampler.occupancy_slots());
    TEST_ASSERT_TRUE(sampler.is_ready());
}

void test_temporal_csi_sampler_handles_wrap_and_window_gap(void) {
    TemporalCsiSampler sampler(100U, 1000U);
    TEST_ASSERT_FALSE(sampler.admit(UINT32_MAX - 4999U));
    TEST_ASSERT_TRUE(sampler.admit(5000U));
    TEST_ASSERT_TRUE(sampler.flush());
    TEST_ASSERT_EQUAL(1U, sampler.current_slot());
    TEST_ASSERT_EQUAL(0U, sampler.gap_resets());

    TEST_ASSERT_FALSE(sampler.admit(1005000U));
    TEST_ASSERT_FALSE(sampler.reset_required());
    TEST_ASSERT_TRUE(sampler.gap_reset_required());
    TEST_ASSERT_TRUE(sampler.flush());
    TEST_ASSERT_TRUE(sampler.reset_required());
    TEST_ASSERT_FALSE(sampler.gap_reset_required());
    TEST_ASSERT_EQUAL(1U, sampler.gap_resets());
    TEST_ASSERT_EQUAL(0U, sampler.current_slot());
    TEST_ASSERT_EQUAL(1U, sampler.occupancy_slots());
}

void test_temporal_csi_sampler_clears_window_without_rephasing(void) {
    TemporalCsiSampler sampler(100U, 1000U);
    TEST_ASSERT_FALSE(sampler.admit(1000000U));
    TEST_ASSERT_TRUE(sampler.admit(1010000U));
    TEST_ASSERT_EQUAL(0U, sampler.current_slot());

    sampler.clear_window_preserving_phase();
    TEST_ASSERT_EQUAL(0U, sampler.occupancy_slots());
    TEST_ASSERT_FALSE(sampler.has_pending_candidate());

    TEST_ASSERT_FALSE(sampler.admit(1020000U));
    TEST_ASSERT_TRUE(sampler.admit(1030000U));
    TEST_ASSERT_EQUAL(2U, sampler.current_slot());
    TEST_ASSERT_EQUAL(1U, sampler.missing_slots_before());
    TEST_ASSERT_EQUAL(1U, sampler.occupancy_slots());

    for (uint32_t slot = 4U; slot <= 72U; ++slot) {
        TEST_ASSERT_TRUE(sampler.admit(1000000U + slot * 10000U));
    }
    TEST_ASSERT_EQUAL(70U, sampler.occupancy_slots());
    TEST_ASSERT_FALSE(sampler.is_ready());

    for (uint32_t slot = 73U; slot <= 101U; ++slot) {
        TEST_ASSERT_TRUE(sampler.admit(1000000U + slot * 10000U));
    }
    TEST_ASSERT_TRUE(sampler.is_ready());
}

void test_temporal_csi_sampler_matches_python_cross_runtime_trace(void) {
    TemporalCsiSampler sampler(20U, 500U);
    const uint32_t timestamps[] = {
        1000000U, 1000100U, 1050000U, 1150000U, 1150000U,
        1149999U, 1300000U, 1800000U, 1800100U, 1850000U,
    };
    const bool expected[] = {
        false, false, true, true, false, false, true, true, false, true,
    };
    for (size_t index = 0U; index < sizeof(timestamps) / sizeof(timestamps[0]); ++index) {
        TEST_ASSERT_EQUAL(expected[index], sampler.admit(timestamps[index]));
    }
    TEST_ASSERT_TRUE(sampler.flush());
    TEST_ASSERT_EQUAL(6U, sampler.accepted_packets());
    TEST_ASSERT_EQUAL(2U, sampler.excess_packets());
    TEST_ASSERT_EQUAL(1U, sampler.duplicate_packets());
    TEST_ASSERT_EQUAL(1U, sampler.out_of_order_packets());
    TEST_ASSERT_EQUAL(3U, sampler.missing_slots());
    TEST_ASSERT_EQUAL(1U, sampler.gap_resets());
    TEST_ASSERT_EQUAL(1U, sampler.current_slot());
    TEST_ASSERT_EQUAL(2U, sampler.occupancy_slots());
    TEST_ASSERT_FALSE(sampler.is_ready());
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
    for (size_t i = 0; i < sizeof(metrics) / sizeof(metrics[0]); ++i) {
        weighted.observe(true, metrics[i], 25U);
        for (uint16_t packet = 0; packet < 25U; ++packet) {
            repeated.observe(true, metrics[i]);
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

// A long quiet prefix must not stop motion-first from accepting: the bootstrap
// keeps only the last two chunks, so 300 quiet packets classify the same way 50
// do.
void test_motion_first_accepts_after_a_long_quiet_prefix(void) {
    StartupThresholdCalibrator calibrator;
    calibrator.begin(500, true);
    for (int i = 0; i < 300; ++i) {
        calibrator.observe(true, 0.05f);
    }
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.12f);
    }
    for (int i = 0; i < 50; ++i) {
        calibrator.observe(true, 0.05f);
    }

    TEST_ASSERT_TRUE(calibrator.is_complete());
    TEST_ASSERT_EQUAL_INT(400, static_cast<int>(calibrator.packet_count()));
    TEST_ASSERT_EQUAL_FLOAT(0.085f, calibrator.threshold_metric());
    TEST_ASSERT_EQUAL_STRING("motion gap midpoint", calibrator.statistic_name());
}

void test_detector_startup_gate_traits(void) {
    LightweightDetector classic;
    HighAccuracyDetector ml;
    TEST_ASSERT_TRUE(classic.startup_gate_enabled());
    TEST_ASSERT_FALSE(ml.startup_gate_enabled());
}

void test_ml_feature_helpers_cover_guard_paths(void) {
    float sample[] = {1.0f, 3.0f, 5.0f, 7.0f};
    float sorted[] = {1.0f, 3.0f, 5.0f, 7.0f};

    TEST_ASSERT_EQUAL_FLOAT(0.0f, median_from_sorted(nullptr, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, median_from_sorted(sorted, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 2, 2.0f, 1.0f, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 4, 4.0f, 0.0f, 1));
    // A scratch that cannot back the sort leaves zcr at zero rather
    // than reading past its end.
    MLStatNeeds needs;
    needs.sorted = true;
    needs.zcr = true;
    MLSeriesStats stats;
    compute_ml_series_stats(sample, 4, &stats, needs, MLSeriesScratch{});
    TEST_ASSERT_EQUAL_FLOAT(0.0f, stats.zcr);

    float sort_scratch[4];
    const MLSeriesScratch scratch{sort_scratch, 4U};
    compute_ml_series_stats(sample, 4, &stats, needs, scratch);
    TEST_ASSERT_EQUAL_FLOAT(1.0f / 3.0f, stats.zcr);

    float duplicate_sample[] = {4.0f, 1.0f, 4.0f, 2.0f,
                                2.0f, 8.0f, 4.0f, 2.0f};
    float duplicate_scratch[8];
    needs.mean = true;
    needs.iqr = true;
    compute_ml_series_stats(
        duplicate_sample, 8, &stats, needs,
        MLSeriesScratch{duplicate_scratch, 8U});
    TEST_ASSERT_EQUAL_FLOAT(2.0f, stats.iqr);
    TEST_ASSERT_EQUAL_FLOAT(5.0f / 7.0f, stats.zcr);
}

void test_lightweight_detector_move_semantics_and_base_accessors(void) {
    auto packet = make_constant_packet(3, 4);

    LightweightDetector source(5, 0.75f);
    source.configure_lowpass(true, 2.0f);
    source.configure_hampel(true, 5, 2.5f);
    source.process_packet(nullptr, packet.size(), DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);
    source.process_packet(packet.data(), packet.size(), DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);

    TEST_ASSERT_NOT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_EQUAL(1, source.get_buffer_count());

    LightweightDetector moved(std::move(source));
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

    LightweightDetector assigned(7, 4.0f);
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

void test_high_accuracy_detector_move_semantics_and_cv_state(void) {
    auto packet = make_constant_packet(3, 4);

    HighAccuracyDetector source(6, 6.0f);
    source.process_packet(packet.data(), packet.size(), DEFAULT_SUBCARRIERS, HT20_SELECTED_BAND_SIZE);

    HighAccuracyDetector moved(std::move(source));
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

    HighAccuracyDetector assigned(10, 7.0f);
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
    RUN_TEST(test_temporal_csi_sampler_matches_fixed_slot_contract);
    RUN_TEST(test_temporal_csi_sampler_rejects_bursts_bad_order_and_stale_packets);
    RUN_TEST(test_temporal_csi_sampler_tolerates_alternating_scheduler_jitter);
    RUN_TEST(test_temporal_csi_sampler_handles_wrap_and_window_gap);
    RUN_TEST(test_temporal_csi_sampler_clears_window_without_rephasing);
    RUN_TEST(test_temporal_csi_sampler_matches_python_cross_runtime_trace);
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
    RUN_TEST(test_motion_first_accepts_after_a_long_quiet_prefix);
    RUN_TEST(test_detector_startup_gate_traits);
    RUN_TEST(test_ml_feature_helpers_cover_guard_paths);
    RUN_TEST(test_required_amplitudes_preserve_selected_and_aggregated_bands);
    RUN_TEST(test_l1_reconfiguration_and_moves_preserve_profile_history);
    RUN_TEST(test_channel_shape_trajectory_is_gain_and_stutter_invariant);
    RUN_TEST(test_trajectory_duplicate_packets_expire_old_motion);
    RUN_TEST(test_shared_packet_frame_matches_direct_trajectory_tracker);
    RUN_TEST(test_lightweight_detector_move_semantics_and_base_accessors);
    RUN_TEST(test_high_accuracy_detector_move_semantics_and_cv_state);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
