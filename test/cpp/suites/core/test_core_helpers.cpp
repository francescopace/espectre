/*
 * ESPectre - Core Helper Unit Tests
 *
 * Targets uncovered helper paths in the reusable core layer.
 */

#include "test_harness.h"

#include <cmath>
#include <utility>
#include <vector>

#include "ml_detector.h"
#include "ml_features.h"
#include "mvs_detector.h"
#include "threshold.h"
#include "utils.h"

using namespace esphome::espectre;

namespace {

std::vector<int8_t> make_constant_packet(int8_t i_value, int8_t q_value) {
    std::vector<int8_t> packet(HT20_CSI_LEN);
    for (uint16_t sc = 0; sc < HT20_NUM_SUBCARRIERS; ++sc) {
        packet[sc * 2] = q_value;
        packet[sc * 2 + 1] = i_value;
    }
    return packet;
}

constexpr uint8_t kBand[] = {12, 14, 16, 18, 20, 24, 28, 36, 40, 44, 48, 52};

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

    TEST_ASSERT_EQUAL_FLOAT(6.0f, apply_cv_normalization(6.0f, 2.0f, false));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, apply_cv_normalization(6.0f, 2.0f, true));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, apply_cv_normalization(6.0f, 0.0f, true));

    float mean_values[] = {2.0f, 4.0f, 6.0f, 8.0f};
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 2.2360679f,
                             calculate_turbulence_from_variance(5.0f, mean_values, 4, false));
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 0.4472136f,
                             calculate_turbulence_from_variance(5.0f, mean_values, 4, true));
}

void test_utils_spatial_turbulence_handles_invalid_inputs(void) {
    float magnitudes[] = {1.0f, 2.0f, 3.0f, 4.0f};
    uint8_t invalid_band[] = {0, 7, 9};

    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(nullptr, invalid_band, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, nullptr, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, invalid_band, 0));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence(magnitudes, invalid_band, 3, 4, true));

    float valid = calculate_spatial_turbulence(magnitudes, invalid_band, 3, 10, false);
    TEST_ASSERT_TRUE(valid >= 0.0f);

    auto packet = make_constant_packet(3, 4);
    uint8_t sparse_band[] = {0, 31, 63, 90};
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(nullptr, packet.size(), sparse_band, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(packet.data(), 1, sparse_band, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(packet.data(), packet.size(), nullptr, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calculate_spatial_turbulence_from_csi(packet.data(), packet.size(), sparse_band, 0));

    float from_csi = calculate_spatial_turbulence_from_csi(packet.data(), packet.size(), sparse_band, 4, false);
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
    TEST_ASSERT_EQUAL_UINT8(DEFAULT_ADAPTIVE_PERCENTILE, get_threshold_percentile(ThresholdMode::AUTO));
    TEST_ASSERT_EQUAL_UINT8(100, get_threshold_percentile(ThresholdMode::MIN));
    TEST_ASSERT_EQUAL_FLOAT(DEFAULT_ADAPTIVE_FACTOR, get_threshold_factor(ThresholdMode::AUTO));
    TEST_ASSERT_EQUAL_FLOAT(1.0f, get_threshold_factor(ThresholdMode::MIN));

    std::vector<float> empty;
    TEST_ASSERT_EQUAL_FLOAT(1.0f, calculate_percentile(empty, 95));

    std::vector<float> values = {1.0f, 2.0f, 3.0f, 9.0f};
    TEST_ASSERT_FLOAT_WITHIN(0.0001f, 2.5f, calculate_percentile(values, 50));
    TEST_ASSERT_EQUAL_FLOAT(9.0f, calculate_percentile(values, 100));

    TEST_ASSERT_TRUE(is_valid_threshold(2.0f, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(NAN, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(0.0f, 0.1f, 3.0f));
    TEST_ASSERT_FALSE(is_valid_threshold(4.0f, 0.1f, 3.0f));

    TEST_ASSERT_EQUAL_FLOAT(0.1f, clamp_threshold(NAN, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.1f, clamp_threshold(-1.0f, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, clamp_threshold(9.0f, 0.1f, 3.0f));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, clamp_threshold(2.0f, 0.1f, 3.0f));

    float adaptive_threshold = 0.0f;
    uint8_t percentile = 0;
    calculate_adaptive_threshold(values, ThresholdMode::AUTO, adaptive_threshold, percentile);
    TEST_ASSERT_EQUAL_UINT8(DEFAULT_ADAPTIVE_PERCENTILE, percentile);
    TEST_ASSERT_TRUE(adaptive_threshold > 0.0f);
    TEST_ASSERT_EQUAL_FLOAT(calculate_percentile(values, 50), calculate_adaptive_threshold(values, 50));
}

void test_ml_feature_helpers_cover_guard_paths(void) {
    float sample[] = {1.0f, 3.0f, 5.0f, 7.0f};
    float sorted[] = {1.0f, 3.0f, 5.0f, 7.0f};

    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_skewness(sample, 2, 2.0f, 1.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_skewness(sample, 4, 4.0f, 0.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, median_from_sorted(nullptr, 0));
    TEST_ASSERT_EQUAL_FLOAT(4.0f, median_from_sorted(sorted, 4));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, interpolate_sorted_percentile(nullptr, 0, 25.0f));
    TEST_ASSERT_EQUAL_FLOAT(1.0f, interpolate_sorted_percentile(sorted, 1, 25.0f));
    TEST_ASSERT_EQUAL_FLOAT(7.0f, interpolate_sorted_percentile(sorted, 4, 100.0f));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_iqr(sample, ML_MAX_SORT_SIZE + 1));
    TEST_ASSERT_EQUAL_FLOAT(3.0f, calc_iqr(sample, 4, sorted));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 2, 2.0f, 1.0f, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_autocorrelation(sample, 4, 4.0f, 0.0f, 1));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_mad(sample, ML_MAX_SORT_SIZE + 1));
    TEST_ASSERT_EQUAL_FLOAT(2.0f, calc_mad(sample, 4, sorted));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_waveform_length(nullptr, 3));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, calc_waveform_length(sample, 1));
    TEST_ASSERT_EQUAL_FLOAT(6.0f, calc_waveform_length(sample, 4));
}

void test_mvs_detector_move_semantics_and_base_accessors(void) {
    auto packet = make_constant_packet(3, 4);

    MVSDetector source(5, 2.5f);
    source.configure_lowpass(true, 2.0f);
    source.configure_hampel(true, 5, 2.5f);
    source.set_cv_normalization(true);
    source.process_packet(nullptr, packet.size(), kBand, HT20_SELECTED_BAND_SIZE);
    source.process_packet(packet.data(), packet.size(), kBand, HT20_SELECTED_BAND_SIZE);

    TEST_ASSERT_TRUE(source.is_cv_normalization_enabled());
    TEST_ASSERT_NOT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_EQUAL(1, source.get_buffer_count());

    MVSDetector moved(std::move(source));
    TEST_ASSERT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_TRUE(moved.is_lowpass_enabled());
    TEST_ASSERT_TRUE(moved.is_hampel_enabled());
    TEST_ASSERT_TRUE(moved.is_cv_normalization_enabled());
    TEST_ASSERT_EQUAL(1, moved.get_total_packets());
    TEST_ASSERT_EQUAL(1, moved.get_buffer_count());
    TEST_ASSERT_EQUAL_FLOAT(2.5f, moved.get_threshold());

    MVSDetector assigned(7, 4.0f);
    assigned = std::move(moved);
    TEST_ASSERT_NULL(moved.get_turbulence_buffer());
    TEST_ASSERT_TRUE(assigned.is_lowpass_enabled());
    TEST_ASSERT_TRUE(assigned.is_hampel_enabled());
    TEST_ASSERT_TRUE(assigned.is_cv_normalization_enabled());
    TEST_ASSERT_EQUAL(1, assigned.get_total_packets());
    TEST_ASSERT_EQUAL(1, assigned.get_buffer_count());
}

void test_ml_detector_move_semantics_and_cv_override(void) {
    auto packet = make_constant_packet(3, 4);

    MLDetector source(6, 6.0f);
    source.process_packet(packet.data(), packet.size(), kBand, HT20_SELECTED_BAND_SIZE);
    source.set_cv_normalization(true);  // No-op in ML detector.

    TEST_ASSERT_FALSE(source.is_cv_normalization_enabled());

    MLDetector moved(std::move(source));
    TEST_ASSERT_NULL(source.get_turbulence_buffer());
    TEST_ASSERT_EQUAL_FLOAT(6.0f, moved.get_threshold());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, moved.get_motion_metric());

    MLDetector assigned(10, 7.0f);
    assigned = std::move(moved);
    TEST_ASSERT_NULL(moved.get_turbulence_buffer());
    TEST_ASSERT_EQUAL_FLOAT(6.0f, assigned.get_threshold());
    TEST_ASSERT_EQUAL_FLOAT(0.0f, assigned.get_motion_metric());
}

int process(void) {
    UNITY_BEGIN();
    RUN_TEST(test_utils_statistical_helpers_cover_edge_cases);
    RUN_TEST(test_utils_spatial_turbulence_handles_invalid_inputs);
    RUN_TEST(test_threshold_helpers_cover_modes_and_ranges);
    RUN_TEST(test_ml_feature_helpers_cover_guard_paths);
    RUN_TEST(test_mvs_detector_move_semantics_and_base_accessors);
    RUN_TEST(test_ml_detector_move_semantics_and_cv_override);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
