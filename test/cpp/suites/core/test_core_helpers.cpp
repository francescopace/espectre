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
#include "ml_feature_trackers.h"
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

// The definition, written straight from the bin table the way the original
// full 56x56 scan walked it. Frequency coherence now walks the two contiguous
// live-band halves instead, so this keeps the pair set, the pair order, and the
// normalization pinned to the formula rather than to the current loop shape.
float reference_frequency_coherence(const std::complex<float>* profile, uint8_t offset) {
    std::complex<float> numerator(0.0f, 0.0f);
    float left_norm = 0.0f;
    float right_norm = 0.0f;
    for (uint8_t left = 0; left < HT20_LIVE_BAND_SIZE; left++) {
        for (uint8_t right = 0; right < HT20_LIVE_BAND_SIZE; right++) {
            const int bin_delta = static_cast<int>(HT20_LIVE_BINS[right]) -
                                  static_cast<int>(HT20_LIVE_BINS[left]);
            if (bin_delta != static_cast<int>(offset)) {
                continue;
            }
            if (HT20_LIVE_BINS[left] < HT20_DC_SUBCARRIER &&
                HT20_DC_SUBCARRIER < HT20_LIVE_BINS[right]) {
                continue;
            }
            numerator += std::conj(profile[left]) * profile[right];
            left_norm += std::norm(profile[left]);
            right_norm += std::norm(profile[right]);
        }
    }
    const float denominator = std::sqrt(left_norm) * std::sqrt(right_norm);
    if (denominator <= 0.0f) {
        return 0.0f;
    }
    return std::abs(numerator) / denominator;
}

// One contiguous subband, written from scratch off the two profiles and in
// double precision, so it stands for the true value rather than for another
// single-precision approximation. Comparing the runtime only against a float
// reference would be circular twice over: same span arithmetic, and same
// rounding behaviour.
double reference_subband_coherence(const std::complex<float>* current,
                                   const std::complex<float>* reference,
                                   uint8_t start, uint8_t count, uint8_t start_bin) {
    std::vector<std::complex<double>> cross;
    double total = 0.0;
    for (uint8_t i = 0; i < count; i++) {
        cross.push_back(std::complex<double>(current[start + i]) *
                        std::conj(std::complex<double>(reference[start + i])));
        total += std::abs(cross[i]);
    }
    if (total <= 0.0) {
        return 0.0;
    }
    std::complex<double> ramp_sum(0.0, 0.0);
    for (uint8_t i = 1; i < count; i++) {
        ramp_sum += cross[i] * std::conj(cross[i - 1U]);
    }
    const double ramp = std::atan2(ramp_sum.imag(), ramp_sum.real());
    std::complex<double> aligned(0.0, 0.0);
    for (uint8_t i = 0; i < count; i++) {
        const double angle = -ramp * static_cast<double>(start_bin + i);
        aligned += cross[i] * std::complex<double>(std::cos(angle), std::sin(angle));
    }
    return std::abs(aligned) / total;
}


// The full live band, written from scratch and in double precision. The DC gap
// makes one index pair non-adjacent in frequency, and it must stay out of the
// ramp estimate.
double reference_full_band_coherence(const std::complex<float>* current,
                                     const std::complex<float>* reference) {
    std::vector<std::complex<double>> cross;
    double total = 0.0;
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        cross.push_back(std::complex<double>(current[i]) *
                        std::conj(std::complex<double>(reference[i])));
        total += std::abs(cross[i]);
    }
    if (total <= 0.0) {
        return 0.0;
    }
    std::complex<double> ramp_sum(0.0, 0.0);
    for (uint8_t i = 1; i < HT20_LIVE_BAND_SIZE; i++) {
        if (HT20_LIVE_BINS[i] - HT20_LIVE_BINS[i - 1U] != 1U) {
            continue;
        }
        ramp_sum += cross[i] * std::conj(cross[i - 1U]);
    }
    const double ramp = std::atan2(ramp_sum.imag(), ramp_sum.real());
    std::complex<double> aligned(0.0, 0.0);
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        const double angle = -ramp * static_cast<double>(HT20_LIVE_BINS[i]);
        aligned += cross[i] * std::complex<double>(std::cos(angle), std::sin(angle));
    }
    return std::abs(aligned) / total;
}


uint16_t reference_pair_count(uint8_t offset) {
    uint16_t pairs = 0;
    for (uint8_t left = 0; left < HT20_LIVE_BAND_SIZE; left++) {
        for (uint8_t right = 0; right < HT20_LIVE_BAND_SIZE; right++) {
            const int bin_delta = static_cast<int>(HT20_LIVE_BINS[right]) -
                                  static_cast<int>(HT20_LIVE_BINS[left]);
            if (bin_delta != static_cast<int>(offset)) {
                continue;
            }
            if (HT20_LIVE_BINS[left] < HT20_DC_SUBCARRIER &&
                HT20_DC_SUBCARRIER < HT20_LIVE_BINS[right]) {
                continue;
            }
            pairs++;
        }
    }
    return pairs;
}

// A packet whose amplitude varies across the band, and whose shape depends on
// `shape`. Constant packets are useless here: the amplitude profile is L2
// normalized, so every constant packet collapses onto the same profile no
// matter its magnitude, and a ring addressing bug would go unnoticed.
std::vector<int8_t> make_shaped_packet(uint16_t shape) {
    std::vector<int8_t> packet(HT20_CSI_LEN);
    uint32_t state = 0x9E3779B9U ^ (static_cast<uint32_t>(shape) * 2654435761U);
    for (uint16_t sc = 0; sc < HT20_NUM_SUBCARRIERS; ++sc) {
        state = state * 1664525U + 1013904223U;
        const int8_t q = static_cast<int8_t>(static_cast<int>((state >> 16U) & 0x3FU) + 8);
        state = state * 1664525U + 1013904223U;
        const int8_t i = static_cast<int8_t>(static_cast<int>((state >> 16U) & 0x3FU) + 8);
        packet[sc * 2] = q;
        packet[sc * 2 + 1] = i;
    }
    return packet;
}

// Deterministic 32-bit LCG: the profiles must be reproducible across runs and
// platforms, which <random> does not guarantee for its distributions.
struct SeededProfiles {
    explicit SeededProfiles(uint32_t seed) : state(seed) {}

    float next_component() {
        state = state * 1664525U + 1013904223U;
        return static_cast<float>(static_cast<int>((state >> 16U) & 0xFFU) - 128);
    }

    void fill(std::complex<float>* profile) {
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            const float real = next_component();
            const float imag = next_component();
            profile[i] = std::complex<float>(real, imag);
        }
    }

    float next_unit() { return (next_component() + 128.0f) / 256.0f; }

    // A coherent pair: one shared channel shape, a per-packet delay ramp, and a
    // small perturbation. This is the regime real CSI sits in, and the one the
    // detector is tuned on; independent random profiles are not.
    void fill_coherent_pair(std::complex<float>* current, std::complex<float>* reference) {
        const double delay = (next_unit() - 0.5f) * 0.08;
        for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
            const double amplitude = 20.0 + 100.0 * next_unit();
            const double phase = 6.283185307 * next_unit();
            const std::complex<double> channel = std::polar(amplitude, phase);
            const std::complex<double> rotated =
                channel * std::polar(1.0, delay * static_cast<double>(HT20_LIVE_BINS[i])) *
                (1.0 + 0.01 * (next_unit() - 0.5f));
            current[i] = std::complex<float>(static_cast<float>(channel.real()),
                                             static_cast<float>(channel.imag()));
            reference[i] = std::complex<float>(static_cast<float>(rotated.real()),
                                               static_cast<float>(rotated.imag()));
        }
    }

    uint32_t state;
};

// Both entry points must land on the double-precision reference within the
// caller's single-precision bound, and the shared-buffer path must land exactly
// where the standalone wrappers do: same products, same order, same
// normalization.
void check_coherence_against_reference(const std::complex<float>* current,
                                       const std::complex<float>* reference,
                                       std::complex<float>* cross, float* magnitude,
                                       double bound) {
    fill_coherence_cross(current, reference, cross, magnitude);
    const double full = delay_compensated_coherence_from_cross(cross, magnitude);
    TEST_ASSERT_TRUE(full == delay_compensated_coherence(current, reference));
    TEST_ASSERT_DOUBLE_WITHIN(bound, reference_full_band_coherence(current, reference), full);

    float shared_bands[HT20_COHERENCE_SUBBAND_COUNT]{};
    float direct_bands[HT20_COHERENCE_SUBBAND_COUNT]{};
    subband_coherences_from_cross(cross, magnitude, shared_bands);
    subband_coherences(current, reference, direct_bands);
    for (uint8_t b = 0; b < HT20_COHERENCE_SUBBAND_COUNT; b++) {
        const uint8_t start = static_cast<uint8_t>(b * HT20_COHERENCE_SUBBAND_SIZE);
        TEST_ASSERT_TRUE(shared_bands[b] == direct_bands[b]);
        TEST_ASSERT_DOUBLE_WITHIN(
            bound,
            reference_subband_coherence(current, reference, start,
                                        HT20_COHERENCE_SUBBAND_SIZE, HT20_LIVE_BINS[start]),
            shared_bands[b]);
    }
}

}  // namespace

void test_frequency_coherence_matches_the_reference_formula(void) {
    const uint8_t offsets[] = {2U, 4U, 12U};
    const uint16_t expected_pairs[] = {52U, 48U, 32U};

    // The half walk must reproduce exactly the pairs the full scan found.
    for (uint8_t i = 0; i < 3U; i++) {
        TEST_ASSERT_EQUAL(expected_pairs[i], reference_pair_count(offsets[i]));
    }

    // A null profile is guarded rather than dereferenced.
    for (uint8_t i = 0; i < 3U; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, frequency_coherence(nullptr, offsets[i]));
    }

    // An all-zero profile hits the denominator guard.
    std::complex<float> zeros[HT20_LIVE_BAND_SIZE]{};
    for (uint8_t i = 0; i < 3U; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, frequency_coherence(zeros, offsets[i]));
        TEST_ASSERT_EQUAL_FLOAT(0.0f, reference_frequency_coherence(zeros, offsets[i]));
    }

    // Deterministic profiles: flat, ramped, sign-flipping, and one silent half.
    std::complex<float> flat[HT20_LIVE_BAND_SIZE];
    std::complex<float> ramp[HT20_LIVE_BAND_SIZE];
    std::complex<float> alternating[HT20_LIVE_BAND_SIZE];
    std::complex<float> half_silent[HT20_LIVE_BAND_SIZE];
    for (uint8_t i = 0; i < HT20_LIVE_BAND_SIZE; i++) {
        flat[i] = std::complex<float>(7.0f, -3.0f);
        ramp[i] = std::complex<float>(static_cast<float>(i) - 28.0f, 0.5f * i);
        alternating[i] = (i % 2U == 0U) ? std::complex<float>(9.0f, 2.0f)
                                        : std::complex<float>(-9.0f, -2.0f);
        half_silent[i] = (i < HT20_LIVE_HALF_SIZE) ? std::complex<float>(0.0f, 0.0f)
                                                   : std::complex<float>(5.0f, 4.0f);
    }
    const std::complex<float>* deterministic[] = {flat, ramp, alternating, half_silent};
    for (const std::complex<float>* profile : deterministic) {
        for (uint8_t i = 0; i < 3U; i++) {
            TEST_ASSERT_FLOAT_WITHIN(1e-6f,
                                     reference_frequency_coherence(profile, offsets[i]),
                                     frequency_coherence(profile, offsets[i]));
        }
    }

    // Seeded pseudo-random profiles over the int8 range CSI actually delivers.
    SeededProfiles generator(20260805U);
    std::complex<float> random_profile[HT20_LIVE_BAND_SIZE];
    for (uint16_t trial = 0; trial < 64U; trial++) {
        generator.fill(random_profile);
        for (uint8_t i = 0; i < 3U; i++) {
            TEST_ASSERT_FLOAT_WITHIN(1e-6f,
                                     reference_frequency_coherence(random_profile, offsets[i]),
                                     frequency_coherence(random_profile, offsets[i]));
        }
    }

    // Offsets with no pair inside a half yield no coherence.
    const uint8_t unsupported[] = {28U, 29U, 55U, 200U};
    for (uint8_t offset : unsupported) {
        TEST_ASSERT_EQUAL(0U, reference_pair_count(offset));
        TEST_ASSERT_EQUAL_FLOAT(0.0f, frequency_coherence(random_profile, offset));
    }
}

void test_frequency_coherences_matches_single_offset_calls(void) {
    SeededProfiles generator(991U);
    std::complex<float> profile[HT20_LIVE_BAND_SIZE];
    float combined[FREQUENCY_COHERENCE_COUNT]{};

    TEST_ASSERT_EQUAL(3U, FREQUENCY_COHERENCE_COUNT);
    TEST_ASSERT_EQUAL_UINT8(2U, FREQUENCY_COHERENCE_OFFSETS[0]);
    TEST_ASSERT_EQUAL_UINT8(4U, FREQUENCY_COHERENCE_OFFSETS[1]);
    TEST_ASSERT_EQUAL_UINT8(12U, FREQUENCY_COHERENCE_OFFSETS[2]);

    for (uint16_t trial = 0; trial < 32U; trial++) {
        generator.fill(profile);
        frequency_coherences(profile, combined);
        for (uint8_t i = 0; i < FREQUENCY_COHERENCE_COUNT; i++) {
            // Same helpers on the same inputs, so this is exact rather than near.
            TEST_ASSERT_TRUE(combined[i] ==
                             frequency_coherence(profile, FREQUENCY_COHERENCE_OFFSETS[i]));
        }
    }

    // A null profile zeroes the whole output instead of leaving it stale.
    for (uint8_t i = 0; i < FREQUENCY_COHERENCE_COUNT; i++) {
        combined[i] = 0.5f;
    }
    frequency_coherences(nullptr, combined);
    for (uint8_t i = 0; i < FREQUENCY_COHERENCE_COUNT; i++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, combined[i]);
    }
}

void test_coherence_shares_cross_products_across_band_and_subbands(void) {
    // Sharing one cross-product array is only valid because the four subbands
    // tile the live band exactly, with no gap and no overlap.
    const uint8_t starts[] = {0U, 14U, 28U, 42U};
    for (uint8_t b = 0; b < HT20_COHERENCE_SUBBAND_COUNT; b++) {
        TEST_ASSERT_EQUAL_UINT8(b * HT20_COHERENCE_SUBBAND_SIZE, starts[b]);
    }
    TEST_ASSERT_EQUAL(HT20_LIVE_BAND_SIZE,
                      HT20_COHERENCE_SUBBAND_COUNT * HT20_COHERENCE_SUBBAND_SIZE);

    SeededProfiles generator(5150U);
    std::complex<float> current[HT20_LIVE_BAND_SIZE];
    std::complex<float> reference[HT20_LIVE_BAND_SIZE];
    std::complex<float> cross[HT20_LIVE_BAND_SIZE]{};
    float magnitude[HT20_LIVE_BAND_SIZE]{};

    // Two regimes, because they bound the error for different reasons.
    //
    // Uniform random profiles are adversarial: they leave the aligned sum
    // almost fully cancelled, and the relative error of a cancelled sum in
    // single precision is large no matter how it is evaluated. Measured over
    // 20000 such trials, Horner and the per-bin trigonometric form it replaced
    // both sit at roughly 1.8e-5, indistinguishable.
    //
    // Coherent profiles are the production regime, and there the two forms
    // separate: roughly 1.1e-6 for Horner against 3.6e-7 for the per-bin form.
    // Both bounds are far below what the data can carry, since the CSI payload
    // is int8 and its own resolution is around 4e-3.
    const double adversarial_bound = 5e-5;
    const double coherent_bound = 1e-5;

    for (uint16_t trial = 0; trial < 512U; trial++) {
        generator.fill(current);
        generator.fill(reference);
        check_coherence_against_reference(current, reference, cross, magnitude,
                                          adversarial_bound);
    }
    for (uint16_t trial = 0; trial < 512U; trial++) {
        generator.fill_coherent_pair(current, reference);
        check_coherence_against_reference(current, reference, cross, magnitude,
                                          coherent_bound);
    }

    // Null inputs stay guarded on every entry point.
    float bands[HT20_COHERENCE_SUBBAND_COUNT]{};
    for (uint8_t b = 0; b < HT20_COHERENCE_SUBBAND_COUNT; b++) {
        bands[b] = 0.5f;
    }
    TEST_ASSERT_EQUAL_FLOAT(0.0f, delay_compensated_coherence(nullptr, reference));
    TEST_ASSERT_EQUAL_FLOAT(0.0f, delay_compensated_coherence(current, nullptr));
    subband_coherences(nullptr, reference, bands);
    for (uint8_t b = 0; b < HT20_COHERENCE_SUBBAND_COUNT; b++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, bands[b]);
    }

    // An all-zero reference drives the denominator guard rather than a divide.
    std::complex<float> zeros[HT20_LIVE_BAND_SIZE]{};
    fill_coherence_cross(zeros, zeros, cross, magnitude);
    TEST_ASSERT_EQUAL_FLOAT(0.0f, delay_compensated_coherence_from_cross(cross, magnitude));
    subband_coherences_from_cross(cross, magnitude, bands);
    for (uint8_t b = 0; b < HT20_COHERENCE_SUBBAND_COUNT; b++) {
        TEST_ASSERT_EQUAL_FLOAT(0.0f, bands[b]);
    }
}

void test_coherence_tracker_refills_the_cross_buffer_per_reference(void) {
    // The tracker fills one shared cross buffer for the lagged reference and
    // then again for the adjacent one. Reusing the first fill would leave the
    // adjacent coherence reading the lagged products, so this drives both and
    // compares against values computed independently per reference.
    const uint16_t lag = 3U;
    const uint16_t packets = 9U;
    ChannelCoherenceTracker tracker;
    tracker.configure(64U, lag);

    std::vector<std::array<std::complex<float>, HT20_LIVE_BAND_SIZE>> profiles;
    for (uint16_t p = 0; p < packets; p++) {
        const auto packet = make_shaped_packet(static_cast<uint16_t>(p * 7U + 1U));
        std::array<std::complex<float>, HT20_LIVE_BAND_SIZE> profile{};
        extract_ht20_live_complex_profile(packet.data(), packet.size(), profile.data());
        profiles.push_back(profile);
        tracker.process_packet(packet.data(), packet.size());
    }

    double lag_sum = 0.0;
    uint16_t lag_count = 0;
    for (uint16_t i = lag; i < packets; i++) {
        lag_sum += delay_compensated_coherence(profiles[i].data(),
                                               profiles[i - lag].data());
        lag_count++;
    }
    double adjacent_sum = 0.0;
    uint16_t adjacent_count = 0;
    for (uint16_t i = 1U; i < packets; i++) {
        adjacent_sum += delay_compensated_coherence(profiles[i].data(),
                                                    profiles[i - 1U].data());
        adjacent_count++;
    }

    TEST_ASSERT_EQUAL(lag_count, tracker.count());
    const float expected_gap = static_cast<float>(adjacent_sum / adjacent_count -
                                                  lag_sum / lag_count);
    TEST_ASSERT_FLOAT_WITHIN(1e-6f, expected_gap, tracker.coherence_gap());
    // Non-vacuity: the two references really do disagree here, so a stale
    // buffer could not pass by accident.
    TEST_ASSERT_TRUE(std::fabs(expected_gap) > 1e-4f);
}

void test_lag_ring_holds_exactly_the_configured_lag(void) {
    // The lag rings are sized to the configured lag rather than to
    // L1_DELTA_LAG_MAX, so the flattened indexing has to keep addressing the
    // right slot. Feeding a sequence whose period equals the lag makes the
    // lagged reference identical to the current packet, which pins the
    // addressing: the lagged distance collapses to zero only if the ring
    // really wrapped onto the matching packet.
    for (uint16_t lag = 1U; lag <= 4U; lag++) {
        ChannelShapeTracker tracker;
        tracker.configure(64U, lag);

        std::vector<std::vector<int8_t>> period;
        for (uint16_t p = 0; p < lag; p++) {
            period.push_back(make_shaped_packet(static_cast<uint16_t>(p + 1U)));
        }
        for (uint16_t step = 0; step < lag * 6U; step++) {
            const std::vector<int8_t>& packet = period[step % lag];
            tracker.process_packet(packet.data(), packet.size());
        }

        // Every lagged comparison saw an identical profile, so the motion
        // energy stays empty and the participation ratio reports nothing.
        TEST_ASSERT_EQUAL(lag * 5U, tracker.count());
        TEST_ASSERT_EQUAL_FLOAT(0.0f, tracker.shape_spread());

        // Non-vacuity: break the period and the very same tracker reports
        // motion energy, so the zero above is a real match and not a tracker
        // that simply never compared anything.
        ChannelShapeTracker moving;
        moving.configure(64U, lag);
        for (uint16_t step = 0; step < lag * 6U; step++) {
            const auto packet = make_shaped_packet(static_cast<uint16_t>(step + 1U));
            moving.process_packet(packet.data(), packet.size());
        }
        TEST_ASSERT_EQUAL(lag * 5U, moving.count());
        TEST_ASSERT_TRUE(moving.shape_spread() > 0.0f);
    }

    // A tracker configured with no capacity is never fed and holds no ring.
    ChannelShapeTracker unused;
    unused.configure(0U, 8U);
    auto packet = make_constant_packet(4, 6);
    unused.process_packet(packet.data(), packet.size());
    TEST_ASSERT_EQUAL(0, unused.count());
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
    ClassicDetector classic;
    MLDetector ml;
    TEST_ASSERT_TRUE(classic.startup_gate_enabled());
    TEST_ASSERT_FALSE(ml.startup_gate_enabled());
}

void test_ml_feature_helpers_cover_guard_paths(void) {
    float sample[] = {1.0f, 3.0f, 5.0f, 7.0f};
    float sorted[] = {1.0f, 3.0f, 5.0f, 7.0f};
    float abs_devs[4];

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
    RUN_TEST(test_motion_first_accepts_after_a_long_quiet_prefix);
    RUN_TEST(test_detector_startup_gate_traits);
    RUN_TEST(test_ml_feature_helpers_cover_guard_paths);
    RUN_TEST(test_frequency_coherence_matches_the_reference_formula);
    RUN_TEST(test_frequency_coherences_matches_single_offset_calls);
    RUN_TEST(test_coherence_shares_cross_products_across_band_and_subbands);
    RUN_TEST(test_coherence_tracker_refills_the_cross_buffer_per_reference);
    RUN_TEST(test_lag_ring_holds_exactly_the_configured_lag);
    RUN_TEST(test_classic_detector_move_semantics_and_base_accessors);
    RUN_TEST(test_ml_detector_move_semantics_and_cv_state);
    return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) { return process(); }
#endif
