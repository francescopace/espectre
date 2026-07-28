/*
 * ESPectre - CSI Format
 *
 * HT20 CSI layout constants, subcarrier band selection, and helpers that
 * extract amplitudes and turbulence directly from raw CSI payloads.
 * Keep aligned with src/python/micro_espectre/config.py and utils.py.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include "utils.h"

namespace espectre {

// =============================================================================
// HT20 Constants (64 subcarriers - do not change)
// =============================================================================
constexpr uint16_t HT20_NUM_SUBCARRIERS = 64;      // HT20: 64 subcarriers
constexpr uint16_t HT20_CSI_LEN = 128;             // 64 SC × 2 bytes (I/Q pairs)
constexpr uint16_t HT20_CSI_LEN_DOUBLE = 256;      // 2 x HT20_CSI_LEN (double-LTF/STBC-like)
constexpr uint16_t HT20_CSI_LEN_SHORT = 114;       // 57 SC × 2 bytes (short HT estimate)
constexpr uint16_t HT20_CSI_LEN_SHORT_DOUBLE = 228; // 2 x HT20_CSI_LEN_SHORT
constexpr uint8_t HT20_CSI_LEN_SHORT_LEFT_PAD = 8; // 4 SC × 2 bytes left guard padding
constexpr uint8_t HT20_GUARD_BAND_LOW = 4;         // First valid subcarrier (-28)
constexpr uint8_t HT20_GUARD_BAND_HIGH = 60;       // Last valid subcarrier (+28)
constexpr uint8_t HT20_DC_SUBCARRIER = 32;         // DC null subcarrier
constexpr uint8_t HT20_SELECTED_BAND_SIZE = 12;    // Selected subcarriers for motion detection
// Subcarriers +/-4, +/-9, +/-14, +/-19, +/-24, +/-28. Spans the full usable range
// because the motion perturbation stays coherent over ~10 subcarriers (3.1 MHz)
// while quiet noise is nearly per-tone independent, so span is what buys
// independent looks. Stops short of |sc| <= 3, where relative jitter rises ~10%.
// See docs/adr/2026-07-25-select-the-classic-band-from-channel-coherence.md.
constexpr uint8_t DEFAULT_SUBCARRIERS[HT20_SELECTED_BAND_SIZE] = {
    4, 8, 13, 18, 23, 28, 36, 41, 46, 51, 56, 60
};
using SelectedSubcarriers = std::array<uint8_t, HT20_SELECTED_BAND_SIZE>;

constexpr SelectedSubcarriers make_default_subcarriers() {
    SelectedSubcarriers subcarriers{};
    for (uint8_t i = 0; i < HT20_SELECTED_BAND_SIZE; ++i) {
        subcarriers[i] = DEFAULT_SUBCARRIERS[i];
    }
    return subcarriers;
}

// =============================================================================
// HT20 Bin Layout
// =============================================================================
// Wi-Fi 6 parts deliver HT20 CSI centered on DC (bin = subcarrier + 32), while
// classic-MAC parts deliver Espressif's native "0~31, -32~-1" order with DC in
// bin 0. `DEFAULT_SUBCARRIERS` and `HT20_DC_SUBCARRIER` assume the centered
// convention, so classic payloads must be rotated before the band means the same
// physical subcarriers on every chip.
//
// The two layouts are told apart by their guard nulls, which the radio reports as
// exactly zero. Bins 0 and 32 are null under both conventions and carry no
// information; these are the bins that are null under exactly one of them.
constexpr uint8_t HT20_CLASSIC_ONLY_NULL_BINS[] = {29, 30, 31, 33, 34, 35};
constexpr uint8_t HT20_CENTERED_ONLY_NULL_BINS[] = {1, 2, 3, 61, 62, 63};

enum class Ht20BinLayout : uint8_t {
    UNKNOWN = 0,
    CENTERED,  // bin = subcarrier + 32, DC at bin 32
    CLASSIC,   // bin = subcarrier mod 64, DC at bin 0
};

inline uint8_t ht20_bins_with_energy(const int8_t* csi_data, const uint8_t* bins, uint8_t count) {
    uint8_t populated = 0;
    for (uint8_t i = 0; i < count; ++i) {
        const uint16_t byte_index = static_cast<uint16_t>(bins[i]) * 2U;
        if (csi_data[byte_index] != 0 || csi_data[byte_index + 1] != 0) {
            populated++;
        }
    }
    return populated;
}

/**
 * Identify which HT20 bin ordering a 64-subcarrier payload uses.
 *
 * Requires positive evidence in both directions: one guard set must be entirely
 * null and the other entirely populated. Absence of energy alone is not enough,
 * because a sparse or degenerate payload is null under both conventions.
 *
 * @param csi_data Raw CSI payload (interleaved I/Q pairs)
 * @param csi_len Payload length in bytes (must be HT20_CSI_LEN)
 * @return The detected layout, or UNKNOWN when the evidence is inconclusive
 */
inline Ht20BinLayout detect_ht20_bin_layout(const int8_t* csi_data, size_t csi_len) {
    if (csi_data == nullptr || csi_len != HT20_CSI_LEN) {
        return Ht20BinLayout::UNKNOWN;
    }

    constexpr uint8_t kNullBinCount =
        static_cast<uint8_t>(sizeof(HT20_CLASSIC_ONLY_NULL_BINS));
    const uint8_t classic_energy =
        ht20_bins_with_energy(csi_data, HT20_CLASSIC_ONLY_NULL_BINS, kNullBinCount);
    const uint8_t centered_energy =
        ht20_bins_with_energy(csi_data, HT20_CENTERED_ONLY_NULL_BINS, kNullBinCount);

    if (classic_energy == 0U && centered_energy == kNullBinCount) {
        return Ht20BinLayout::CLASSIC;
    }
    if (centered_energy == 0U && classic_energy == kNullBinCount) {
        return Ht20BinLayout::CENTERED;
    }
    return Ht20BinLayout::UNKNOWN;
}

/**
 * Rotate a classic-order HT20 payload into the centered convention.
 *
 * Rotating by half the FFT size is its own inverse, so one swap of the payload
 * halves maps `0~31, -32~-1` onto `-32~+31`.
 *
 * @param csi_data Source payload of HT20_CSI_LEN bytes
 * @param out Destination buffer of HT20_CSI_LEN bytes (must not alias the source)
 */
inline void rotate_ht20_classic_to_centered(const int8_t* csi_data, int8_t* out) {
    if (csi_data == nullptr || out == nullptr) {
        return;
    }
    constexpr uint16_t kHalf = HT20_CSI_LEN / 2U;
    std::memcpy(out, csi_data + kHalf, kHalf);
    std::memcpy(out + kHalf, csi_data, kHalf);
}

/**
 * Calculate spatial turbulence from pre-calculated magnitudes
 *
 * Spatial turbulence is the standard deviation of magnitudes across
 * selected subcarriers. It measures the spatial variability of the
 * Wi-Fi channel - higher values indicate motion/disturbance.
 *
 * @param magnitudes Array of magnitude values (one per subcarrier)
 * @param subcarriers Array of selected subcarrier indices
 * @param num_subcarriers Number of selected subcarriers (max 12)
 * @param max_subcarrier Maximum valid subcarrier index (default: 64 for HT20)
 * @return Turbulence value
 */
inline float calculate_spatial_turbulence(const float* magnitudes,
                                          const uint8_t* subcarriers,
                                          uint8_t num_subcarriers,
                                          uint16_t max_subcarrier = 64) {
    if (num_subcarriers == 0 || !magnitudes || !subcarriers) {
        return 0.0f;
    }

    // Collect valid magnitudes (max 12 subcarriers for band selection)
    float valid_mags[12];
    uint8_t valid_count = 0;

    for (uint8_t i = 0; i < num_subcarriers && valid_count < 12; i++) {
        if (subcarriers[i] < max_subcarrier) {
            valid_mags[valid_count++] = magnitudes[subcarriers[i]];
        }
    }

    if (valid_count == 0) {
        return 0.0f;
    }

    float variance = calculate_variance_two_pass(valid_mags, valid_count);
    return calculate_turbulence_from_variance(variance, valid_mags, valid_count);
}

/**
 * Extract subcarrier amplitudes from raw CSI data (I/Q pairs)
 *
 * Mirrors the Python `SegmentationContext._fill_amplitude_buffer` helper.
 *
 * @param csi_data Raw CSI data (interleaved I/Q pairs, Espressif format)
 * @param csi_len Length of CSI data in bytes
 * @param subcarriers Array of selected subcarrier indices
 * @param num_subcarriers Number of selected subcarriers
 * @param out Output amplitude buffer
 * @param out_capacity Capacity of the output buffer
 * @return Number of amplitudes written
 */
inline uint8_t extract_subcarrier_amplitudes(const int8_t* csi_data,
                                             size_t csi_len,
                                             const uint8_t* subcarriers,
                                             uint8_t num_subcarriers,
                                             float* out,
                                             uint8_t out_capacity) {
    if (!csi_data || csi_len < 2 || num_subcarriers == 0 || !subcarriers || !out) {
        return 0;
    }

    int total_subcarriers = static_cast<int>(csi_len / 2);
    uint8_t valid_count = 0;

    for (int i = 0; i < num_subcarriers && valid_count < out_capacity; i++) {
        int sc_idx = subcarriers[i];
        if (sc_idx >= total_subcarriers) {
            continue;
        }

        // Espressif CSI format: [Imaginary, Real, ...] per subcarrier
        out[valid_count++] = calculate_magnitude(csi_data[sc_idx * 2 + 1],
                                                 csi_data[sc_idx * 2]);
    }
    return valid_count;
}

inline float calculate_spatial_turbulence_from_amplitudes(const float* amplitudes,
                                                          uint8_t count) {
    if (amplitudes == nullptr || count == 0) {
        return 0.0f;
    }
    const float variance = calculate_variance_two_pass(amplitudes, count);
    return calculate_turbulence_from_variance(variance, amplitudes, count);
}

/**
 * Calculate spatial turbulence directly from raw CSI data (I/Q pairs)
 *
 * This is a convenience wrapper that calculates magnitudes internally
 * before computing spatial turbulence.
 *
 * HT20 only: 64 subcarriers, 128 bytes CSI data.
 *
 * @param csi_data Raw CSI data (interleaved I/Q pairs)
 * @param csi_len Length of CSI data in bytes (expected: 128 for HT20)
 * @param subcarriers Array of selected subcarrier indices
 * @param num_subcarriers Number of selected subcarriers (max 12)
 * @return Turbulence value
 */
inline float calculate_spatial_turbulence_from_csi(const int8_t* csi_data,
                                                   size_t csi_len,
                                                   const uint8_t* subcarriers,
                                                   uint8_t num_subcarriers) {
    float amplitudes[HT20_SELECTED_BAND_SIZE];
    uint8_t valid_count = extract_subcarrier_amplitudes(
        csi_data, csi_len, subcarriers, num_subcarriers,
        amplitudes, HT20_SELECTED_BAND_SIZE);
    return calculate_spatial_turbulence_from_amplitudes(amplitudes, valid_count);
}

}  // namespace espectre
