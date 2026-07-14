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
#include <cstdint>

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
constexpr uint8_t HT20_GUARD_BAND_LOW = 11;        // First valid subcarrier
constexpr uint8_t HT20_GUARD_BAND_HIGH = 52;       // Last valid subcarrier
constexpr uint8_t HT20_DC_SUBCARRIER = 32;         // DC null subcarrier
constexpr uint8_t HT20_SELECTED_BAND_SIZE = 12;    // Selected subcarriers for motion detection
constexpr uint8_t DEFAULT_SUBCARRIERS[HT20_SELECTED_BAND_SIZE] = {
    14, 17, 20, 23, 26, 29, 35, 38, 41, 44, 47, 50
};
using SelectedSubcarriers = std::array<uint8_t, HT20_SELECTED_BAND_SIZE>;

constexpr SelectedSubcarriers make_default_subcarriers() {
    SelectedSubcarriers subcarriers{};
    for (uint8_t i = 0; i < HT20_SELECTED_BAND_SIZE; ++i) {
        subcarriers[i] = DEFAULT_SUBCARRIERS[i];
    }
    return subcarriers;
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
        float Q = static_cast<float>(csi_data[sc_idx * 2]);
        float I = static_cast<float>(csi_data[sc_idx * 2 + 1]);
        out[valid_count++] = std::sqrt(I * I + Q * Q);
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
