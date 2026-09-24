/*
 * ESPectre - CSI Public Types
 *
 * Stable HT20 dimensions and detector subcarrier selection shared by the
 * core-only and full-runtime SDK surfaces.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <cstdint>

namespace espectre {

/**
 * @name HT20 layout
 * The detectors work on one layout: 64 subcarriers over 20 MHz, centered, as
 * interleaved 8-bit I/Q pairs. See
 * [CSI.md](https://github.com/francescopace/espectre/blob/main/docs/CSI.md#normalization)
 * for how other payload sizes map onto it.
 * @{
 */
constexpr uint16_t HT20_NUM_SUBCARRIERS = 64U;
/** Normalized payload size in bytes: 64 subcarriers of I/Q. */
constexpr uint16_t HT20_CSI_LEN = 128U;
/** Raw size of a payload carrying two HT20 estimates. */
constexpr uint16_t HT20_CSI_LEN_DOUBLE = 256U;
/** Raw size of a short HT estimate: 57 subcarriers. */
constexpr uint16_t HT20_CSI_LEN_SHORT = 114U;
/** Raw size of a compact LLTF estimate: 53 subcarriers, `-26..+26`. */
constexpr uint16_t LLTF20_CSI_LEN_SHORT = 106U;
/** Raw size of a payload carrying two short HT estimates. */
constexpr uint16_t HT20_CSI_LEN_SHORT_DOUBLE = 228U;
/** Bytes of padding before a short HT estimate when it is normalized. */
constexpr uint8_t HT20_CSI_LEN_SHORT_LEFT_PAD = 8U;
/** First live subcarrier index; lower bins are guard tones. */
constexpr uint8_t HT20_GUARD_BAND_LOW = 4U;
/** Last live subcarrier index; higher bins are guard tones. */
constexpr uint8_t HT20_GUARD_BAND_HIGH = 60U;
/** Index of the DC null in the centered layout. */
constexpr uint8_t HT20_DC_SUBCARRIER = 32U;
/** Number of subcarriers the detectors measure on. */
constexpr uint8_t HT20_SELECTED_BAND_SIZE = 12U;

/** Subcarriers the detectors measure on; fixed for this SDK version. */
inline constexpr uint8_t DEFAULT_SUBCARRIERS[HT20_SELECTED_BAND_SIZE] = {
    4U, 8U, 13U, 18U, 23U, 28U, 36U, 41U, 46U, 51U, 56U, 60U,
};
/** A selection of `HT20_SELECTED_BAND_SIZE` subcarrier indices. */
using SelectedSubcarriers = std::array<uint8_t, HT20_SELECTED_BAND_SIZE>;

/** DEFAULT_SUBCARRIERS as a SelectedSubcarriers value. */
constexpr SelectedSubcarriers make_default_subcarriers() {
  SelectedSubcarriers subcarriers{};
  for (uint8_t i = 0U; i < HT20_SELECTED_BAND_SIZE; ++i) {
    subcarriers[i] = DEFAULT_SUBCARRIERS[i];
  }
  return subcarriers;
}
/** @} */

}  // namespace espectre
