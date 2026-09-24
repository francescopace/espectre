/*
 * ESPectre - Filter Configuration
 *
 * Public filter ranges shared by detector implementations and RuntimeConfig.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

/** @name Low-pass filter on the turbulence stream, in Hz */
/** @{ */
constexpr float LOWPASS_CUTOFF_DEFAULT = 11.0f;
constexpr float LOWPASS_CUTOFF_MIN = 5.0f;
constexpr float LOWPASS_CUTOFF_MAX = 20.0f;
/** Sample rate the filter is designed for: the nominal 100 pps. */
constexpr float LOWPASS_SAMPLE_RATE = 100.0f;
/** @} */

/** @name Hampel outlier filter on the turbulence stream */
/** @{ */
/** Smallest window, in samples. */
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MIN = 3U;
/** Largest window, in samples. */
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_MAX = 11U;
constexpr uint8_t HAMPEL_TURBULENCE_WINDOW_DEFAULT = 7U;
/** Default outlier threshold, as a multiple of the median absolute deviation. */
constexpr float HAMPEL_TURBULENCE_THRESHOLD_DEFAULT = 5.0f;
/** @} */

}  // namespace espectre
