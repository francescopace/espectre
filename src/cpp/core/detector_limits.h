/*
 * ESPectre - Detector Limits
 *
 * Shared detector window size and calibration buffer limits.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

// The public contract is temporal. These packet counts are only the resolved
// storage bounds used after the runtime has measured the CSI cadence.
constexpr uint32_t DETECTOR_WINDOW_SIZE_MS_MIN = 1000U;
constexpr uint32_t DETECTOR_WINDOW_SIZE_MS_MAX = 2000U;
constexpr uint32_t DETECTOR_WINDOW_SIZE_MS_DEFAULT = 1000U;
constexpr uint16_t DETECTOR_DEFAULT_WINDOW_SIZE = 100;
// A target-rate/window pair may resolve to a single slot. Readiness is governed
// by the derived temporal occupancy floor, never by an absolute packet count.
constexpr uint16_t DETECTOR_MIN_WINDOW_SIZE = 1;
// Bounded by measurement, not stack: detector working buffers are allocated
// to the resolved window, and the ceiling limits drift away from the feature
// geometry used during fitting.
constexpr uint16_t DETECTOR_MAX_WINDOW_SIZE = 1000;

constexpr uint16_t CALIBRATION_NUM_WINDOWS = 10;
constexpr uint16_t CALIBRATION_DEFAULT_BUFFER_SIZE =
    DETECTOR_DEFAULT_WINDOW_SIZE * CALIBRATION_NUM_WINDOWS;

// Detector timing contract, in microseconds. The packet counts above and in
// csi_features.h are what these durations resolve to at the nominal 100 pps; on
// a stream that runs faster or slower they are re-derived from the measured
// cadence. Keep aligned with src/python/micro_espectre/config.py.
constexpr uint32_t EVALUATION_INTERVAL_US = 250000U; // Time between evaluations
constexpr uint32_t L1_DELTA_LAG_US = 100000U;        // Profile-displacement lag
constexpr uint32_t TURB_AUTOCORR_LAG_US = 10000U;    // Autocorrelation lag
constexpr uint16_t DETECTOR_L1_DELTA_LAG_DEFAULT = 10U;
constexpr uint16_t DETECTOR_AUTOCORR_LAG_DEFAULT = 1U;
// The L1 profile ring is statically sized in firmware, so the displacement
// lag is capped. 32 packets covers the 100 ms contract up to ~320 pps, well
// past what any supported chip sustains; above that the lag saturates and
// spans less than 100 ms. Measured cost at 1000 pps is a few points of
// recall, and the decisive high-rate lag is the autocorrelation one, which
// stays far below this bound.
//
// The ring costs L1_DELTA_LAG_MAX x HT20_SELECTED_BAND_SIZE floats, i.e.
// 32 x 12 x 4 = 1536 bytes. An earlier revision of this comment said 4.5 KB,
// which never matched the declaration.
constexpr uint16_t L1_DELTA_LAG_MAX = 32;
// A cadence faster than this is not a CSI stream, it is a batch delivered
// faster than real time. The packet-rate estimator ignores it when deriving
// feature geometry; evaluation cadence still follows the packet timestamps.
// There is no upper bound because a stream slower than one window is already
// handled as a hole by the configured detector window duration.
constexpr uint32_t MIN_PLAUSIBLE_PACKET_INTERVAL_US = 200U;    // 5000 pps

}  // namespace espectre
