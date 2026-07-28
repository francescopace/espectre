/*
 * ESPectre - Detector Limits
 *
 * Shared detector window size and calibration buffer limits.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

namespace espectre {

constexpr uint16_t DETECTOR_DEFAULT_WINDOW_SIZE = 100;
// Measured floor: the window features are estimator averages, and under
// roughly 100 samples they get noisy enough that startup calibration lifts
// the threshold to hold false positives and recall collapses instead. A
// 25-sample window scores 60.2% recall where 100 scores 98.7%.
constexpr uint16_t DETECTOR_MIN_WINDOW_SIZE = 100;
// Bounded by measurement, not stack: the detector working buffers are heap
// allocated to the real window, so the ceiling now only caps how far the
// window may drift from the rate the coefficients were fitted at. A window
// wide enough to span several seconds smears short movements into the
// background; the 1000 pps capture scores 83.5% recall at a 2 s window
// against 75.1% at 200 samples, but at the cost of a slower response.
// Kept in lockstep with RUNTIME_SEGMENTATION_WINDOW_SIZE_MAX by static_assert.
constexpr uint16_t DETECTOR_MAX_WINDOW_SIZE = 200;

constexpr uint16_t CALIBRATION_NUM_WINDOWS = 10;
constexpr uint16_t CALIBRATION_DEFAULT_BUFFER_SIZE =
    DETECTOR_DEFAULT_WINDOW_SIZE * CALIBRATION_NUM_WINDOWS;

// Detector timing contract, in microseconds. The packet counts above and in
// csi_features.h are what these durations resolve to at the nominal 100 pps; on
// a stream that runs faster or slower they are re-derived from the measured
// cadence. Keep aligned with src/python/micro_espectre/config.py.
constexpr uint32_t SEG_WINDOW_US = 1000000U;         // Detector window span
constexpr uint32_t EVALUATION_INTERVAL_US = 250000U; // Time between evaluations
constexpr uint32_t L1_DELTA_LAG_US = 100000U;        // Profile-displacement lag
constexpr uint32_t TURB_AUTOCORR_LAG_US = 10000U;    // Autocorrelation lag
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
// Rate-derived windows use the same bounds as configured ones: the floor is
// the measured one and the ceiling is the stack one, and neither changes
// because the cadence rather than the operator chose the value.
// Treat 80-133 pps as nominal: adapting inside this band costs feature
// homogeneity and buys nothing.
constexpr float RATE_ADAPTATION_DEAD_BAND = 0.25f;
// A cadence faster than this is not a CSI stream, it is a batch delivered
// faster than real time. Elapsed time is only trusted above it; below, the
// packet-count fallback covers. There is no upper bound because a stream
// slower than one window is already handled as a hole by SEG_WINDOW_US.
constexpr uint32_t MIN_PLAUSIBLE_PACKET_INTERVAL_US = 200U;    // 5000 pps

}  // namespace espectre
