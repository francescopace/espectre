/*
 * ESPectre - Detector Timing
 *
 * Timestamp arithmetic shared by the evaluation cadence and the replay
 * harness.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

/**
 * Elapsed microseconds between two 32-bit arrival timestamps.
 *
 * The MAC receive timestamp wraps roughly every 71.6 minutes, so the delta is
 * taken modulo the counter width. A result past half the range is a counter
 * that went backwards rather than a very long gap, and is reported as zero so
 * callers ignore it instead of inventing an hour of coverage.
 */
inline uint32_t elapsed_since_timestamp_us(uint32_t now_us, uint32_t previous_us) {
  const uint32_t delta = now_us - previous_us;
  return delta < 0x80000000U ? delta : 0U;
}

}  // namespace espectre
