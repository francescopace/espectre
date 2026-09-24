/*
 * ESPectre - Runtime Time
 *
 * Monotonic time helpers used by shared runtime components. Portable
 * shim: uses esp_timer when available and degrades on host builds.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

namespace espectre {

/** Monotonic time since boot in microseconds: `esp_timer` on ESP-IDF, `steady_clock` on hosts. */
uint64_t monotonic_now_us();
/** monotonic_now_us() in milliseconds, wrapping modulo 2^32. */
uint32_t monotonic_now_ms();

}  // namespace espectre
