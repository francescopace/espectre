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

uint64_t monotonic_now_us();
uint32_t monotonic_now_ms();

}  // namespace espectre
