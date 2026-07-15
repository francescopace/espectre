/*
 * ESPectre - Runtime Time
 *
 * Monotonic time helpers used by shared runtime components.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

namespace espectre {

uint64_t monotonic_now_us();
uint32_t monotonic_now_ms();

}  // namespace espectre
