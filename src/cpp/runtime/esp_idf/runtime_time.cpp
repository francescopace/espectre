/*
 * ESPectre - Runtime Time
 *
 * Monotonic time helpers used by shared runtime components.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "runtime_time.h"

#if __has_include("esp_timer.h")
#include "esp_timer.h"
#define ESPECTRE_HAVE_ESP_TIMER 1
#endif

namespace espectre {

uint64_t monotonic_now_us() {
#ifdef ESPECTRE_HAVE_ESP_TIMER
  return static_cast<uint64_t>(esp_timer_get_time());
#else
  return 0U;
#endif
}

uint32_t monotonic_now_ms() { return static_cast<uint32_t>(monotonic_now_us() / 1000ULL); }

}  // namespace espectre
