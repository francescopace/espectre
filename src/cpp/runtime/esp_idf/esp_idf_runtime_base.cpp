/*
 * ESPectre - ESP-IDF Runtime Base
 *
 * Shared platform diagnostics for ESP-IDF runtime implementations.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "esp_idf_runtime_base.h"

#include <sdkconfig.h>

#if __has_include("esp_heap_caps.h")
#include "esp_heap_caps.h"
#define ESPECTRE_HAVE_ESP_HEAP_CAPS 1
#endif

namespace espectre {

RuntimeDiagnosticsSnapshot EspIdfRuntimeBase::get_diagnostics() const {
  RuntimeDiagnosticsSnapshot diagnostics;
#ifdef ESPECTRE_HAVE_ESP_HEAP_CAPS
  diagnostics.free_memory_bytes = heap_caps_get_free_size(MALLOC_CAP_DEFAULT);
  diagnostics.minimum_free_memory_bytes = heap_caps_get_minimum_free_size(MALLOC_CAP_DEFAULT);
  diagnostics.largest_free_memory_block_bytes = heap_caps_get_largest_free_block(MALLOC_CAP_DEFAULT);
#endif
#if defined(CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ)
  diagnostics.cpu_frequency_mhz = CONFIG_ESP_DEFAULT_CPU_FREQ_MHZ;
#elif defined(CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ)
  diagnostics.cpu_frequency_mhz = CONFIG_ESP32_DEFAULT_CPU_FREQ_MHZ;
#endif
  const RuntimePerformanceDiagnosticsSnapshot performance = performance_diagnostics_.snapshot();
  diagnostics.performance_window_ready = performance.window_ready;
  diagnostics.performance_window_duration_us = performance.window_duration_us;
  diagnostics.runtime_load_percent = performance.runtime_load_percent;
  diagnostics.loop_samples = performance.loop_samples;
  diagnostics.loop_average_us = performance.loop_average_us;
  diagnostics.loop_maximum_us = performance.loop_maximum_us;
  diagnostics.detection_timing_supported = detection_timing_supported_;
  diagnostics.detection_samples = performance.detection_samples;
  diagnostics.detection_sum_us = performance.detection_sum_us;
  diagnostics.detection_average_us = performance.detection_average_us;
  diagnostics.detection_minimum_us = performance.detection_minimum_us;
  diagnostics.detection_maximum_us = performance.detection_maximum_us;
  return diagnostics;
}

}  // namespace espectre
