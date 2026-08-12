/*
 * ESPectre - Streamer Firmware Entrypoint
 *
 * Streamer firmware application entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "streamer_frontend.h"

#include <esp_log.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "debug_telemetry_log_helpers.h"
#include "espectre_banner.h"

static const char *TAG = "espectre.streamer.app";

namespace {
TickType_t clamp_delay_to_tick_(uint32_t delay_ms) {
  const TickType_t ticks = pdMS_TO_TICKS(delay_ms);
  return ticks > 0 ? ticks : 1;
}
}  // namespace

extern "C" void app_main() {
  espectre::configure_debug_telemetry_log_levels();
  espectre::log_espectre_banner([](const char *line) { ESP_LOGI(TAG, "%s", line); });
  static espectre::StreamerFrontend frontend;
  if (!frontend.setup()) {
    return;
  }

  while (true) {
    frontend.loop();
    // Keep the control loop responsive so the external pacing socket is
    // drained quickly and collector address changes propagate with low jitter.
    // Clamp to at least one tick so lower FreeRTOS tick rates still yield here.
    vTaskDelay(clamp_delay_to_tick_(2U));
  }
}
