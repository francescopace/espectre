/*
 * ESPectre - Debug Telemetry Log Helpers
 *
 * Applies runtime log filtering for shared status and debug telemetry.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "debug_telemetry_log_helpers.h"

#include <sdkconfig.h>

#include "espectre_log.h"

namespace espectre {

void configure_debug_telemetry_log_levels() {
  // ESPHome keeps the global IDF default at ERROR. Enable only the tags the
  // shared SDK actually prints; Wi-Fi and lwIP debug stay compiled out of
  // those components.
  esp_log_level_set("espectre", ESP_LOG_INFO);
  esp_log_level_set("espectre.runtime", ESP_LOG_INFO);
  esp_log_level_set("CsiCapture", ESP_LOG_INFO);
  esp_log_level_set("TrafficGen", ESP_LOG_INFO);
  esp_log_level_set("WiFiLifecycle", ESP_LOG_INFO);
#if CONFIG_ESPECTRE_DEBUG_TELEMETRY
  esp_log_level_set("espectre.runtime", ESP_LOG_DEBUG);
  esp_log_level_set("espectre.stream.runtime", ESP_LOG_DEBUG);
#endif
}

}  // namespace espectre
