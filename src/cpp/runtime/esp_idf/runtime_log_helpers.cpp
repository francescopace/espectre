/*
 * ESPectre - Runtime Log Helpers
 *
 * Applies runtime log filtering for shared production status messages.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "runtime_log_helpers.h"

#include "espectre_log.h"

namespace espectre {

void configure_runtime_log_levels() {
  // ESPHome keeps the global IDF default at ERROR. Enable only the tags used
  // by the shared production status surface; Wi-Fi and lwIP stay unchanged.
  esp_log_level_set("espectre", ESP_LOG_INFO);
  esp_log_level_set("espectre.runtime", ESP_LOG_INFO);
  esp_log_level_set("CsiCapture", ESP_LOG_INFO);
  esp_log_level_set("TrafficGen", ESP_LOG_INFO);
  esp_log_level_set("WiFiLifecycle", ESP_LOG_INFO);
}

}  // namespace espectre
