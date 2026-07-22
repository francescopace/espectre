/*
 * ESPectre - Debug Telemetry Log Helpers
 *
 * Applies runtime log filtering for debug telemetry.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "debug_telemetry_log_helpers.h"

#include <sdkconfig.h>

#include "espectre_log.h"

namespace espectre {

void configure_debug_telemetry_log_levels() {
#if CONFIG_ESPECTRE_DEBUG_TELEMETRY
  // Keep ESP-IDF internals at the global INFO level while exposing only the
  // shared runtime DEBUG telemetry tags.
  esp_log_level_set("espectre.runtime", ESP_LOG_DEBUG);
  esp_log_level_set("espectre.stream.runtime", ESP_LOG_DEBUG);
#endif
}

}  // namespace espectre
