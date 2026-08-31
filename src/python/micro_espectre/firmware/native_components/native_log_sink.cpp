/*
 * Micro-ESPectre - Native Log Sink
 *
 * ESPectre SDK logging adapter owned by the MicroPython frontend.
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#ifndef NO_QSTR

#include "native_log_sink.h"

#include "espectre_core_sdk.h"

#include "esp_log.h"

namespace {

esp_log_level_t idf_log_level(espectre::LogLevel level) {
  switch (level) {
    case espectre::LogLevel::ERROR:
      return ESP_LOG_ERROR;
    case espectre::LogLevel::WARNING:
      return ESP_LOG_WARN;
    case espectre::LogLevel::INFO:
      return ESP_LOG_INFO;
    case espectre::LogLevel::DEBUG:
      return ESP_LOG_DEBUG;
    case espectre::LogLevel::VERBOSE:
      return ESP_LOG_VERBOSE;
  }
  return ESP_LOG_NONE;
}

bool idf_log_enabled(void *, espectre::LogLevel level, const char *tag) {
  return tag != nullptr && idf_log_level(level) <= esp_log_level_get(tag);
}

void idf_log_write(void *, espectre::LogLevel level, const char *tag, int,
                   const char *format, va_list args) {
  esp_log_va(ESP_LOG_CONFIG_INIT(idf_log_level(level) | ESP_LOG_CONFIGS_DEFAULT),
             tag, format, args);
}

}  // namespace

void espectre_native_ensure_log_sink() {
  static const bool registered =
      espectre::set_log_sink({nullptr, &idf_log_enabled, &idf_log_write});
  (void) registered;
}

#endif
