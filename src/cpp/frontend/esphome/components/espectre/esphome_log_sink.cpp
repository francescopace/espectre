/*
 * ESPectre - ESPHome Log Sink
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "esphome_log_sink.h"

#include "esphome/components/logger/logger.h"
#include "esphome/core/log.h"

namespace esphome {

namespace logger {

// ESPHome declares this method inline in logger.h but defines it in logger.cpp.
// Repeat the identical inline definition here so external components can use
// the per-tag filter without depending on an out-of-line symbol.
inline uint8_t Logger::level_for(const char *tag) {
#ifdef USE_LOGGER_RUNTIME_TAG_LEVELS
  auto it = this->log_levels_.find(tag);
  if (it != this->log_levels_.end())
    return it->second;
#endif
  return this->current_level_;
}

}  // namespace logger

namespace espectre_component {

namespace {

uint8_t esphome_log_level(::espectre::LogLevel level) {
  switch (level) {
    case ::espectre::LogLevel::ERROR:
      return ESPHOME_LOG_LEVEL_ERROR;
    case ::espectre::LogLevel::WARNING:
      return ESPHOME_LOG_LEVEL_WARN;
    case ::espectre::LogLevel::INFO:
      return ESPHOME_LOG_LEVEL_INFO;
    case ::espectre::LogLevel::DEBUG:
      return ESPHOME_LOG_LEVEL_DEBUG;
    case ::espectre::LogLevel::VERBOSE:
      return ESPHOME_LOG_LEVEL_VERBOSE;
  }
  return ESPHOME_LOG_LEVEL_NONE;
}

bool esphome_log_enabled(void *, ::espectre::LogLevel level, const char *tag) {
  return logger::global_logger != nullptr && tag != nullptr &&
         esphome_log_level(level) <= logger::global_logger->level_for(tag);
}

void esphome_log_write(void *, ::espectre::LogLevel level, const char *tag, int line,
                       const char *format, va_list args) {
  ::esphome::esp_log_vprintf_(esphome_log_level(level), tag, line, format, args);
}

}  // namespace

::espectre::LogSink make_esphome_log_sink() {
  return {nullptr, &esphome_log_enabled, &esphome_log_write};
}

}  // namespace espectre_component
}  // namespace esphome
