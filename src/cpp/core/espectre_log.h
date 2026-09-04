/*
 * ESPectre - Log Sink
 *
 * Portable logging contract shared by the SDK and its frontends.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdarg>
#include <cstdint>

namespace espectre {

/** Severity attached to one ESPectre log message. */
enum class LogLevel : uint8_t {
  ERROR = 1,
  WARNING = 2,
  INFO = 3,
  DEBUG = 4,
  VERBOSE = 5,
};

/** Return whether a sink accepts a message with the supplied level and tag. */
using LogEnabledCallback = bool (*)(void *context, LogLevel level, const char *tag);

/** Consume one enabled ESPectre log message before the supplied argument list expires. */
using LogWriteCallback = void (*)(void *context, LogLevel level, const char *tag, int line,
                                  const char *format, va_list args);

/**
 * Frontend-owned logging callbacks.
 *
 * ESPectre copies this value when it is registered but does not own `context`.
 * The context and callbacks must remain valid until the sink is cleared.
 */
struct LogSink {
  void *context{nullptr};
  LogEnabledCallback enabled{nullptr};
  LogWriteCallback write{nullptr};
};

/**
 * Register a complete frontend logging sink.
 *
 * Registration must happen before runtime setup, and replacement is supported
 * only while no ESPectre runtime is active. An invalid sink leaves the current
 * registration unchanged.
 *
 * @param sink Callback value copied by ESPectre.
 * @return `true` when both required callbacks were registered.
 */
bool set_log_sink(const LogSink &sink);

/** Clear the current sink while no ESPectre runtime is active. */
void clear_log_sink();

/**
 * Return whether the current sink accepts one level and tag.
 *
 * @param level Message severity.
 * @param tag Stable logger tag.
 * @return `false` when no complete sink is registered or the sink filters the message.
 */
bool log_enabled(LogLevel level, const char *tag);

/** @cond INTERNAL */
namespace detail {

void log_printf(LogLevel level, const char *tag, int line, const char *format, ...)
#if defined(__GNUC__)
    __attribute__((format(printf, 4, 5)))
#endif
    ;

}  // namespace detail
/** @endcond */

}  // namespace espectre

#define ESPECTRE_LOG_AT_LEVEL(level, tag, format, ...)                                            \
  do {                                                                                            \
    const char *const espectre_log_tag__ = (tag);                                                 \
    if (::espectre::log_enabled((level), espectre_log_tag__)) {                                   \
      ::espectre::detail::log_printf((level), espectre_log_tag__, __LINE__, (format),             \
                                     ##__VA_ARGS__);                                               \
    }                                                                                             \
  } while (false)

#define ESPECTRE_LOGE(tag, format, ...)                                                           \
  ESPECTRE_LOG_AT_LEVEL(::espectre::LogLevel::ERROR, tag, format, ##__VA_ARGS__)
#define ESPECTRE_LOGW(tag, format, ...)                                                           \
  ESPECTRE_LOG_AT_LEVEL(::espectre::LogLevel::WARNING, tag, format, ##__VA_ARGS__)
#define ESPECTRE_LOGI(tag, format, ...)                                                           \
  ESPECTRE_LOG_AT_LEVEL(::espectre::LogLevel::INFO, tag, format, ##__VA_ARGS__)
#define ESPECTRE_LOGD(tag, format, ...)                                                           \
  ESPECTRE_LOG_AT_LEVEL(::espectre::LogLevel::DEBUG, tag, format, ##__VA_ARGS__)
#define ESPECTRE_LOGV(tag, format, ...)                                                           \
  ESPECTRE_LOG_AT_LEVEL(::espectre::LogLevel::VERBOSE, tag, format, ##__VA_ARGS__)
