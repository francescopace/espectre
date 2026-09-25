/*
 * ESPectre - Log Sink
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "espectre_log.h"

#include <algorithm>
#include <chrono>
#include <limits>

#if __has_include("esp_timer.h")
#include "esp_timer.h"
#define ESPECTRE_LOG_HAVE_ESP_TIMER 1
#endif

namespace espectre {

namespace {

LogSink current_sink;
thread_local detail::LogSinkTiming thread_sink_timing;

uint64_t log_clock_us() {
#ifdef ESPECTRE_LOG_HAVE_ESP_TIMER
  return static_cast<uint64_t>(esp_timer_get_time());
#else
  const auto now = std::chrono::steady_clock::now().time_since_epoch();
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(now).count());
#endif
}

}  // namespace

bool set_log_sink(const LogSink &sink) {
  if (sink.enabled == nullptr || sink.write == nullptr) {
    return false;
  }
  current_sink = sink;
  return true;
}

void clear_log_sink() { current_sink = {}; }

bool log_enabled(LogLevel level, const char *tag) {
  return current_sink.enabled != nullptr && current_sink.write != nullptr &&
         current_sink.enabled(current_sink.context, level, tag);
}

namespace detail {

void log_printf(LogLevel level, const char *tag, int line, const char *format, ...) {
  if (current_sink.write == nullptr) {
    return;
  }
  va_list args;
  va_start(args, format);
  const uint64_t start_us = log_clock_us();
  current_sink.write(current_sink.context, level, tag, line, format, args);
  const uint64_t end_us = log_clock_us();
  va_end(args);
  const uint32_t elapsed_us = static_cast<uint32_t>(std::min<uint64_t>(
      end_us > start_us ? end_us - start_us : 0U, std::numeric_limits<uint32_t>::max()));
  thread_sink_timing.total_us += elapsed_us;
  thread_sink_timing.writes += 1U;
  thread_sink_timing.maximum_us = std::max(thread_sink_timing.maximum_us, elapsed_us);
}

LogSinkTiming take_log_sink_timing() {
  const LogSinkTiming timing = thread_sink_timing;
  thread_sink_timing = {};
  return timing;
}

uint64_t log_sink_total_us() { return thread_sink_timing.total_us; }

}  // namespace detail

}  // namespace espectre
