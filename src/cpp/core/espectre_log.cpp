/*
 * ESPectre - Log Sink
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */

#include "espectre_log.h"

namespace espectre {

namespace {

LogSink current_sink;

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
  current_sink.write(current_sink.context, level, tag, line, format, args);
  va_end(args);
}

}  // namespace detail

}  // namespace espectre
