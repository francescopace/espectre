/*
 * ESPectre - Arduino Core Logging Mock
 *
 * Host-side stand-in for the Arduino-ESP32 log header, so the Arduino-style
 * sensor builds its core log sink in host tests. The ROM printf it writes
 * through is captured instead of printed.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdarg>
#include <cstdio>
#include <string>

#ifndef ARDUHAL_LOG_LEVEL
#define ARDUHAL_LOG_LEVEL 3
#endif

namespace arduino_log_mock {

inline std::string output;

inline void reset() { output.clear(); }

}  // namespace arduino_log_mock

inline int ets_printf(const char *format, ...) {
  char line[256];
  va_list args;
  va_start(args, format);
  const int written = vsnprintf(line, sizeof(line), format, args);
  va_end(args);
  arduino_log_mock::output += line;
  return written;
}
