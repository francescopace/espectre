/*
 * ESPectre - Log Helpers
 *
 * Portable logging macros shared across ESPHome, ESP-IDF, and host tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdarg>
#include <cstdio>

#if __has_include("esphome/core/log.h")
#include "esphome/core/log.h"
#elif defined(ESP_PLATFORM)
#include "esp_log.h"
#else
#ifndef ESP_LOGE
#define ESP_LOGE(tag, format, ...) std::fprintf(stderr, "[E][%s] " format "\n", tag, ##__VA_ARGS__)
#endif
#ifndef ESP_LOGW
#define ESP_LOGW(tag, format, ...) std::fprintf(stderr, "[W][%s] " format "\n", tag, ##__VA_ARGS__)
#endif
#ifndef ESP_LOGI
#define ESP_LOGI(tag, format, ...) std::fprintf(stdout, "[I][%s] " format "\n", tag, ##__VA_ARGS__)
#endif
#ifndef ESP_LOGD
#define ESP_LOGD(tag, format, ...) std::fprintf(stdout, "[D][%s] " format "\n", tag, ##__VA_ARGS__)
#endif
#ifndef ESP_LOGV
#define ESP_LOGV(tag, format, ...) ((void)0)
#endif
#endif

namespace espectre {

// progress fills the bar on a 0-1 scale of width. threshold_pos overlays a
// marker at that character index; pass -1 to hide it.
inline void log_progress_bar(const char *tag, float progress, int width = 20, int threshold_pos = -1,
                             const char *format = nullptr, ...) {
  if (width < 1) {
    width = 1;
  } else if (width > 20) {
    width = 20;
  }
  if (threshold_pos >= width) {
    threshold_pos = width - 1;
  }

  int filled = static_cast<int>(progress * static_cast<float>(width));
  filled = (filled < 0) ? 0 : (filled > width ? width : filled);

  char bar[24];
  int idx = 0;
  bar[idx++] = '[';
  for (int i = 0; i < width; i++) {
    if (threshold_pos >= 0 && i == threshold_pos) {
      bar[idx++] = '|';
    } else if (i < filled) {
      bar[idx++] = '#';
    } else {
      bar[idx++] = '-';
    }
  }
  bar[idx++] = ']';
  bar[idx] = '\0';

  if (format != nullptr) {
    char text[256];
    va_list args;
    va_start(args, format);
    std::vsnprintf(text, sizeof(text), format, args);
    va_end(args);
    ESP_LOGI(tag, "%s %s", bar, text);
  } else {
    ESP_LOGI(tag, "%s", bar);
  }
}

}  // namespace espectre
