#pragma once

// Mock ESPHome logging for PlatformIO tests
// Redirects to ESP-IDF logging

#include <esp_log.h>

// ESPHome uses same log macros as ESP-IDF, just ensure they're available
// No need to redefine, ESP-IDF already provides:
// ESP_LOGD, ESP_LOGI, ESP_LOGW, ESP_LOGE, ESP_LOGV

// ESPHome-specific: LOGCONFIG is same as LOGI
#ifndef ESP_LOGCONFIG
#define ESP_LOGCONFIG(tag, ...) ESP_LOGI(tag, __VA_ARGS__)
#endif

// ESPHome log level constants
#define ESPHOME_LOG_LEVEL_NONE 0
#define ESPHOME_LOG_LEVEL_ERROR 1
#define ESPHOME_LOG_LEVEL_WARN 2
#define ESPHOME_LOG_LEVEL_INFO 3
#define ESPHOME_LOG_LEVEL_CONFIG 4
#define ESPHOME_LOG_LEVEL_DEBUG 5
#define ESPHOME_LOG_LEVEL_VERBOSE 6
#define ESPHOME_LOG_LEVEL_VERY_VERBOSE 7

namespace esphome {

// Mock log functions
inline void esp_log_printf_(int level, const char *tag, const char *format, ...) {
    // Redirect to ESP-IDF logging
}

} // namespace esphome
