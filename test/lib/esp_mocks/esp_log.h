#ifndef ESP_LOG_H
#define ESP_LOG_H

#include <stdio.h>
#include <stdarg.h>

#ifdef __cplusplus
extern "C" {
#endif

// Log levels
typedef enum {
    ESP_LOG_NONE,
    ESP_LOG_ERROR,
    ESP_LOG_WARN,
    ESP_LOG_INFO,
    ESP_LOG_DEBUG,
    ESP_LOG_VERBOSE
} esp_log_level_t;

// Color codes for terminal output
#define LOG_COLOR_BLACK   "30"
#define LOG_COLOR_RED     "31"
#define LOG_COLOR_GREEN   "32"
#define LOG_COLOR_BROWN   "33"
#define LOG_COLOR_BLUE    "34"
#define LOG_COLOR_PURPLE  "35"
#define LOG_COLOR_CYAN    "36"
#define LOG_COLOR(COLOR)  "\033[0;" COLOR "m"
#define LOG_BOLD(COLOR)   "\033[1;" COLOR "m"
#define LOG_RESET_COLOR   "\033[0m"
#define LOG_COLOR_E       LOG_COLOR(LOG_COLOR_RED)
#define LOG_COLOR_W       LOG_COLOR(LOG_COLOR_BROWN)
#define LOG_COLOR_I       LOG_COLOR(LOG_COLOR_GREEN)
#define LOG_COLOR_D       LOG_COLOR(LOG_COLOR_CYAN)
#define LOG_COLOR_V

// Mock logging macros
#define ESP_LOGE(tag, format, ...) printf(LOG_COLOR_E "E (%s) %s: " format LOG_RESET_COLOR "\n", "native", tag, ##__VA_ARGS__)
#define ESP_LOGW(tag, format, ...) printf(LOG_COLOR_W "W (%s) %s: " format LOG_RESET_COLOR "\n", "native", tag, ##__VA_ARGS__)
#define ESP_LOGI(tag, format, ...) printf(LOG_COLOR_I "I (%s) %s: " format LOG_RESET_COLOR "\n", "native", tag, ##__VA_ARGS__)
#define ESP_LOGD(tag, format, ...) printf(LOG_COLOR_D "D (%s) %s: " format LOG_RESET_COLOR "\n", "native", tag, ##__VA_ARGS__)
#define ESP_LOGV(tag, format, ...) printf(LOG_COLOR_V "V (%s) %s: " format LOG_RESET_COLOR "\n", "native", tag, ##__VA_ARGS__)

// Log level setting (no-op in mock)
static inline void esp_log_level_set(const char* tag, esp_log_level_t level) {
    (void)tag;
    (void)level;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_LOG_H
