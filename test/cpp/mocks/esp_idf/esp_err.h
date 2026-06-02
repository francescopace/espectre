#ifndef ESP_ERR_H
#define ESP_ERR_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t esp_err_t;

// ESP-IDF error codes
#define ESP_OK 0
#define ESP_FAIL -1
#define ESP_ERR_NO_MEM 0x101
#define ESP_ERR_INVALID_ARG 0x102
#define ESP_ERR_INVALID_STATE 0x103
#define ESP_ERR_INVALID_SIZE 0x104
#define ESP_ERR_NOT_FOUND 0x105
#define ESP_ERR_NOT_SUPPORTED 0x106
#define ESP_ERR_TIMEOUT 0x107
#define ESP_ERR_INVALID_RESPONSE 0x108
#define ESP_ERR_INVALID_CRC 0x109
#define ESP_ERR_INVALID_VERSION 0x10A
#define ESP_ERR_INVALID_MAC 0x10B
#define ESP_ERR_WIFI_BASE 0x3000

// Error checking macros
#define ESP_ERROR_CHECK(x)                                                     \
  do {                                                                         \
    esp_err_t err_rc_ = (x);                                                   \
    (void)err_rc_;                                                             \
  } while (0)

// Error to name conversion
static inline const char *esp_err_to_name(esp_err_t code) {
  switch (code) {
  case ESP_OK:
    return "ESP_OK";
  case ESP_FAIL:
    return "ESP_FAIL";
  case ESP_ERR_NO_MEM:
    return "ESP_ERR_NO_MEM";
  case ESP_ERR_INVALID_ARG:
    return "ESP_ERR_INVALID_ARG";
  case ESP_ERR_INVALID_STATE:
    return "ESP_ERR_INVALID_STATE";
  default:
    return "UNKNOWN_ERROR";
  }
}

#ifdef __cplusplus
}
#endif

#endif // ESP_ERR_H
