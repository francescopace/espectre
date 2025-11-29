#ifndef ESP_TIMER_H
#define ESP_TIMER_H

#include "esp_err.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Timer handle
typedef struct esp_timer *esp_timer_handle_t;

// Timer callback
typedef void (*esp_timer_cb_t)(void *arg);

// Timer creation arguments
typedef struct {
  esp_timer_cb_t callback;
  void *arg;
  const char *name;
  bool skip_unhandled_events;
} esp_timer_create_args_t;

// Mock timer functions
static inline esp_err_t
esp_timer_create(const esp_timer_create_args_t *create_args,
                 esp_timer_handle_t *out_handle) {
  (void)create_args;
  (void)out_handle;
  return ESP_OK;
}

static inline esp_err_t esp_timer_start_periodic(esp_timer_handle_t timer,
                                                 uint64_t period) {
  (void)timer;
  (void)period;
  return ESP_OK;
}

static inline esp_err_t esp_timer_start_once(esp_timer_handle_t timer,
                                             uint64_t timeout_us) {
  (void)timer;
  (void)timeout_us;
  return ESP_OK;
}

static inline esp_err_t esp_timer_stop(esp_timer_handle_t timer) {
  (void)timer;
  return ESP_OK;
}

static inline esp_err_t esp_timer_delete(esp_timer_handle_t timer) {
  (void)timer;
  return ESP_OK;
}

static inline int64_t esp_timer_get_time(void) { return 0; }

#ifdef __cplusplus
}
#endif

#endif // ESP_TIMER_H
