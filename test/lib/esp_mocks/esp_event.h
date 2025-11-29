#ifndef ESP_EVENT_H
#define ESP_EVENT_H

#include "esp_err.h"
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// Event loop handle
typedef void *esp_event_loop_handle_t;

// Event base type
typedef const char *esp_event_base_t;

// Event handler
typedef void (*esp_event_handler_t)(void *event_handler_arg,
                                    esp_event_base_t event_base,
                                    int32_t event_id, void *event_data);

// Event handler instance
typedef void *esp_event_handler_instance_t;

// WiFi events
#define WIFI_EVENT "WIFI_EVENT"

typedef enum {
  WIFI_EVENT_STA_START,
  WIFI_EVENT_STA_STOP,
  WIFI_EVENT_STA_CONNECTED,
  WIFI_EVENT_STA_DISCONNECTED,
  WIFI_EVENT_AP_START,
  WIFI_EVENT_AP_STOP,
} wifi_event_t;

// IP events
#define IP_EVENT "IP_EVENT"

typedef enum {
  IP_EVENT_STA_GOT_IP,
  IP_EVENT_STA_LOST_IP,
} ip_event_t;

// Mock event functions
static inline esp_err_t esp_event_loop_create_default(void) { return ESP_OK; }

static inline esp_err_t
esp_event_handler_register(esp_event_base_t event_base, int32_t event_id,
                           esp_event_handler_t event_handler,
                           void *event_handler_arg) {
  (void)event_base;
  (void)event_id;
  (void)event_handler;
  (void)event_handler_arg;
  return ESP_OK;
}

static inline esp_err_t
esp_event_handler_unregister(esp_event_base_t event_base, int32_t event_id,
                             esp_event_handler_t event_handler) {
  (void)event_base;
  (void)event_id;
  (void)event_handler;
  return ESP_OK;
}

static inline esp_err_t esp_event_handler_instance_register(
    esp_event_base_t event_base, int32_t event_id,
    esp_event_handler_t event_handler, void *event_handler_arg,
    esp_event_handler_instance_t *instance) {
  (void)event_base;
  (void)event_id;
  (void)event_handler;
  (void)event_handler_arg;
  if (instance)
    *instance = NULL;
  return ESP_OK;
}

static inline esp_err_t
esp_event_handler_instance_unregister(esp_event_base_t event_base,
                                      int32_t event_id,
                                      esp_event_handler_instance_t instance) {
  (void)event_base;
  (void)event_id;
  (void)instance;
  return ESP_OK;
}

static inline esp_err_t esp_event_post(esp_event_base_t event_base,
                                       int32_t event_id, void *event_data,
                                       size_t event_data_size,
                                       uint32_t ticks_to_wait) {
  (void)event_base;
  (void)event_id;
  (void)event_data;
  (void)event_data_size;
  (void)ticks_to_wait;
  return ESP_OK;
}

#ifdef __cplusplus
}
#endif

#endif // ESP_EVENT_H
