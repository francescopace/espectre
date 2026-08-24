/*
 * ESP-IDF MQTT client mock for bounded transport host tests.
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"
#include "esp_event.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *esp_mqtt_client_handle_t;

typedef enum {
  MQTT_EVENT_ANY = -1,
  MQTT_EVENT_ERROR,
  MQTT_EVENT_CONNECTED,
  MQTT_EVENT_DISCONNECTED,
  MQTT_EVENT_SUBSCRIBED,
  MQTT_EVENT_UNSUBSCRIBED,
  MQTT_EVENT_PUBLISHED,
  MQTT_EVENT_DATA,
} esp_mqtt_event_id_t;

typedef struct esp_mqtt_event {
  esp_mqtt_event_id_t event_id;
  const char *topic;
  int topic_len;
  const char *data;
  int data_len;
  int total_data_len;
  int current_data_offset;
} esp_mqtt_event_t;

typedef esp_mqtt_event_t *esp_mqtt_event_handle_t;

typedef struct {
  struct {
    struct {
      const char *uri;
    } address;
  } broker;
  struct {
    const char *username;
    struct {
      const char *password;
    } authentication;
  } credentials;
  struct {
    struct {
      const char *topic;
      const char *msg;
      int msg_len;
      int qos;
      bool retain;
    } last_will;
  } session;
  struct {
    uint64_t limit;
  } outbox;
} esp_mqtt_client_config_t;

typedef struct {
  esp_mqtt_client_handle_t init_result;
  esp_err_t start_result;
  int enqueue_result;
  int outbox_size;
  int subscribe_result;
  int init_calls;
  int start_calls;
  int stop_calls;
  int destroy_calls;
  int register_calls;
  int enqueue_calls;
  int subscribe_calls;
  uint64_t outbox_limit;
  char broker_uri[256];
  char enqueued_topics[32][256];
  char enqueued_payloads[32][4096];
  bool enqueued_retain[32];
  esp_event_handler_t event_handler;
  void *event_handler_arg;
} mqtt_client_mock_state_t;

extern mqtt_client_mock_state_t g_mqtt_client_mock;

void mqtt_client_mock_reset(void);
void mqtt_client_mock_emit(esp_mqtt_event_id_t event_id,
                           const char *topic,
                           const char *data,
                           int total_data_len,
                           int current_data_offset);

esp_mqtt_client_handle_t esp_mqtt_client_init(const esp_mqtt_client_config_t *config);
esp_err_t esp_mqtt_client_register_event(esp_mqtt_client_handle_t client,
                                         esp_mqtt_event_id_t event,
                                         esp_event_handler_t handler,
                                         void *handler_args);
esp_err_t esp_mqtt_client_start(esp_mqtt_client_handle_t client);
esp_err_t esp_mqtt_client_stop(esp_mqtt_client_handle_t client);
esp_err_t esp_mqtt_client_destroy(esp_mqtt_client_handle_t client);
int esp_mqtt_client_enqueue(esp_mqtt_client_handle_t client,
                            const char *topic,
                            const char *data,
                            int len,
                            int qos,
                            int retain,
                            bool store);
int esp_mqtt_client_subscribe(esp_mqtt_client_handle_t client, const char *topic, int qos);
int esp_mqtt_client_get_outbox_size(esp_mqtt_client_handle_t client);

#ifdef __cplusplus
}
#endif
