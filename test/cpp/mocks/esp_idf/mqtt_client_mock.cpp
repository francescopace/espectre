/*
 * ESP-IDF MQTT client mock for bounded transport host tests.
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "mqtt_client.h"

#include <algorithm>
#include <cstring>

mqtt_client_mock_state_t g_mqtt_client_mock{};

void mqtt_client_mock_reset(void) {
  g_mqtt_client_mock = {};
  g_mqtt_client_mock.init_result = &g_mqtt_client_mock;
  g_mqtt_client_mock.start_result = ESP_OK;
  g_mqtt_client_mock.enqueue_result = 1;
  g_mqtt_client_mock.subscribe_result = 1;
}

void mqtt_client_mock_emit(esp_mqtt_event_id_t event_id,
                           const char *topic,
                           const char *data,
                           int total_data_len,
                           int current_data_offset) {
  if (g_mqtt_client_mock.event_handler == nullptr) return;
  esp_mqtt_event_t event{};
  event.event_id = event_id;
  event.topic = topic;
  event.topic_len = topic != nullptr ? static_cast<int>(std::strlen(topic)) : 0;
  event.data = data;
  event.data_len = data != nullptr ? static_cast<int>(std::strlen(data)) : 0;
  event.total_data_len = total_data_len >= 0 ? total_data_len : event.data_len;
  event.current_data_offset = current_data_offset;
  g_mqtt_client_mock.event_handler(
      g_mqtt_client_mock.event_handler_arg, "MQTT_EVENTS", static_cast<int32_t>(event_id), &event);
}

esp_mqtt_client_handle_t esp_mqtt_client_init(const esp_mqtt_client_config_t *config) {
  g_mqtt_client_mock.init_calls++;
  if (config != nullptr) {
    g_mqtt_client_mock.outbox_limit = config->outbox.limit;
    if (config->broker.address.uri != nullptr) {
      std::strncpy(g_mqtt_client_mock.broker_uri,
                   config->broker.address.uri,
                   sizeof(g_mqtt_client_mock.broker_uri) - 1U);
    }
  }
  return g_mqtt_client_mock.init_result;
}

esp_err_t esp_mqtt_client_register_event(esp_mqtt_client_handle_t client,
                                         esp_mqtt_event_id_t event,
                                         esp_event_handler_t handler,
                                         void *handler_args) {
  (void) client;
  (void) event;
  g_mqtt_client_mock.register_calls++;
  g_mqtt_client_mock.event_handler = handler;
  g_mqtt_client_mock.event_handler_arg = handler_args;
  return ESP_OK;
}

esp_err_t esp_mqtt_client_start(esp_mqtt_client_handle_t client) {
  (void) client;
  g_mqtt_client_mock.start_calls++;
  return g_mqtt_client_mock.start_result;
}

esp_err_t esp_mqtt_client_stop(esp_mqtt_client_handle_t client) {
  (void) client;
  g_mqtt_client_mock.stop_calls++;
  return ESP_OK;
}

esp_err_t esp_mqtt_client_destroy(esp_mqtt_client_handle_t client) {
  (void) client;
  g_mqtt_client_mock.destroy_calls++;
  return ESP_OK;
}

int esp_mqtt_client_enqueue(esp_mqtt_client_handle_t client,
                            const char *topic,
                            const char *data,
                            int len,
                            int qos,
                            int retain,
                            bool store) {
  (void) client;
  (void) qos;
  (void) store;
  const int index = g_mqtt_client_mock.enqueue_calls++;
  if (index < 32) {
    std::strncpy(g_mqtt_client_mock.enqueued_topics[index],
                 topic != nullptr ? topic : "",
                 sizeof(g_mqtt_client_mock.enqueued_topics[index]) - 1U);
    const size_t payload_size = data == nullptr
        ? 0U
        : std::min(len > 0 ? static_cast<size_t>(len) : std::strlen(data),
                   sizeof(g_mqtt_client_mock.enqueued_payloads[index]) - 1U);
    if (payload_size > 0U) {
      std::memcpy(g_mqtt_client_mock.enqueued_payloads[index], data, payload_size);
    }
    g_mqtt_client_mock.enqueued_retain[index] = retain != 0;
  }
  return g_mqtt_client_mock.enqueue_result;
}

int esp_mqtt_client_subscribe(esp_mqtt_client_handle_t client, const char *topic, int qos) {
  (void) client;
  (void) topic;
  (void) qos;
  g_mqtt_client_mock.subscribe_calls++;
  return g_mqtt_client_mock.subscribe_result;
}

int esp_mqtt_client_get_outbox_size(esp_mqtt_client_handle_t client) {
  (void) client;
  return g_mqtt_client_mock.outbox_size;
}

namespace {
struct MqttClientMockInitializer {
  MqttClientMockInitializer() { mqtt_client_mock_reset(); }
} g_mqtt_client_mock_initializer;
}  // namespace
