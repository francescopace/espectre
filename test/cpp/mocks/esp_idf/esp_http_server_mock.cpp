/*
 * ESP-IDF HTTP server mock for Direct HTTP host tests.
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "esp_http_server.h"

#include <algorithm>
#include <cstring>

httpd_mock_state_t g_httpd_mock{};

namespace {

void copy_string(char *target, size_t capacity, const char *value) {
  if (capacity == 0U) return;
  std::strncpy(target, value != nullptr ? value : "", capacity - 1U);
  target[capacity - 1U] = '\0';
}

char *header_storage(const char *name, size_t *capacity) {
  if (std::strcmp(name, "Origin") == 0) {
    *capacity = sizeof(g_httpd_mock.origin);
    return g_httpd_mock.origin;
  }
  if (std::strcmp(name, "Content-Type") == 0) {
    *capacity = sizeof(g_httpd_mock.content_type);
    return g_httpd_mock.content_type;
  }
  if (std::strcmp(name, "Authorization") == 0) {
    *capacity = sizeof(g_httpd_mock.authorization);
    return g_httpd_mock.authorization;
  }
  if (std::strcmp(name, "Access-Control-Request-Private-Network") == 0) {
    *capacity = sizeof(g_httpd_mock.request_private_network);
    return g_httpd_mock.request_private_network;
  }
  *capacity = 0U;
  return nullptr;
}

esp_err_t capture_payload(httpd_req_t *request, const char *payload, size_t length, bool chunk) {
  if (g_httpd_mock.send_result != ESP_OK) return g_httpd_mock.send_result;
  if (chunk) g_httpd_mock.chunk_calls++;
  const int index = g_httpd_mock.send_calls++;
  if (index >= 0 && index < 64) {
    const size_t copied = std::min(length, sizeof(g_httpd_mock.sent_payloads[index]));
    if (payload != nullptr && copied > 0U) {
      std::memcpy(g_httpd_mock.sent_payloads[index], payload, copied);
    }
    g_httpd_mock.sent_lengths[index] = copied;
    g_httpd_mock.sent_fds[index] = request != nullptr ? request->fd : -1;
  }
  return ESP_OK;
}

}  // namespace

void httpd_mock_reset(void) {
  g_httpd_mock = {};
  g_httpd_mock.start_result = ESP_OK;
  g_httpd_mock.register_result = ESP_OK;
  g_httpd_mock.receive_result = ESP_OK;
  g_httpd_mock.send_result = ESP_OK;
}

void httpd_mock_set_header(const char *name, const char *value) {
  size_t capacity = 0U;
  char *target = header_storage(name, &capacity);
  if (target != nullptr) copy_string(target, capacity, value);
}

void httpd_mock_set_incoming(const char *payload) {
  httpd_mock_set_incoming_bytes(payload, payload != nullptr ? std::strlen(payload) : 0U);
}

void httpd_mock_set_incoming_bytes(const void *payload, size_t length) {
  g_httpd_mock.incoming_length = std::min(length, sizeof(g_httpd_mock.incoming_payload));
  if (payload != nullptr && g_httpd_mock.incoming_length > 0U) {
    std::memcpy(g_httpd_mock.incoming_payload, payload, g_httpd_mock.incoming_length);
  }
}

esp_err_t httpd_start(httpd_handle_t *handle, const httpd_config_t *config) {
  g_httpd_mock.start_calls++;
  if (config != nullptr) g_httpd_mock.last_config = *config;
  if (g_httpd_mock.start_result == ESP_OK && handle != nullptr) *handle = &g_httpd_mock;
  return g_httpd_mock.start_result;
}

esp_err_t httpd_stop(httpd_handle_t handle) {
  (void) handle;
  g_httpd_mock.stop_calls++;
  return ESP_OK;
}

esp_err_t httpd_register_uri_handler(httpd_handle_t handle, const httpd_uri_t *uri) {
  (void) handle;
  const int index = g_httpd_mock.register_calls++;
  if (uri != nullptr) {
    if (index == 0) g_httpd_mock.registered_uri = *uri;
    if (index >= 0 && index < 8) g_httpd_mock.registered_uris[index] = *uri;
  }
  return g_httpd_mock.register_result;
}

size_t httpd_req_get_hdr_value_len(httpd_req_t *request, const char *name) {
  (void) request;
  size_t capacity = 0U;
  const char *value = header_storage(name, &capacity);
  return value != nullptr ? std::strlen(value) : 0U;
}

esp_err_t httpd_req_get_hdr_value_str(httpd_req_t *request,
                                       const char *name,
                                       char *buffer,
                                       size_t size) {
  (void) request;
  size_t capacity = 0U;
  const char *value = header_storage(name, &capacity);
  if (value == nullptr || std::strlen(value) + 1U > size) return ESP_ERR_INVALID_SIZE;
  std::strcpy(buffer, value);
  return ESP_OK;
}

int httpd_req_recv(httpd_req_t *request, char *buffer, size_t size) {
  if (g_httpd_mock.receive_result != ESP_OK || request == nullptr || buffer == nullptr) return -1;
  if (request->receive_offset >= g_httpd_mock.incoming_length) return 0;
  const size_t available = g_httpd_mock.incoming_length - request->receive_offset;
  const size_t copied = std::min(size, available);
  std::memcpy(buffer, g_httpd_mock.incoming_payload + request->receive_offset, copied);
  request->receive_offset += copied;
  return static_cast<int>(copied);
}

esp_err_t httpd_resp_send_err(httpd_req_t *request, const char *status, const char *message) {
  (void) request;
  g_httpd_mock.response_error_calls++;
  copy_string(g_httpd_mock.response_status, sizeof(g_httpd_mock.response_status), status);
  copy_string(g_httpd_mock.response_message, sizeof(g_httpd_mock.response_message), message);
  return ESP_OK;
}

esp_err_t httpd_resp_set_type(httpd_req_t *request, const char *type) {
  (void) request;
  copy_string(g_httpd_mock.response_type, sizeof(g_httpd_mock.response_type), type);
  return ESP_OK;
}

esp_err_t httpd_resp_set_hdr(httpd_req_t *request, const char *name, const char *value) {
  (void) request;
  if (std::strcmp(name, "Access-Control-Allow-Origin") == 0) {
    copy_string(g_httpd_mock.allow_origin, sizeof(g_httpd_mock.allow_origin), value);
  } else if (std::strcmp(name, "Access-Control-Allow-Private-Network") == 0) {
    copy_string(g_httpd_mock.allow_private_network,
                sizeof(g_httpd_mock.allow_private_network), value);
  } else if (std::strcmp(name, "Cache-Control") == 0) {
    copy_string(g_httpd_mock.cache_control, sizeof(g_httpd_mock.cache_control), value);
  }
  return ESP_OK;
}

esp_err_t httpd_resp_set_status(httpd_req_t *request, const char *status) {
  (void) request;
  copy_string(g_httpd_mock.response_status, sizeof(g_httpd_mock.response_status), status);
  return ESP_OK;
}

esp_err_t httpd_resp_send(httpd_req_t *request, const char *payload, size_t length) {
  return capture_payload(request, payload, length, false);
}

esp_err_t httpd_resp_send_chunk(httpd_req_t *request, const char *payload, size_t length) {
  return capture_payload(request, payload, length, true);
}

esp_err_t httpd_req_async_handler_begin(httpd_req_t *request, httpd_req_t **out) {
  if (request == nullptr || out == nullptr) return ESP_ERR_INVALID_ARG;
  auto *copy = new httpd_req_t(*request);
  copy->async_copy = true;
  *out = copy;
  g_httpd_mock.async_begin_calls++;
  if (g_httpd_mock.async_begin_callback != nullptr) {
    g_httpd_mock.async_begin_callback(g_httpd_mock.async_begin_callback_context);
  }
  return ESP_OK;
}

esp_err_t httpd_req_async_handler_complete(httpd_req_t *request) {
  if (request == nullptr) return ESP_ERR_INVALID_ARG;
  g_httpd_mock.async_complete_calls++;
  if (request->async_copy) delete request;
  return ESP_OK;
}

int httpd_req_to_sockfd(httpd_req_t *request) { return request != nullptr ? request->fd : -1; }

namespace {
struct HttpdMockInitializer {
  HttpdMockInitializer() { httpd_mock_reset(); }
} g_httpd_mock_initializer;
}  // namespace
