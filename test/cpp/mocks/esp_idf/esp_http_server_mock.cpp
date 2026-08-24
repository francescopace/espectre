/*
 * ESP-IDF HTTP server mock for Direct WebSocket host tests.
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "esp_http_server.h"

#include <algorithm>
#include <cstring>

httpd_mock_state_t g_httpd_mock{};

void httpd_mock_reset(void) {
  g_httpd_mock = {};
  g_httpd_mock.start_result = ESP_OK;
  g_httpd_mock.register_result = ESP_OK;
  g_httpd_mock.client_list_result = ESP_OK;
  g_httpd_mock.receive_result = ESP_OK;
  g_httpd_mock.send_result = ESP_OK;
  g_httpd_mock.send_completion_result = ESP_OK;
  g_httpd_mock.incoming_type = HTTPD_WS_TYPE_TEXT;
  g_httpd_mock.incoming_final = true;
}

void httpd_mock_set_header(const char *name, const char *value) {
  char *target = std::strcmp(name, "Origin") == 0 ? g_httpd_mock.origin : g_httpd_mock.subprotocol;
  const size_t capacity = std::strcmp(name, "Origin") == 0 ? sizeof(g_httpd_mock.origin) : sizeof(g_httpd_mock.subprotocol);
  std::strncpy(target, value != nullptr ? value : "", capacity - 1U);
}

void httpd_mock_set_incoming(const char *payload, httpd_ws_type_t type, bool final, bool fragmented) {
  g_httpd_mock.incoming_length = payload != nullptr ? std::strlen(payload) : 0U;
  if (payload != nullptr) {
    std::memcpy(g_httpd_mock.incoming_payload, payload, g_httpd_mock.incoming_length);
  }
  g_httpd_mock.incoming_type = type;
  g_httpd_mock.incoming_final = final;
  g_httpd_mock.incoming_fragmented = fragmented;
}

void httpd_mock_set_clients(const int *fds, size_t count) {
  g_httpd_mock.client_count = std::min(count, static_cast<size_t>(16U));
  for (size_t index = 0U; index < g_httpd_mock.client_count; ++index) {
    g_httpd_mock.client_fds[index] = fds[index];
    g_httpd_mock.websocket_clients[index] = true;
  }
}

void httpd_mock_complete_next_send(esp_err_t result) {
  if (g_httpd_mock.pending_send_completions == 0U) return;
  transfer_complete_cb callback = g_httpd_mock.pending_send_callbacks[0];
  void *arg = g_httpd_mock.pending_send_args[0];
  const int socket = g_httpd_mock.pending_send_sockets[0];
  for (size_t index = 1U; index < g_httpd_mock.pending_send_completions; ++index) {
    g_httpd_mock.pending_send_callbacks[index - 1U] = g_httpd_mock.pending_send_callbacks[index];
    g_httpd_mock.pending_send_args[index - 1U] = g_httpd_mock.pending_send_args[index];
    g_httpd_mock.pending_send_sockets[index - 1U] = g_httpd_mock.pending_send_sockets[index];
  }
  g_httpd_mock.pending_send_completions--;
  if (callback != nullptr) callback(result, socket, arg);
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
  g_httpd_mock.register_calls++;
  if (uri != nullptr) g_httpd_mock.registered_uri = *uri;
  return g_httpd_mock.register_result;
}

const char *header_value(const char *name) {
  return std::strcmp(name, "Origin") == 0 ? g_httpd_mock.origin : g_httpd_mock.subprotocol;
}

size_t httpd_req_get_hdr_value_len(httpd_req_t *request, const char *name) {
  (void) request;
  return std::strlen(header_value(name));
}

esp_err_t httpd_req_get_hdr_value_str(httpd_req_t *request, const char *name, char *buffer, size_t size) {
  (void) request;
  const char *value = header_value(name);
  if (std::strlen(value) + 1U > size) return ESP_ERR_INVALID_SIZE;
  std::strcpy(buffer, value);
  return ESP_OK;
}

esp_err_t httpd_resp_send_err(httpd_req_t *request, const char *status, const char *message) {
  (void) request;
  g_httpd_mock.response_error_calls++;
  std::strncpy(g_httpd_mock.response_status, status, sizeof(g_httpd_mock.response_status) - 1U);
  std::strncpy(g_httpd_mock.response_message, message, sizeof(g_httpd_mock.response_message) - 1U);
  return ESP_OK;
}

int httpd_req_to_sockfd(httpd_req_t *request) { return request != nullptr ? request->fd : -1; }

esp_err_t httpd_ws_recv_frame(httpd_req_t *request, httpd_ws_frame_t *frame, size_t max_len) {
  (void) request;
  if (g_httpd_mock.receive_result != ESP_OK) return g_httpd_mock.receive_result;
  frame->len = g_httpd_mock.incoming_length;
  frame->type = g_httpd_mock.incoming_type;
  frame->final = g_httpd_mock.incoming_final;
  frame->fragmented = g_httpd_mock.incoming_fragmented;
  if (max_len != 0U && frame->payload != nullptr) {
    std::memcpy(frame->payload, g_httpd_mock.incoming_payload, std::min(max_len, g_httpd_mock.incoming_length));
  }
  return ESP_OK;
}

esp_err_t httpd_get_client_list(httpd_handle_t handle, size_t *count, int *fds) {
  (void) handle;
  if (g_httpd_mock.client_list_result != ESP_OK) return g_httpd_mock.client_list_result;
  const size_t copy_count = std::min(*count, g_httpd_mock.client_count);
  for (size_t index = 0U; index < copy_count; ++index) fds[index] = g_httpd_mock.client_fds[index];
  *count = copy_count;
  return ESP_OK;
}

httpd_ws_client_info_t httpd_ws_get_fd_info(httpd_handle_t handle, int fd) {
  (void) handle;
  for (size_t index = 0U; index < g_httpd_mock.client_count; ++index) {
    if (g_httpd_mock.client_fds[index] == fd) {
      return g_httpd_mock.websocket_clients[index] ? HTTPD_WS_CLIENT_WEBSOCKET : HTTPD_WS_CLIENT_HTTP;
    }
  }
  return HTTPD_WS_CLIENT_INVALID;
}

esp_err_t httpd_ws_send_data_async(httpd_handle_t handle,
                                   int socket,
                                   httpd_ws_frame_t *frame,
                                   transfer_complete_cb callback,
                                   void *arg) {
  (void) handle;
  const int index = g_httpd_mock.send_calls++;
  if (g_httpd_mock.send_result != ESP_OK) return g_httpd_mock.send_result;
  if (index < 16) {
    const size_t size = std::min(frame->len, sizeof(g_httpd_mock.sent_payloads[index]) - 1U);
    std::memcpy(g_httpd_mock.sent_payloads[index], frame->payload, size);
    g_httpd_mock.sent_fds[index] = socket;
  }
  if (callback != nullptr && g_httpd_mock.defer_send_completions) {
    const size_t pending = g_httpd_mock.pending_send_completions;
    if (pending < 16U) {
      g_httpd_mock.pending_send_callbacks[pending] = callback;
      g_httpd_mock.pending_send_args[pending] = arg;
      g_httpd_mock.pending_send_sockets[pending] = socket;
      g_httpd_mock.pending_send_completions++;
    }
  } else if (callback != nullptr) {
    callback(g_httpd_mock.send_completion_result, socket, arg);
  }
  return ESP_OK;
}

esp_err_t httpd_sess_trigger_close(httpd_handle_t handle, int socket) {
  (void) handle;
  g_httpd_mock.trigger_close_calls++;
  g_httpd_mock.last_closed_fd = socket;
  return ESP_OK;
}

namespace {
struct HttpdMockInitializer {
  HttpdMockInitializer() { httpd_mock_reset(); }
} g_httpd_mock_initializer;
}  // namespace
