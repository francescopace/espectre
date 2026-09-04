/*
 * ESPectre - Mock esp_http_client.h
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <string>

#include "esp_err.h"

enum esp_http_client_event_id_t { HTTP_EVENT_ON_DATA = 0 };

struct esp_http_client_event_t {
  esp_http_client_event_id_t event_id{HTTP_EVENT_ON_DATA};
  void* user_data{nullptr};
  const char* data{nullptr};
  int data_len{0};
};

using esp_http_client_event_cb_t = esp_err_t (*)(esp_http_client_event_t* event);
using esp_http_client_handle_t = void*;

struct esp_http_client_config_t {
  const char* url{nullptr};
  int timeout_ms{0};
  esp_err_t (*crt_bundle_attach)(void*){nullptr};
  int buffer_size{0};
  int buffer_size_tx{0};
  esp_http_client_event_cb_t event_handler{nullptr};
  void* user_data{nullptr};
};

struct esp_http_client_mock_state_t {
  bool init_succeeds{true};
  esp_err_t perform_result{ESP_OK};
  int status_code{200};
  std::string response_body;
  esp_http_client_config_t last_config{};
  int init_calls{0};
  int cleanup_calls{0};
};

inline esp_http_client_mock_state_t g_esp_http_client_mock{};

inline void esp_http_client_mock_reset() {
  g_esp_http_client_mock = esp_http_client_mock_state_t{};
}

inline esp_http_client_handle_t esp_http_client_init(
    const esp_http_client_config_t* config) {
  g_esp_http_client_mock.init_calls++;
  if (config != nullptr) {
    g_esp_http_client_mock.last_config = *config;
  }
  return g_esp_http_client_mock.init_succeeds
             ? static_cast<esp_http_client_handle_t>(&g_esp_http_client_mock)
             : nullptr;
}

inline esp_err_t esp_http_client_perform(esp_http_client_handle_t) {
  if (g_esp_http_client_mock.perform_result != ESP_OK) {
    return g_esp_http_client_mock.perform_result;
  }
  const auto& config = g_esp_http_client_mock.last_config;
  if (config.event_handler != nullptr && !g_esp_http_client_mock.response_body.empty()) {
    esp_http_client_event_t event;
    event.user_data = config.user_data;
    event.data = g_esp_http_client_mock.response_body.data();
    event.data_len = static_cast<int>(g_esp_http_client_mock.response_body.size());
    return config.event_handler(&event);
  }
  return ESP_OK;
}

inline int esp_http_client_get_status_code(esp_http_client_handle_t) {
  return g_esp_http_client_mock.status_code;
}

inline esp_err_t esp_http_client_cleanup(esp_http_client_handle_t) {
  g_esp_http_client_mock.cleanup_calls++;
  return ESP_OK;
}
