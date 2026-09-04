/*
 * ESP-IDF HTTP server mock for Direct HTTP host tests.
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef void *httpd_handle_t;
typedef int httpd_method_t;
typedef esp_err_t (*httpd_open_func_t)(httpd_handle_t server, int socket);
typedef bool (*httpd_uri_match_func_t)(const char *template_uri, const char *uri, size_t match_upto);

#define HTTP_GET 0
#define HTTP_POST 1
#define HTTP_OPTIONS 2
#define HTTP_PUT 3
#define HTTP_DELETE 4
#define HTTP_PATCH 5
#define HTTPD_400_BAD_REQUEST "400 Bad Request"
#define HTTPD_401_UNAUTHORIZED "401 Unauthorized"
#define HTTPD_403_FORBIDDEN "403 Forbidden"
#define HTTPD_413_CONTENT_TOO_LARGE "413 Content Too Large"
#define HTTPD_415_UNSUPPORTED_MEDIA_TYPE "415 Unsupported Media Type"
#define HTTPD_429_TOO_MANY_REQUESTS "429 Too Many Requests"
#define HTTPD_503_SERVICE_UNAVAILABLE "503 Service Unavailable"

typedef struct httpd_req {
  void *user_ctx;
  int fd;
  httpd_method_t method;
  const char *uri;
  size_t content_len;
  size_t receive_offset;
  bool async_copy;
} httpd_req_t;

typedef struct {
  uint32_t task_priority;
  uint16_t server_port;
  uint16_t ctrl_port;
  uint16_t max_open_sockets;
  uint16_t max_uri_handlers;
  bool lru_purge_enable;
  uint16_t recv_wait_timeout;
  uint16_t send_wait_timeout;
  int core_id;
  httpd_open_func_t open_fn;
  httpd_uri_match_func_t uri_match_fn;
} httpd_config_t;

#ifdef __cplusplus
#define HTTPD_DEFAULT_CONFIG() \
  (httpd_config_t{5U, 80U, 32768U, 7U, 8U, false, 5U, 5U, -1, nullptr, nullptr})
#else
#define HTTPD_DEFAULT_CONFIG() \
  ((httpd_config_t){5U, 80U, 32768U, 7U, 8U, false, 5U, 5U, -1, NULL, NULL})
#endif

typedef esp_err_t (*httpd_uri_func_t)(httpd_req_t *request);

typedef struct {
  const char *uri;
  httpd_method_t method;
  httpd_uri_func_t handler;
  void *user_ctx;
  bool is_websocket;
  bool handle_ws_control_frames;
  const char *supported_subprotocol;
  httpd_uri_func_t ws_pre_handshake_cb;
} httpd_uri_t;

typedef struct {
  esp_err_t start_result;
  esp_err_t register_result;
  esp_err_t receive_result;
  esp_err_t send_result;
  int start_calls;
  int stop_calls;
  int register_calls;
  int send_calls;
  int chunk_calls;
  int response_error_calls;
  int async_begin_calls;
  int async_complete_calls;
  void (*async_begin_callback)(void *context);
  void *async_begin_callback_context;
  char response_status[48];
  char response_message[192];
  char response_type[96];
  char origin[192];
  char content_type[96];
  char authorization[192];
  char request_private_network[16];
  char allow_origin[192];
  const char *pending_allow_origin;
  char allow_private_network[16];
  char cache_control[64];
  uint8_t incoming_payload[8192];
  size_t incoming_length;
  uint8_t sent_payloads[64][8192];
  size_t sent_lengths[64];
  int sent_fds[64];
  httpd_uri_t registered_uri;
  httpd_uri_t registered_uris[16];
  httpd_config_t last_config;
} httpd_mock_state_t;

extern httpd_mock_state_t g_httpd_mock;

void httpd_mock_reset(void);
void httpd_mock_set_header(const char *name, const char *value);
void httpd_mock_set_incoming(const char *payload);
void httpd_mock_set_incoming_bytes(const void *payload, size_t length);

esp_err_t httpd_start(httpd_handle_t *handle, const httpd_config_t *config);
esp_err_t httpd_stop(httpd_handle_t handle);
esp_err_t httpd_register_uri_handler(httpd_handle_t handle, const httpd_uri_t *uri);
size_t httpd_req_get_hdr_value_len(httpd_req_t *request, const char *name);
esp_err_t httpd_req_get_hdr_value_str(httpd_req_t *request, const char *name, char *buffer, size_t size);
int httpd_req_recv(httpd_req_t *request, char *buffer, size_t size);
esp_err_t httpd_resp_send_err(httpd_req_t *request, const char *status, const char *message);
esp_err_t httpd_resp_set_type(httpd_req_t *request, const char *type);
esp_err_t httpd_resp_set_hdr(httpd_req_t *request, const char *name, const char *value);
esp_err_t httpd_resp_set_status(httpd_req_t *request, const char *status);
esp_err_t httpd_resp_send(httpd_req_t *request, const char *payload, size_t length);
esp_err_t httpd_resp_send_chunk(httpd_req_t *request, const char *payload, size_t length);
esp_err_t httpd_req_async_handler_begin(httpd_req_t *request, httpd_req_t **out);
esp_err_t httpd_req_async_handler_complete(httpd_req_t *request);
int httpd_req_to_sockfd(httpd_req_t *request);
bool httpd_uri_match_wildcard(const char *template_uri, const char *uri, size_t match_upto);

#ifdef __cplusplus
}
#endif
