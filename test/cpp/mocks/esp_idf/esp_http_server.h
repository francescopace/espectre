/*
 * ESP-IDF HTTP server mock for Direct WebSocket host tests.
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

#define HTTP_GET 0
#define HTTPD_400_BAD_REQUEST "400 Bad Request"
#define HTTPD_403_FORBIDDEN "403 Forbidden"

typedef struct httpd_req {
  void *user_ctx;
  int fd;
} httpd_req_t;

typedef struct {
  uint16_t server_port;
  uint16_t max_open_sockets;
  bool lru_purge_enable;
  uint16_t recv_wait_timeout;
  uint16_t send_wait_timeout;
} httpd_config_t;

#ifdef __cplusplus
#define HTTPD_DEFAULT_CONFIG() (httpd_config_t{80U, 7U, false, 5U, 5U})
#else
#define HTTPD_DEFAULT_CONFIG() ((httpd_config_t){80U, 7U, false, 5U, 5U})
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

typedef enum {
  HTTPD_WS_TYPE_CONTINUE = 0x0,
  HTTPD_WS_TYPE_TEXT = 0x1,
  HTTPD_WS_TYPE_BINARY = 0x2,
  HTTPD_WS_TYPE_CLOSE = 0x8,
  HTTPD_WS_TYPE_PING = 0x9,
  HTTPD_WS_TYPE_PONG = 0xa,
} httpd_ws_type_t;

typedef struct {
  bool final;
  bool fragmented;
  httpd_ws_type_t type;
  uint8_t *payload;
  size_t len;
} httpd_ws_frame_t;

typedef enum {
  HTTPD_WS_CLIENT_INVALID = 0,
  HTTPD_WS_CLIENT_HTTP = 1,
  HTTPD_WS_CLIENT_WEBSOCKET = 2,
} httpd_ws_client_info_t;

typedef void (*transfer_complete_cb)(esp_err_t result, int socket, void *arg);

typedef struct {
  esp_err_t start_result;
  esp_err_t register_result;
  esp_err_t client_list_result;
  esp_err_t receive_result;
  esp_err_t send_result;
  esp_err_t send_completion_result;
  bool defer_send_completions;
  int start_calls;
  int stop_calls;
  int register_calls;
  int send_calls;
  int response_error_calls;
  int trigger_close_calls;
  int last_closed_fd;
  char response_status[32];
  char response_message[128];
  char origin[192];
  char subprotocol[192];
  uint8_t incoming_payload[8192];
  size_t incoming_length;
  httpd_ws_type_t incoming_type;
  bool incoming_final;
  bool incoming_fragmented;
  int client_fds[16];
  bool websocket_clients[16];
  size_t client_count;
  char sent_payloads[16][8192];
  int sent_fds[16];
  transfer_complete_cb pending_send_callbacks[16];
  void *pending_send_args[16];
  int pending_send_sockets[16];
  size_t pending_send_completions;
  httpd_uri_t registered_uri;
  httpd_config_t last_config;
} httpd_mock_state_t;

extern httpd_mock_state_t g_httpd_mock;

void httpd_mock_reset(void);
void httpd_mock_set_header(const char *name, const char *value);
void httpd_mock_set_incoming(const char *payload, httpd_ws_type_t type, bool final, bool fragmented);
void httpd_mock_set_clients(const int *fds, size_t count);
void httpd_mock_complete_next_send(esp_err_t result);

esp_err_t httpd_start(httpd_handle_t *handle, const httpd_config_t *config);
esp_err_t httpd_stop(httpd_handle_t handle);
esp_err_t httpd_register_uri_handler(httpd_handle_t handle, const httpd_uri_t *uri);
size_t httpd_req_get_hdr_value_len(httpd_req_t *request, const char *name);
esp_err_t httpd_req_get_hdr_value_str(httpd_req_t *request, const char *name, char *buffer, size_t size);
esp_err_t httpd_resp_send_err(httpd_req_t *request, const char *status, const char *message);
int httpd_req_to_sockfd(httpd_req_t *request);
esp_err_t httpd_ws_recv_frame(httpd_req_t *request, httpd_ws_frame_t *frame, size_t max_len);
esp_err_t httpd_get_client_list(httpd_handle_t handle, size_t *count, int *fds);
httpd_ws_client_info_t httpd_ws_get_fd_info(httpd_handle_t handle, int fd);
esp_err_t httpd_ws_send_data_async(httpd_handle_t handle,
                                   int socket,
                                   httpd_ws_frame_t *frame,
                                   transfer_complete_cb callback,
                                   void *arg);
esp_err_t httpd_sess_trigger_close(httpd_handle_t handle, int socket);

#ifdef __cplusplus
}
#endif
