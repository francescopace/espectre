/*
 * ESPectre - ESP-IDF Direct WebSocket Service
 *
 * Bounded local WebSocket transport built on esp_http_server.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <deque>
#include <string>
#include <vector>

#include <esp_http_server.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>

#include "direct_websocket_service.h"

namespace espectre {

class EspIdfDirectWebSocketService final : public IDirectWebSocketService {
 public:
  EspIdfDirectWebSocketService();
  ~EspIdfDirectWebSocketService() override;

  bool setup(const DirectWebSocketServiceConfig &config,
             RequestHandler request_handler,
             ClientCountCallback client_count_callback) override;
  bool setup_deferred(const DirectWebSocketServiceConfig &config,
                      DeferredRequestHandler request_handler,
                      ClientCountCallback client_count_callback) override;
  bool complete_deferred_response(uint64_t connection_token, std::string response) override;
  void loop() override;
  void shutdown() override;
  bool running() const override;
  size_t client_count() const override;
  bool publish_event(const std::string &event_name,
                     const std::string &data_json,
                     bool replaceable_telemetry) override;
  DirectWebSocketServiceDiagnostics diagnostics() const override;

 private:
  struct OutboundMessage {
    std::string payload;
    std::string event_name;
    bool replaceable_telemetry{false};
  };

  struct ClientState {
    int fd{-1};
    uint64_t connection_token{0U};
    uint64_t mutation_window_started_us{0U};
    uint16_t mutation_count{0U};
    uint8_t consecutive_send_failures{0U};
    bool send_in_flight{false};
    std::deque<OutboundMessage> outbound;
  };

  struct PendingRequest {
    uint64_t connection_token{0U};
    DirectWebSocketRequest request;
  };

  static esp_err_t websocket_handler_(httpd_req_t *request);
  static esp_err_t websocket_pre_handshake_(httpd_req_t *request);
  static void websocket_send_complete_(esp_err_t result, int fd, void *arg);
  void handle_send_complete_(esp_err_t result, int fd, OutboundMessage message);
  esp_err_t handle_websocket_(httpd_req_t *request);
  bool validate_handshake_(httpd_req_t *request);
  bool header_token_present_(httpd_req_t *request, const char *header, const char *token) const;
  ClientState *find_client_locked_(int fd);
  ClientState *find_client_token_locked_(uint64_t connection_token);
  ClientState *ensure_client_locked_(int fd);
  bool mutation_allowed_locked_(ClientState *client, const std::string &method, uint64_t now_us);
  bool enqueue_locked_(ClientState *client, OutboundMessage message);
  void sync_clients_();
  void send_queued_();
  void notify_client_count_(size_t count);
  bool lock_() const;
  void unlock_() const;

  mutable SemaphoreHandle_t mutex_{nullptr};
  httpd_handle_t server_{nullptr};
  DirectWebSocketServiceConfig config_{};
  RequestHandler request_handler_{};
  DeferredRequestHandler deferred_request_handler_{};
  ClientCountCallback client_count_callback_{};
  std::vector<ClientState> clients_;
  std::deque<PendingRequest> inbound_;
  DirectWebSocketServiceDiagnostics diagnostics_{};
  uint64_t next_connection_token_{1U};
};

}  // namespace espectre
