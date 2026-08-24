/*
 * ESPectre - ESP-IDF Direct WebSocket Service
 *
 * Bounded local WebSocket transport built on esp_http_server.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "direct_websocket_service_esp_idf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <new>
#include <utility>

#include <esp_log.h>
#include <esp_timer.h>

namespace espectre {

namespace {

const char *const TAG = "espectre.direct";
constexpr size_t kHeaderBufferSize = 192U;
constexpr uint64_t kMutationWindowUs = 60ULL * 1000ULL * 1000ULL;

struct AsyncWebSocketPayload {
  EspIdfDirectWebSocketService *service{nullptr};
  std::string payload;
  std::string event_name;
  bool replaceable_telemetry{false};
};

constexpr uint8_t kMaxConsecutiveSendFailures = 3U;

bool read_only_method(const std::string &method) {
  return method == "capabilities" || method == "info" || method == "status" ||
         method == "config" || method == "diagnostics" || method == "ota_status" ||
         method == "discover_peers";
}

bool valid_loopback_port_suffix(const std::string &suffix) {
  if (suffix.empty()) {
    return true;
  }
  if (suffix.size() < 2U || suffix.front() != ':') {
    return false;
  }
  uint32_t port = 0U;
  for (size_t index = 1U; index < suffix.size(); ++index) {
    const unsigned char character = static_cast<unsigned char>(suffix[index]);
    if (!std::isdigit(character)) {
      return false;
    }
    port = port * 10U + static_cast<uint32_t>(character - static_cast<unsigned char>('0'));
    if (port > 65535U) {
      return false;
    }
  }
  return port > 0U;
}

bool http_loopback_origin(const std::string &origin) {
  std::string normalized = origin;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  constexpr const char *kLoopbackPrefixes[] = {
      "http://localhost",
      "http://127.0.0.1",
      "http://[::1]",
  };
  for (const char *prefix : kLoopbackPrefixes) {
    const size_t prefix_length = std::strlen(prefix);
    if (normalized.compare(0U, prefix_length, prefix) == 0 &&
        valid_loopback_port_suffix(normalized.substr(prefix_length))) {
      return true;
    }
  }
  return false;
}

}  // namespace

EspIdfDirectWebSocketService::EspIdfDirectWebSocketService() { mutex_ = xSemaphoreCreateMutex(); }

EspIdfDirectWebSocketService::~EspIdfDirectWebSocketService() {
  shutdown();
  if (mutex_ != nullptr) {
    vSemaphoreDelete(mutex_);
    mutex_ = nullptr;
  }
}

bool EspIdfDirectWebSocketService::setup(const DirectWebSocketServiceConfig &config,
                                         RequestHandler request_handler,
                                         ClientCountCallback client_count_callback) {
  if (mutex_ == nullptr || !request_handler || config.max_clients == 0U || config.outbound_queue_depth == 0U ||
      config.max_clients > 8U) {
    return false;
  }
  shutdown();
  config_ = config;
  request_handler_ = std::move(request_handler);
  deferred_request_handler_ = {};
  client_count_callback_ = std::move(client_count_callback);

  httpd_config_t http_config = HTTPD_DEFAULT_CONFIG();
  http_config.server_port = config_.port;
  http_config.max_open_sockets = static_cast<uint16_t>(config_.max_clients + 3U);
  http_config.lru_purge_enable = true;
  http_config.recv_wait_timeout = 1U;
  http_config.send_wait_timeout = 1U;
  if (httpd_start(&server_, &http_config) != ESP_OK) {
    server_ = nullptr;
    ESP_LOGE(TAG, "Failed to start HTTP server on port %u", static_cast<unsigned>(config_.port));
    return false;
  }

  httpd_uri_t websocket_uri{};
  websocket_uri.uri = ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT;
  websocket_uri.method = HTTP_GET;
  websocket_uri.handler = &EspIdfDirectWebSocketService::websocket_handler_;
  websocket_uri.user_ctx = this;
  websocket_uri.is_websocket = true;
  websocket_uri.handle_ws_control_frames = false;
  websocket_uri.supported_subprotocol = ESPECTRE_DIRECT_WEBSOCKET_SUBPROTOCOL;
  websocket_uri.ws_pre_handshake_cb = &EspIdfDirectWebSocketService::websocket_pre_handshake_;
  if (httpd_register_uri_handler(server_, &websocket_uri) != ESP_OK) {
    httpd_stop(server_);
    server_ = nullptr;
    ESP_LOGE(TAG, "Failed to register Direct WebSocket endpoint");
    return false;
  }
  ESP_LOGI(TAG, "Direct WebSocket listening on port %u", static_cast<unsigned>(config_.port));
  return true;
}

bool EspIdfDirectWebSocketService::setup_deferred(const DirectWebSocketServiceConfig &config,
                                                  DeferredRequestHandler request_handler,
                                                  ClientCountCallback client_count_callback) {
  if (!request_handler) {
    return false;
  }
  const bool started = setup(config, [](const DirectWebSocketRequest &) { return std::string{}; },
                             std::move(client_count_callback));
  if (started) {
    request_handler_ = {};
    deferred_request_handler_ = std::move(request_handler);
  }
  return started;
}

bool EspIdfDirectWebSocketService::complete_deferred_response(uint64_t connection_token,
                                                              std::string response) {
  if (response.empty() || response.size() > ESPECTRE_DIRECT_MAX_FRAME_SIZE || !lock_()) {
    return false;
  }
  ClientState *client = find_client_token_locked_(connection_token);
  const bool queued = client != nullptr &&
                      enqueue_locked_(client, OutboundMessage{std::move(response), {}, false});
  if (!queued && client != nullptr) {
    diagnostics_.send_failures += 1U;
  }
  unlock_();
  return queued;
}

void EspIdfDirectWebSocketService::loop() {
  if (server_ == nullptr) {
    return;
  }
  sync_clients_();

  PendingRequest pending;
  bool have_request = false;
  if (lock_()) {
    if (!inbound_.empty()) {
      pending = std::move(inbound_.front());
      inbound_.pop_front();
      have_request = true;
    }
    unlock_();
  }
  if (have_request && (request_handler_ || deferred_request_handler_)) {
    bool deferred = false;
    std::string response;
    if (deferred_request_handler_) {
      DeferredRequestResult result = deferred_request_handler_(pending.connection_token, pending.request);
      deferred = result.deferred;
      response = std::move(result.response);
    } else {
      response = request_handler_(pending.request);
    }
    if (deferred) {
      send_queued_();
      return;
    }
    if (response.empty()) {
      response = direct_websocket_error_response(pending.request.id, "internal_error", "empty Direct response");
    }
    if (lock_()) {
      ClientState *client = find_client_token_locked_(pending.connection_token);
      if (client != nullptr && !enqueue_locked_(client, OutboundMessage{std::move(response), {}, false})) {
        diagnostics_.send_failures += 1U;
      }
      unlock_();
    }
  }
  send_queued_();
}

void EspIdfDirectWebSocketService::shutdown() {
  httpd_handle_t server = server_;
  server_ = nullptr;
  if (server != nullptr) {
    httpd_stop(server);
  }
  if (lock_()) {
    clients_.clear();
    inbound_.clear();
    diagnostics_.queued_messages = 0U;
    unlock_();
  }
  request_handler_ = {};
  deferred_request_handler_ = {};
  notify_client_count_(0U);
}

bool EspIdfDirectWebSocketService::running() const { return server_ != nullptr; }

size_t EspIdfDirectWebSocketService::client_count() const {
  size_t count = 0U;
  if (lock_()) {
    count = clients_.size();
    unlock_();
  }
  return count;
}

bool EspIdfDirectWebSocketService::publish_event(const std::string &event_name,
                                                 const std::string &data_json,
                                                 bool replaceable_telemetry) {
  if (server_ == nullptr || event_name.empty()) {
    return false;
  }
  const std::string payload = direct_websocket_event(event_name.c_str(), data_json);
  bool accepted = false;
  if (lock_()) {
    for (auto &client : clients_) {
      accepted = enqueue_locked_(&client, OutboundMessage{payload, event_name, replaceable_telemetry}) || accepted;
    }
    unlock_();
  }
  return accepted;
}

DirectWebSocketServiceDiagnostics EspIdfDirectWebSocketService::diagnostics() const {
  DirectWebSocketServiceDiagnostics snapshot;
  if (lock_()) {
    snapshot = diagnostics_;
    snapshot.client_limit = config_.max_clients;
    snapshot.queue_capacity = config_.outbound_queue_depth;
    unlock_();
  }
  return snapshot;
}

esp_err_t EspIdfDirectWebSocketService::websocket_handler_(httpd_req_t *request) {
  if (request == nullptr || request->user_ctx == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  return static_cast<EspIdfDirectWebSocketService *>(request->user_ctx)->handle_websocket_(request);
}

esp_err_t EspIdfDirectWebSocketService::websocket_pre_handshake_(httpd_req_t *request) {
  if (request == nullptr || request->user_ctx == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }
  auto *service = static_cast<EspIdfDirectWebSocketService *>(request->user_ctx);
  if (service->validate_handshake_(request)) {
    return ESP_OK;
  }
  if (service->lock_()) {
    service->diagnostics_.rejected_connections += 1U;
    service->unlock_();
  }
  return ESP_FAIL;
}

void EspIdfDirectWebSocketService::websocket_send_complete_(esp_err_t result, int fd, void *arg) {
  auto *payload = static_cast<AsyncWebSocketPayload *>(arg);
  if (payload == nullptr) {
    return;
  }
  if (payload->service != nullptr) {
    payload->service->handle_send_complete_(
        result,
        fd,
        OutboundMessage{std::move(payload->payload), std::move(payload->event_name), payload->replaceable_telemetry});
  }
  delete payload;
}

void EspIdfDirectWebSocketService::handle_send_complete_(esp_err_t result,
                                                         int fd,
                                                         OutboundMessage message) {
  bool close_client = false;
  if (lock_()) {
    ClientState *client = find_client_locked_(fd);
    if (client != nullptr) {
      client->send_in_flight = false;
      if (result == ESP_OK) {
        client->consecutive_send_failures = 0U;
      } else {
        diagnostics_.send_failures += 1U;
        client->consecutive_send_failures += 1U;
        close_client = client->consecutive_send_failures >= kMaxConsecutiveSendFailures;
        if (close_client) {
          diagnostics_.slow_client_disconnects += 1U;
          client->outbound.clear();
        } else if (client->outbound.size() < config_.outbound_queue_depth) {
          client->outbound.push_front(std::move(message));
        }
      }
    }
    unlock_();
  }
  if (close_client && server_ != nullptr) {
    (void) httpd_sess_trigger_close(server_, fd);
  }
}

esp_err_t EspIdfDirectWebSocketService::handle_websocket_(httpd_req_t *request) {
  httpd_ws_frame_t frame{};
  if (httpd_ws_recv_frame(request, &frame, 0U) != ESP_OK) {
    return ESP_FAIL;
  }
  if (frame.len > ESPECTRE_DIRECT_MAX_FRAME_SIZE) {
    if (lock_()) {
      diagnostics_.oversized_frames += 1U;
      unlock_();
    }
    return ESP_FAIL;
  }
  if (frame.type != HTTPD_WS_TYPE_TEXT || !frame.final || frame.fragmented) {
    if (lock_()) {
      diagnostics_.malformed_frames += 1U;
      unlock_();
    }
    return ESP_FAIL;
  }

  std::string payload(frame.len, '\0');
  frame.payload = reinterpret_cast<uint8_t *>(payload.data());
  if (httpd_ws_recv_frame(request, &frame, payload.size()) != ESP_OK) {
    return ESP_FAIL;
  }
  DirectWebSocketRequest direct_request;
  std::string error;
  if (!parse_direct_websocket_request(payload, &direct_request, &error)) {
    if (lock_()) {
      diagnostics_.malformed_frames += 1U;
      unlock_();
    }
    return ESP_FAIL;
  }

  const int fd = httpd_req_to_sockfd(request);
  bool queued = false;
  size_t previous_client_count = 0U;
  size_t current_client_count = 0U;
  if (lock_()) {
    previous_client_count = clients_.size();
    ClientState *client = ensure_client_locked_(fd);
    current_client_count = clients_.size();
    const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
    const bool mutation_allowed = mutation_allowed_locked_(client, direct_request.method, now_us);
    if (client != nullptr && mutation_allowed && inbound_.size() < config_.outbound_queue_depth) {
      inbound_.push_back(PendingRequest{client->connection_token, std::move(direct_request)});
      queued = true;
    } else if (client != nullptr) {
      diagnostics_.rate_limited_requests += 1U;
      const char *message = mutation_allowed ? "Direct request queue is full" : "Direct mutation rate limit exceeded";
      queued = enqueue_locked_(client,
                               OutboundMessage{direct_websocket_error_response(
                                                   direct_request.id, "rate_limited", message),
                                               {},
                                               false});
    }
    unlock_();
  }
  if (previous_client_count != current_client_count) {
    notify_client_count_(current_client_count);
  }
  return queued ? ESP_OK : ESP_FAIL;
}

bool EspIdfDirectWebSocketService::validate_handshake_(httpd_req_t *request) {
  if (!header_token_present_(request, "Sec-WebSocket-Protocol", ESPECTRE_DIRECT_WEBSOCKET_SUBPROTOCOL)) {
    (void) httpd_resp_send_err(request, HTTPD_400_BAD_REQUEST, "Direct subprotocol required");
    return false;
  }

  const size_t origin_length = httpd_req_get_hdr_value_len(request, "Origin");
  if (origin_length == 0U) {
    if (!config_.allow_missing_origin) {
      (void) httpd_resp_send_err(request, HTTPD_403_FORBIDDEN, "Origin required");
      return false;
    }
  } else {
    if (origin_length >= kHeaderBufferSize) {
      (void) httpd_resp_send_err(request, HTTPD_403_FORBIDDEN, "Origin rejected");
      return false;
    }
    std::array<char, kHeaderBufferSize> origin{};
    if (httpd_req_get_hdr_value_str(request, "Origin", origin.data(), origin.size()) != ESP_OK) {
      (void) httpd_resp_send_err(request, HTTPD_403_FORBIDDEN, "Origin rejected");
      return false;
    }
    const std::string requested_origin = origin.data();
    const bool exact_origin_allowed =
        std::find(config_.allowed_origins.begin(), config_.allowed_origins.end(), requested_origin) !=
        config_.allowed_origins.end();
    if (!exact_origin_allowed &&
        !(config_.allow_http_loopback_origins && http_loopback_origin(requested_origin))) {
      (void) httpd_resp_send_err(request, HTTPD_403_FORBIDDEN, "Origin rejected");
      return false;
    }
  }
  size_t active_websocket_count = client_count();
  std::array<int, 11U> fds{};
  size_t fd_count = std::min(fds.size(), config_.max_clients + 3U);
  if (httpd_get_client_list(server_, &fd_count, fds.data()) == ESP_OK) {
    active_websocket_count = 0U;
    for (size_t index = 0U; index < fd_count; ++index) {
      if (httpd_ws_get_fd_info(server_, fds[index]) == HTTPD_WS_CLIENT_WEBSOCKET) {
        active_websocket_count += 1U;
      }
    }
  }
  if (active_websocket_count >= config_.max_clients) {
    (void) httpd_resp_send_err(request, HTTPD_403_FORBIDDEN, "Direct client limit reached");
    return false;
  }
  return true;
}

bool EspIdfDirectWebSocketService::header_token_present_(httpd_req_t *request,
                                                         const char *header,
                                                         const char *token) const {
  if (request == nullptr || header == nullptr || token == nullptr) {
    return false;
  }
  const size_t length = httpd_req_get_hdr_value_len(request, header);
  if (length == 0U || length >= kHeaderBufferSize) {
    return false;
  }
  std::array<char, kHeaderBufferSize> value{};
  if (httpd_req_get_hdr_value_str(request, header, value.data(), value.size()) != ESP_OK) {
    return false;
  }
  const char *cursor = value.data();
  while (*cursor != '\0') {
    while (*cursor == ',' || std::isspace(static_cast<unsigned char>(*cursor))) {
      ++cursor;
    }
    const char *end = cursor;
    while (*end != '\0' && *end != ',') {
      ++end;
    }
    const char *trimmed_end = end;
    while (trimmed_end > cursor && std::isspace(static_cast<unsigned char>(trimmed_end[-1]))) {
      --trimmed_end;
    }
    if (static_cast<size_t>(trimmed_end - cursor) == std::strlen(token) &&
        std::strncmp(cursor, token, std::strlen(token)) == 0) {
      return true;
    }
    cursor = end;
  }
  return false;
}

EspIdfDirectWebSocketService::ClientState *EspIdfDirectWebSocketService::find_client_locked_(int fd) {
  const auto it = std::find_if(clients_.begin(), clients_.end(), [fd](const ClientState &client) {
    return client.fd == fd;
  });
  return it == clients_.end() ? nullptr : &*it;
}

EspIdfDirectWebSocketService::ClientState *
EspIdfDirectWebSocketService::find_client_token_locked_(uint64_t connection_token) {
  const auto it = std::find_if(clients_.begin(), clients_.end(), [connection_token](const ClientState &client) {
    return client.connection_token == connection_token;
  });
  return it == clients_.end() ? nullptr : &*it;
}

EspIdfDirectWebSocketService::ClientState *EspIdfDirectWebSocketService::ensure_client_locked_(int fd) {
  ClientState *client = find_client_locked_(fd);
  if (client != nullptr) {
    return client;
  }
  if (clients_.size() >= config_.max_clients) {
    return nullptr;
  }
  clients_.push_back(ClientState{});
  clients_.back().fd = fd;
  clients_.back().connection_token = next_connection_token_++;
  if (next_connection_token_ == 0U) {
    next_connection_token_ = 1U;
  }
  diagnostics_.accepted_connections += 1U;
  return &clients_.back();
}

bool EspIdfDirectWebSocketService::mutation_allowed_locked_(ClientState *client,
                                                            const std::string &method,
                                                            uint64_t now_us) {
  if (client == nullptr || read_only_method(method)) {
    return client != nullptr;
  }
  if (client->mutation_window_started_us == 0U || now_us - client->mutation_window_started_us >= kMutationWindowUs) {
    client->mutation_window_started_us = now_us;
    client->mutation_count = 0U;
  }
  if (client->mutation_count >= config_.max_mutations_per_minute) {
    return false;
  }
  client->mutation_count += 1U;
  return true;
}

bool EspIdfDirectWebSocketService::enqueue_locked_(ClientState *client, OutboundMessage message) {
  if (client == nullptr) {
    return false;
  }
  if (message.replaceable_telemetry) {
    const auto replace = std::find_if(client->outbound.rbegin(),
                                      client->outbound.rend(),
                                      [&message](const OutboundMessage &queued) {
                                        return queued.replaceable_telemetry && queued.event_name == message.event_name;
                                      });
    if (replace != client->outbound.rend()) {
      *replace = std::move(message);
      return true;
    }
  }
  if (client->outbound.size() >= config_.outbound_queue_depth) {
    if (message.replaceable_telemetry) {
      diagnostics_.dropped_telemetry_events += 1U;
      return false;
    }
    const auto stale = std::find_if(client->outbound.begin(),
                                    client->outbound.end(),
                                    [](const OutboundMessage &queued) { return queued.replaceable_telemetry; });
    if (stale == client->outbound.end()) {
      return false;
    }
    client->outbound.erase(stale);
    diagnostics_.dropped_telemetry_events += 1U;
  }
  client->outbound.push_back(std::move(message));
  diagnostics_.queued_messages += 1U;
  return true;
}

void EspIdfDirectWebSocketService::sync_clients_() {
  std::array<int, 11U> fds{};
  size_t count = std::min(fds.size(), config_.max_clients + 3U);
  if (httpd_get_client_list(server_, &count, fds.data()) != ESP_OK) {
    return;
  }
  std::vector<int> websocket_fds;
  websocket_fds.reserve(config_.max_clients);
  for (size_t index = 0U; index < count; ++index) {
    if (httpd_ws_get_fd_info(server_, fds[index]) == HTTPD_WS_CLIENT_WEBSOCKET) {
      websocket_fds.push_back(fds[index]);
    }
  }

  size_t previous_count = 0U;
  size_t current_count = 0U;
  if (lock_()) {
    previous_count = clients_.size();
    clients_.erase(std::remove_if(clients_.begin(),
                                  clients_.end(),
                                  [&websocket_fds](const ClientState &client) {
                                    return std::find(websocket_fds.begin(), websocket_fds.end(), client.fd) ==
                                           websocket_fds.end();
                                  }),
                   clients_.end());
    for (const int fd : websocket_fds) {
      (void) ensure_client_locked_(fd);
    }
    current_count = clients_.size();
    diagnostics_.queued_messages = inbound_.size();
    for (const auto &client : clients_) {
      diagnostics_.queued_messages += client.outbound.size();
    }
    unlock_();
  }
  if (previous_count != current_count) {
    notify_client_count_(current_count);
  }
}

void EspIdfDirectWebSocketService::send_queued_() {
  std::vector<std::pair<int, OutboundMessage>> sends;
  if (lock_()) {
    sends.reserve(clients_.size());
    for (auto &client : clients_) {
      if (!client.send_in_flight && !client.outbound.empty()) {
        sends.emplace_back(client.fd, std::move(client.outbound.front()));
        client.outbound.pop_front();
        client.send_in_flight = true;
      }
    }
    unlock_();
  }
  for (auto &send : sends) {
    auto *payload = new (std::nothrow) AsyncWebSocketPayload{
        this,
        std::move(send.second.payload),
        std::move(send.second.event_name),
        send.second.replaceable_telemetry};
    if (payload == nullptr) {
      if (lock_()) {
        diagnostics_.send_failures += 1U;
        ClientState *client = find_client_locked_(send.first);
        if (client != nullptr) {
          client->send_in_flight = false;
          if (client->outbound.size() < config_.outbound_queue_depth) {
            client->outbound.push_front(std::move(send.second));
          }
        }
        unlock_();
      }
      continue;
    }
    httpd_ws_frame_t frame{};
    frame.type = HTTPD_WS_TYPE_TEXT;
    frame.payload = reinterpret_cast<uint8_t *>(payload->payload.data());
    frame.len = payload->payload.size();
    const esp_err_t send_result =
        httpd_ws_send_data_async(server_, send.first, &frame, &websocket_send_complete_, payload);
    if (send_result != ESP_OK) {
      websocket_send_complete_(send_result, send.first, payload);
    }
  }
}

void EspIdfDirectWebSocketService::notify_client_count_(size_t count) {
  if (client_count_callback_) {
    client_count_callback_(count);
  }
}

bool EspIdfDirectWebSocketService::lock_() const {
  return mutex_ != nullptr && xSemaphoreTake(mutex_, pdMS_TO_TICKS(10)) == pdTRUE;
}

void EspIdfDirectWebSocketService::unlock_() const { xSemaphoreGive(mutex_); }

}  // namespace espectre
