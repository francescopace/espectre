/*
 * ESPectre - ESP-IDF Direct HTTP Service
 *
 * Bounded local HTTP, SSE, and binary streaming transport.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "direct_http_service_esp_idf.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <cstring>
#include <limits>
#include <utility>

#include <esp_log.h>
#include <esp_timer.h>
#include <lwip/sockets.h>
#if defined(ESP_PLATFORM)
#include <lwip/tcp.h>
#else
#include <netinet/tcp.h>
#endif

#include "protocol_json.h"
#include "espectre_protocol.h"

namespace espectre {

namespace {

[[maybe_unused]] const char *const TAG = "espectre.http";
constexpr size_t kHeaderBufferSize = 256U;
constexpr uint64_t kMutationWindowUs = 60ULL * 1000ULL * 1000ULL;
constexpr uint64_t kEventHeartbeatUs = 10ULL * 1000ULL * 1000ULL;
constexpr uint64_t kRawBindTimeoutUs = 5ULL * 1000ULL * 1000ULL;
constexpr uint8_t kMaxConsecutiveSendFailures = 3U;
constexpr const char *kHttp400 = "400 Bad Request";
constexpr const char *kHttp401 = "401 Unauthorized";
constexpr const char *kHttp403 = "403 Forbidden";
constexpr const char *kHttp413 = "413 Content Too Large";
constexpr const char *kHttp415 = "415 Unsupported Media Type";
constexpr const char *kHttp429 = "429 Too Many Requests";
constexpr const char *kHttp503 = "503 Service Unavailable";
#if defined(ESP_PLATFORM)
constexpr TickType_t kWorkerShutdownPollTicks = pdMS_TO_TICKS(1U);
constexpr uint32_t kWorkerShutdownTimeoutMs = 1500U;
#endif

bool read_only_method(const std::string &method) {
  return method == "capabilities" || method == "info" || method == "status" ||
         method == "config" || method == "diagnostics" || method == "ota_status" ||
         method == "discover_peers";
}

bool valid_loopback_port_suffix(const std::string &suffix) {
  if (suffix.empty()) return true;
  if (suffix.size() < 2U || suffix.front() != ':') return false;
  uint32_t port = 0U;
  for (size_t index = 1U; index < suffix.size(); ++index) {
    const unsigned char character = static_cast<unsigned char>(suffix[index]);
    if (!std::isdigit(character)) return false;
    port = port * 10U + static_cast<uint32_t>(character - static_cast<unsigned char>('0'));
    if (port > 65535U) return false;
  }
  return port > 0U;
}

bool http_loopback_origin(const std::string &origin) {
  std::string normalized = origin;
  std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char character) {
    return static_cast<char>(std::tolower(character));
  });
  constexpr const char *prefixes[] = {
      "http://localhost", "http://127.0.0.1", "http://[::1]",
  };
  for (const char *prefix : prefixes) {
    const size_t length = std::strlen(prefix);
    if (normalized.compare(0U, length, prefix) == 0 &&
        valid_loopback_port_suffix(normalized.substr(length))) {
      return true;
    }
  }
  return false;
}

bool session_id_present(const uint8_t *session_id) {
  if (session_id == nullptr) return false;
  for (size_t index = 0U; index < ESPECTRE_RAW_CSI_SESSION_ID_BYTES; ++index) {
    if (session_id[index] != 0U) return true;
  }
  return false;
}

std::string session_id_hex(const uint8_t *session_id) {
  static constexpr char kHex[] = "0123456789abcdef";
  std::string out(ESPECTRE_RAW_CSI_SESSION_ID_BYTES * 2U, '0');
  for (size_t index = 0U; index < ESPECTRE_RAW_CSI_SESSION_ID_BYTES; ++index) {
    out[index * 2U] = kHex[(session_id[index] >> 4U) & 0x0fU];
    out[index * 2U + 1U] = kHex[session_id[index] & 0x0fU];
  }
  return out;
}

std::string sse_payload(const std::string &event_name, const std::string &data_json) {
  return "event: " + event_name + "\ndata: " + data_json + "\n\n";
}

esp_err_t send_http_error(httpd_req_t *request, const char *status, const char *message) {
  if (request == nullptr || status == nullptr || message == nullptr) return ESP_ERR_INVALID_ARG;
  (void) httpd_resp_set_status(request, status);
  (void) httpd_resp_set_type(request, "text/plain; charset=utf-8");
  (void) httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  return httpd_resp_send(request, message, std::strlen(message));
}

}  // namespace

EspIdfDirectHttpService::EspIdfDirectHttpService() {
  mutex_ = xSemaphoreCreateMutex();
  raw_send_mutex_ = xSemaphoreCreateMutex();
}

EspIdfDirectHttpService::~EspIdfDirectHttpService() {
  shutdown();
  if (mutex_ != nullptr) {
    vSemaphoreDelete(mutex_);
    mutex_ = nullptr;
  }
  if (raw_send_mutex_ != nullptr) {
    vSemaphoreDelete(raw_send_mutex_);
    raw_send_mutex_ = nullptr;
  }
}

bool EspIdfDirectHttpService::setup(const DirectHttpServiceConfig &config,
                                    RequestHandler request_handler,
                                    ClientCountCallback client_count_callback) {
  if (mutex_ == nullptr || raw_send_mutex_ == nullptr || !request_handler ||
      (config.allowed_origins.empty() && !config.allow_missing_origin) ||
      config.max_event_clients == 0U ||
      config.max_event_clients > 2U || config.max_pending_requests == 0U ||
      config.outbound_queue_depth == 0U) {
    return false;
  }
  shutdown();
  if (lock_()) {
    diagnostics_ = {};
    next_request_token_ = 1U;
    mutation_window_started_us_ = 0U;
    mutation_count_ = 0U;
    pending_event_connections_ = 0U;
    unlock_();
  }
  config_ = config;
  request_handler_ = std::move(request_handler);
  deferred_request_handler_ = {};
  client_count_callback_ = std::move(client_count_callback);

  httpd_config_t http_config = HTTPD_DEFAULT_CONFIG();
  http_config.server_port = config_.port;
  http_config.ctrl_port = static_cast<uint16_t>(http_config.ctrl_port + 1U);
  http_config.max_open_sockets = static_cast<uint16_t>(config_.max_event_clients + 5U);
  http_config.max_uri_handlers = 8U;
  http_config.lru_purge_enable = false;
  http_config.recv_wait_timeout = 1U;
  http_config.send_wait_timeout = 1U;
  if (httpd_start(&server_, &http_config) != ESP_OK) {
    server_ = nullptr;
    ESP_LOGE(TAG, "Failed to start Direct HTTP on port %u", static_cast<unsigned>(config_.port));
    return false;
  }

  const auto register_uri = [this](const char *uri, httpd_method_t method,
                                   esp_err_t (*handler)(httpd_req_t *)) {
    httpd_uri_t descriptor{};
    descriptor.uri = uri;
    descriptor.method = method;
    descriptor.handler = handler;
    descriptor.user_ctx = this;
    return httpd_register_uri_handler(server_, &descriptor) == ESP_OK;
  };
  const bool registered =
      register_uri(ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT, HTTP_POST, &request_uri_handler_) &&
      register_uri(ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT, HTTP_GET, &events_handler_) &&
      register_uri(ESPECTRE_RAW_CSI_ENDPOINT, HTTP_GET, &raw_handler_) &&
      register_uri(ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT, HTTP_OPTIONS, &options_handler_) &&
      register_uri(ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT, HTTP_OPTIONS, &options_handler_) &&
      register_uri(ESPECTRE_RAW_CSI_ENDPOINT, HTTP_OPTIONS, &options_handler_);
  if (!registered) {
    httpd_stop(server_);
    server_ = nullptr;
    ESP_LOGE(TAG, "Failed to register Direct HTTP endpoints");
    return false;
  }

  worker_running_.store(true, std::memory_order_release);
#if defined(ESP_PLATFORM)
  if (xTaskCreate(&worker_entry_, "espectre_http", 4096U, this,
                  tskIDLE_PRIORITY + 2U, &worker_task_) != pdPASS) {
    worker_running_.store(false, std::memory_order_release);
    httpd_stop(server_);
    server_ = nullptr;
    ESP_LOGE(TAG, "Failed to start Direct HTTP streaming worker");
    return false;
  }
  raw_worker_running_.store(true, std::memory_order_release);
  if (xTaskCreate(&raw_worker_entry_, "espectre_raw", 4096U, this,
                  tskIDLE_PRIORITY + 3U, &raw_worker_task_) != pdPASS) {
    raw_worker_running_.store(false, std::memory_order_release);
    worker_running_.store(false, std::memory_order_release);
    vTaskDelete(worker_task_);
    worker_task_ = nullptr;
    httpd_stop(server_);
    server_ = nullptr;
    ESP_LOGE(TAG, "Failed to start Direct raw CSI worker");
    return false;
  }
#endif
  ESP_LOGI(TAG, "Direct HTTP listening on port %u", static_cast<unsigned>(config_.port));
  return true;
}

bool EspIdfDirectHttpService::setup_deferred(const DirectHttpServiceConfig &config,
                                             DeferredRequestHandler request_handler,
                                             ClientCountCallback client_count_callback) {
  if (!request_handler) return false;
  const bool started = setup(config, [](const DirectRequest &) { return std::string{}; },
                             std::move(client_count_callback));
  if (started) {
    request_handler_ = {};
    deferred_request_handler_ = std::move(request_handler);
  }
  return started;
}

bool EspIdfDirectHttpService::complete_deferred_response(uint64_t request_token,
                                                         std::string response) {
  if (response.empty() || response.size() > ESPECTRE_DIRECT_MAX_RESPONSE_SIZE || !lock_()) {
    return false;
  }
  PendingRequest *stored = find_deferred_locked_(request_token);
  if (stored == nullptr) {
    unlock_();
    return false;
  }
  PendingRequest pending = std::move(*stored);
  deferred_.erase(std::remove_if(deferred_.begin(), deferred_.end(),
                                 [request_token](const PendingRequest &item) {
                                   return item.token == request_token;
                                 }),
                  deferred_.end());
  unlock_();
  return finish_request_(std::move(pending), response);
}

void EspIdfDirectHttpService::loop() {
  if (server_ == nullptr) return;
  dispatch_pending_callbacks_();
  service_raw_timeouts_();

  PendingRequest pending;
  bool have_request = false;
  if (lock_()) {
    if (!inbound_.empty()) {
      pending = std::move(inbound_.front());
      inbound_.pop_front();
      have_request = true;
    }
    diagnostics_.queued_messages = inbound_.size() + deferred_.size();
    for (const EventClient &client : event_clients_) diagnostics_.queued_messages += client.outbound.size();
    unlock_();
  }
  if (have_request) {
    bool deferred = false;
    std::string response;
    if (deferred_request_handler_) {
      DeferredRequestResult result = deferred_request_handler_(pending.token, pending.direct);
      deferred = result.deferred;
      response = std::move(result.response);
    } else if (request_handler_) {
      response = request_handler_(pending.direct);
    }
    if (deferred) {
      if (lock_()) {
        deferred_.push_back(std::move(pending));
        unlock_();
      }
    } else {
      if (response.empty()) {
        EspectreDeviceConfig device;
        device.device_id = config_.device_id;
        EspectreCommand command;
        command.command_id = pending.direct.command_id;
        command.command = pending.direct.command;
        response = espectre_command_result_payload(
            device, command, false, "internal_error", "empty Direct response");
      } else if (response.size() > ESPECTRE_DIRECT_MAX_RESPONSE_SIZE) {
        EspectreDeviceConfig device;
        device.device_id = config_.device_id;
        EspectreCommand command;
        command.command_id = pending.direct.command_id;
        command.command = pending.direct.command;
        response = espectre_command_result_payload(
            device, command, false, "internal_error", "Direct response exceeds the size limit");
      }
      (void) finish_request_(std::move(pending), response);
    }
  }
#if !defined(ESP_PLATFORM)
  worker_loop_();
#endif
  dispatch_pending_callbacks_();
}

void EspIdfDirectHttpService::shutdown() {
  worker_running_.store(false, std::memory_order_release);
  raw_worker_running_.store(false, std::memory_order_release);
#if defined(ESP_PLATFORM)
  if (raw_worker_task_ != nullptr) xTaskNotifyGive(raw_worker_task_);
#endif
  (void) stop_raw_session(RawCsiStopReason::SHUTDOWN);
#if defined(ESP_PLATFORM)
  uint32_t waited_ms = 0U;
  while ((worker_task_ != nullptr || raw_worker_task_ != nullptr) &&
         waited_ms < kWorkerShutdownTimeoutMs) {
    vTaskDelay(kWorkerShutdownPollTicks);
    waited_ms += 1U;
  }
  if (worker_task_ != nullptr) {
    ESP_LOGW(TAG, "Direct HTTP worker did not stop within %u ms",
             static_cast<unsigned>(kWorkerShutdownTimeoutMs));
    vTaskDelete(worker_task_);
    worker_task_ = nullptr;
  }
  if (raw_worker_task_ != nullptr) {
    ESP_LOGW(TAG, "Direct raw worker did not stop within %u ms",
             static_cast<unsigned>(kWorkerShutdownTimeoutMs));
    vTaskDelete(raw_worker_task_);
    raw_worker_task_ = nullptr;
  }
#endif

  std::vector<httpd_req_t *> requests;
  if (lock_()) {
    for (EventClient &client : event_clients_) requests.push_back(client.request);
    for (PendingRequest &pending : inbound_) requests.push_back(pending.request);
    for (PendingRequest &pending : deferred_) requests.push_back(pending.request);
    event_clients_.clear();
    pending_event_connections_ = 0U;
    inbound_.clear();
    deferred_.clear();
    diagnostics_.queued_messages = 0U;
    unlock_();
  }
  for (httpd_req_t *request : requests) {
    if (request != nullptr) {
      (void) httpd_resp_send_chunk(request, nullptr, 0U);
      (void) httpd_req_async_handler_complete(request);
    }
  }

  httpd_handle_t server = server_;
  server_ = nullptr;
  if (server != nullptr) (void) httpd_stop(server);
  request_handler_ = {};
  deferred_request_handler_ = {};
  notify_client_count_(0U);
  dispatch_pending_callbacks_();
}

bool EspIdfDirectHttpService::running() const { return server_ != nullptr; }

size_t EspIdfDirectHttpService::event_client_count() const {
  size_t count = 0U;
  if (lock_()) {
    count = event_clients_.size();
    unlock_();
  }
  return count;
}

bool EspIdfDirectHttpService::publish_event(const std::string &event_name,
                                            const std::string &data_json,
                                            bool replaceable_telemetry) {
  if (server_ == nullptr || event_name.empty()) return false;
  std::vector<JsonObjectField> fields;
  if (!parse_json_object_fields(data_json, &fields, nullptr) ||
      data_json.size() > ESPECTRE_DIRECT_MAX_RESPONSE_SIZE) {
    return false;
  }
  const OutboundEvent event{sse_payload(event_name, data_json), event_name, replaceable_telemetry};
  bool accepted = false;
  if (lock_()) {
    for (EventClient &client : event_clients_) {
      accepted = enqueue_event_locked_(&client, event) || accepted;
    }
    unlock_();
  }
  return accepted;
}

DirectHttpServiceDiagnostics EspIdfDirectHttpService::diagnostics() const {
  DirectHttpServiceDiagnostics snapshot;
  if (lock_()) {
    snapshot = diagnostics_;
    snapshot.event_client_limit = config_.max_event_clients;
    snapshot.queue_capacity = config_.outbound_queue_depth;
    unlock_();
  }
  return snapshot;
}

bool EspIdfDirectHttpService::start_raw_session(
    const RawCsiSessionConfig &config,
    RawSessionStoppedCallback stopped_callback) {
  if (server_ == nullptr || !session_id_present(config.session_id) || !lock_()) {
    return false;
  }
  if (raw_session_active_.load(std::memory_order_acquire)) {
    unlock_();
    return false;
  }
  if (pending_raw_stopped_callback_) {
    unlock_();
    return false;
  }
  raw_session_ = {};
  raw_session_.config = config;
  raw_session_.stopped_callback = std::move(stopped_callback);
  raw_session_.generation = next_raw_session_generation_++;
  if (next_raw_session_generation_ == 0U) next_raw_session_generation_ = 1U;
  raw_session_.opened_at_us = static_cast<uint64_t>(esp_timer_get_time());
  raw_sample_head_.store(0U, std::memory_order_relaxed);
  raw_sample_tail_.store(0U, std::memory_order_relaxed);
  raw_offer_sequence_.store(0U, std::memory_order_relaxed);
  raw_drop_total_.store(0U, std::memory_order_relaxed);
  raw_send_backpressure_total_.store(0U, std::memory_order_relaxed);
  raw_fresh_record_total_.store(0U, std::memory_order_relaxed);
  raw_session_active_.store(true, std::memory_order_release);
  unlock_();
  return true;
}

bool EspIdfDirectHttpService::stop_raw_session(RawCsiStopReason reason) {
  bool expected_active = true;
  if (!raw_session_active_.compare_exchange_strong(
          expected_active, false, std::memory_order_acq_rel)) return false;
  while (raw_producer_active_.load(std::memory_order_acquire) != 0U) {
    vTaskDelay(1U);
  }
  if (xSemaphoreTake(raw_send_mutex_, portMAX_DELAY) != pdTRUE) return false;
  httpd_req_t *request = nullptr;
  if (xSemaphoreTake(mutex_, portMAX_DELAY) != pdTRUE) {
    xSemaphoreGive(raw_send_mutex_);
    return false;
  }
  const uint64_t head = raw_sample_head_.load(std::memory_order_acquire);
  const uint64_t tail = raw_sample_tail_.load(std::memory_order_acquire);
  if (tail > head) {
    raw_drop_total_.fetch_add(tail - head, std::memory_order_relaxed);
  }
  request = raw_session_.request;
  pending_raw_stopped_callback_ = std::move(raw_session_.stopped_callback);
  pending_raw_stop_reason_ = reason;
  reset_raw_session_locked_();
  unlock_();
  xSemaphoreGive(raw_send_mutex_);
  if (request != nullptr) {
    (void) httpd_resp_send_chunk(request, nullptr, 0U);
    (void) httpd_req_async_handler_complete(request);
  }
  return true;
}

bool EspIdfDirectHttpService::offer_raw_packet(const RawCsiPacketView &packet) {
  if (!raw_session_active_.load(std::memory_order_acquire)) return false;
  raw_producer_active_.fetch_add(1U, std::memory_order_acq_rel);
  if (!raw_session_active_.load(std::memory_order_acquire)) {
    raw_producer_active_.fetch_sub(1U, std::memory_order_release);
    return false;
  }
  const uint64_t sequence = raw_offer_sequence_.fetch_add(1U, std::memory_order_relaxed) + 1U;
  if (packet.csi == nullptr || packet.csi_len == 0U ||
      packet.csi_len > RAW_CSI_MAX_PAYLOAD_BYTES || (packet.csi_len & 1U) != 0U) {
    raw_drop_total_.fetch_add(1U, std::memory_order_relaxed);
    raw_producer_active_.fetch_sub(1U, std::memory_order_release);
    return false;
  }
  const uint64_t tail = raw_sample_tail_.load(std::memory_order_relaxed);
  const uint64_t head = raw_sample_head_.load(std::memory_order_acquire);
  if (tail - head >= raw_samples_.size()) {
    raw_drop_total_.fetch_add(1U, std::memory_order_relaxed);
    raw_producer_active_.fetch_sub(1U, std::memory_order_release);
    return false;
  }
  RawSampleSlot &slot = raw_samples_[tail % raw_samples_.size()];
  slot.metadata = packet;
  std::memcpy(slot.csi.data(), packet.csi, packet.csi_len);
  slot.metadata.csi = slot.csi.data();
  slot.stream_sequence = sequence;
  raw_sample_tail_.store(tail + 1U, std::memory_order_release);
  raw_producer_active_.fetch_sub(1U, std::memory_order_release);
#if defined(ESP_PLATFORM)
  if (raw_worker_task_ != nullptr) xTaskNotifyGive(raw_worker_task_);
#endif
  return true;
}

RawCsiSessionDiagnostics EspIdfDirectHttpService::raw_diagnostics() const {
  RawCsiSessionDiagnostics snapshot;
  snapshot.active = raw_session_active_.load(std::memory_order_acquire);
  snapshot.raw_drop_total = raw_drop_total_.load(std::memory_order_relaxed);
  snapshot.raw_send_backpressure_total = raw_send_backpressure_total_.load(std::memory_order_relaxed);
  snapshot.fresh_record_total = raw_fresh_record_total_.load(std::memory_order_relaxed);
  if (lock_()) {
    snapshot.binary_bound = raw_session_.binary_bound;
    snapshot.stream_sequence = raw_offer_sequence_.load(std::memory_order_relaxed);
    unlock_();
  }
  return snapshot;
}

esp_err_t EspIdfDirectHttpService::request_uri_handler_(httpd_req_t *request) {
  if (request == nullptr || request->user_ctx == nullptr) return ESP_ERR_INVALID_ARG;
  return static_cast<EspIdfDirectHttpService *>(request->user_ctx)->handle_request_(request);
}

esp_err_t EspIdfDirectHttpService::events_handler_(httpd_req_t *request) {
  if (request == nullptr || request->user_ctx == nullptr) return ESP_ERR_INVALID_ARG;
  return static_cast<EspIdfDirectHttpService *>(request->user_ctx)->handle_events_(request);
}

esp_err_t EspIdfDirectHttpService::raw_handler_(httpd_req_t *request) {
  if (request == nullptr || request->user_ctx == nullptr) return ESP_ERR_INVALID_ARG;
  return static_cast<EspIdfDirectHttpService *>(request->user_ctx)->handle_raw_(request);
}

esp_err_t EspIdfDirectHttpService::options_handler_(httpd_req_t *request) {
  if (request == nullptr || request->user_ctx == nullptr) return ESP_ERR_INVALID_ARG;
  return static_cast<EspIdfDirectHttpService *>(request->user_ctx)->handle_options_(request);
}

void EspIdfDirectHttpService::worker_entry_(void *context) {
  auto *service = static_cast<EspIdfDirectHttpService *>(context);
  while (service != nullptr && service->worker_running_.load(std::memory_order_acquire)) {
    service->worker_loop_();
    vTaskDelay(pdMS_TO_TICKS(1));
  }
  if (service != nullptr) service->worker_task_ = nullptr;
  vTaskDelete(nullptr);
}

void EspIdfDirectHttpService::raw_worker_entry_(void *context) {
#if defined(ESP_PLATFORM)
  auto *service = static_cast<EspIdfDirectHttpService *>(context);
  while (service != nullptr && service->raw_worker_running_.load(std::memory_order_acquire)) {
    (void) ulTaskNotifyTake(pdTRUE, portMAX_DELAY);
    while (service->raw_worker_running_.load(std::memory_order_acquire) &&
           service->service_raw_stream_()) {
    }
  }
  if (service != nullptr) service->raw_worker_task_ = nullptr;
  vTaskDelete(nullptr);
#else
  (void) context;
#endif
}

esp_err_t EspIdfDirectHttpService::handle_request_(httpd_req_t *request) {
  std::string origin;
  if (!validate_origin_(request, &origin)) return ESP_FAIL;
  std::string content_type;
  if (!read_header_(request, "Content-Type", &content_type) ||
      content_type.compare(0U, std::strlen("application/json"), "application/json") != 0) {
    (void) send_error_(request, kHttp415, "application/json required", origin);
    return ESP_FAIL;
  }
  if (request->content_len == 0U || request->content_len > ESPECTRE_DIRECT_MAX_REQUEST_SIZE) {
    if (lock_()) {
      diagnostics_.oversized_requests += request->content_len > ESPECTRE_DIRECT_MAX_REQUEST_SIZE ? 1U : 0U;
      unlock_();
    }
    (void) send_error_(request,
                       request->content_len > ESPECTRE_DIRECT_MAX_REQUEST_SIZE
                           ? kHttp413 : kHttp400,
                       "invalid Direct request size", origin);
    return ESP_FAIL;
  }
  std::string payload(request->content_len, '\0');
  size_t received = 0U;
  while (received < payload.size()) {
    const int result = httpd_req_recv(request, payload.data() + received, payload.size() - received);
    if (result <= 0) {
      (void) send_error_(request, kHttp400, "incomplete Direct request", origin);
      return ESP_FAIL;
    }
    received += static_cast<size_t>(result);
  }
  DirectRequest direct;
  std::string error;
  if (!parse_direct_http_request(payload, &direct, &error)) {
    if (lock_()) {
      diagnostics_.malformed_requests += 1U;
      unlock_();
    }
    (void) send_error_(request, kHttp400, error.c_str(), origin);
    return ESP_FAIL;
  }
  (void) read_bearer_(request, &direct.authorization);

  bool accepted = false;
  bool rate_limited = false;
  bool queue_full = false;
  uint64_t token = 0U;
  if (lock_()) {
    const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
    const bool mutation_allowed = mutation_allowed_locked_(direct.command, now_us);
    queue_full = inbound_.size() + deferred_.size() >= config_.max_pending_requests;
    if (mutation_allowed && !queue_full) {
      token = next_request_token_++;
      if (next_request_token_ == 0U) next_request_token_ = 1U;
      accepted = true;
    } else {
      rate_limited = !mutation_allowed;
      if (rate_limited) diagnostics_.rate_limited_requests += 1U;
    }
    unlock_();
  }
  if (!accepted) {
    (void) send_error_(request, rate_limited ? kHttp429 : kHttp503,
                       queue_full ? "Direct request queue is full"
                                  : "Direct request rate limit reached", origin);
    return ESP_FAIL;
  }
  httpd_req_t *async_request = nullptr;
  if (httpd_req_async_handler_begin(request, &async_request) != ESP_OK || async_request == nullptr) {
    (void) send_error_(request, kHttp503, "Direct request unavailable", origin);
    return ESP_FAIL;
  }
  if (!lock_()) {
    (void) send_error_(async_request, kHttp503, "Direct request unavailable", origin);
    (void) httpd_req_async_handler_complete(async_request);
    return ESP_FAIL;
  }
  inbound_.push_back(PendingRequest{token, async_request, std::move(direct), std::move(origin)});
  unlock_();
  return ESP_OK;
}

esp_err_t EspIdfDirectHttpService::handle_events_(httpd_req_t *request) {
  std::string origin;
  if (!validate_origin_(request, &origin)) return ESP_FAIL;
  if (!lock_()) return ESP_FAIL;
  const bool available = event_clients_.size() + pending_event_connections_ <
                         config_.max_event_clients;
  if (available) {
    pending_event_connections_ += 1U;
  } else {
    diagnostics_.rejected_connections += 1U;
  }
  unlock_();
  if (!available) {
    (void) send_error_(request, kHttp503, "Direct event client limit reached", origin);
    return ESP_FAIL;
  }
  (void) httpd_resp_set_type(request, "text/event-stream; charset=utf-8");
  (void) httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  (void) httpd_resp_set_hdr(request, "Connection", "keep-alive");
  httpd_req_t *async_request = nullptr;
  if (httpd_req_async_handler_begin(request, &async_request) != ESP_OK || async_request == nullptr) {
    if (lock_()) {
      if (pending_event_connections_ > 0U) pending_event_connections_ -= 1U;
      diagnostics_.rejected_connections += 1U;
      unlock_();
    }
    (void) send_error_(request, kHttp503, "Direct event stream unavailable", origin);
    return ESP_FAIL;
  }
  set_response_headers_(async_request, origin);
  static constexpr char kConnected[] = "retry: 3000\n: connected\n\n";
  if (httpd_resp_send_chunk(async_request, kConnected, sizeof(kConnected) - 1U) != ESP_OK) {
    if (lock_()) {
      if (pending_event_connections_ > 0U) pending_event_connections_ -= 1U;
      diagnostics_.rejected_connections += 1U;
      diagnostics_.send_failures += 1U;
      unlock_();
    }
    (void) httpd_req_async_handler_complete(async_request);
    return ESP_FAIL;
  }
  const int fd = httpd_req_to_sockfd(async_request);
  size_t count = 0U;
  if (lock_()) {
    if (pending_event_connections_ > 0U) pending_event_connections_ -= 1U;
    event_clients_.push_back(EventClient{async_request, fd, 0U,
                                         static_cast<uint64_t>(esp_timer_get_time()), {}});
    diagnostics_.accepted_connections += 1U;
    count = event_clients_.size();
    unlock_();
  }
  notify_client_count_(count);
  return ESP_OK;
}

esp_err_t EspIdfDirectHttpService::handle_raw_(httpd_req_t *request) {
  std::string origin;
  if (!validate_origin_(request, &origin)) return ESP_FAIL;
  std::string bearer;
  if (!read_bearer_(request, &bearer)) {
    (void) send_error_(request, kHttp401, "raw CSI bearer required", origin);
    return ESP_FAIL;
  }
  bool accepted = false;
  uint64_t session_generation = 0U;
  if (lock_()) {
    accepted = raw_session_active_.load(std::memory_order_acquire) &&
               !raw_session_.binary_bound &&
               bearer == session_id_hex(raw_session_.config.session_id);
    if (accepted) session_generation = raw_session_.generation;
    unlock_();
  }
  if (!accepted) {
    (void) send_error_(request, kHttp403, "raw CSI session unavailable", origin);
    return ESP_FAIL;
  }
  (void) httpd_resp_set_type(request, "application/octet-stream");
  (void) httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  httpd_req_t *async_request = nullptr;
  if (httpd_req_async_handler_begin(request, &async_request) != ESP_OK || async_request == nullptr) {
    (void) send_error_(request, kHttp503, "raw CSI stream unavailable", origin);
    return ESP_FAIL;
  }
  if (!lock_()) {
    (void) httpd_req_async_handler_complete(async_request);
    return ESP_FAIL;
  }
  const bool session_still_available =
      raw_session_active_.load(std::memory_order_acquire) &&
      !raw_session_.binary_bound &&
      raw_session_.generation == session_generation &&
      bearer == session_id_hex(raw_session_.config.session_id);
  if (!session_still_available) {
    unlock_();
    (void) send_error_(async_request, kHttp403, "raw CSI session unavailable", origin);
    (void) httpd_req_async_handler_complete(async_request);
    return ESP_FAIL;
  }
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  raw_session_.request = async_request;
  raw_session_.fd = httpd_req_to_sockfd(async_request);
  raw_session_.binary_bound = true;
  raw_session_.last_send_us = now_us;
  raw_session_.origin = std::move(origin);
  set_response_headers_(async_request, raw_session_.origin);
  const int keepalive = 1;
  (void) setsockopt(raw_session_.fd, SOL_SOCKET, SO_KEEPALIVE, &keepalive, sizeof(keepalive));
#if defined(TCP_KEEPIDLE)
  const int keepidle = 10;
  (void) setsockopt(raw_session_.fd, IPPROTO_TCP, TCP_KEEPIDLE, &keepidle, sizeof(keepidle));
#endif
#if defined(TCP_KEEPINTVL)
  const int keepinterval = 5;
  (void) setsockopt(raw_session_.fd, IPPROTO_TCP, TCP_KEEPINTVL, &keepinterval, sizeof(keepinterval));
#endif
#if defined(TCP_KEEPCNT)
  const int keepcount = 3;
  (void) setsockopt(raw_session_.fd, IPPROTO_TCP, TCP_KEEPCNT, &keepcount, sizeof(keepcount));
#endif
  unlock_();
#if defined(ESP_PLATFORM)
  if (raw_worker_task_ != nullptr) xTaskNotifyGive(raw_worker_task_);
#endif
  return ESP_OK;
}

esp_err_t EspIdfDirectHttpService::handle_options_(httpd_req_t *request) {
  std::string origin;
  if (!validate_origin_(request, &origin)) return ESP_FAIL;
  set_response_headers_(request, origin);
  (void) httpd_resp_set_hdr(request, "Access-Control-Allow-Methods", "GET, POST, OPTIONS");
  (void) httpd_resp_set_hdr(request, "Access-Control-Allow-Headers", "Authorization, Content-Type");
  (void) httpd_resp_set_hdr(request, "Access-Control-Max-Age", "600");
  (void) httpd_resp_set_hdr(request, "Cache-Control", "no-store");
  (void) httpd_resp_set_status(request, "204 No Content");
  return httpd_resp_send(request, nullptr, 0U);
}

bool EspIdfDirectHttpService::validate_origin_(httpd_req_t *request, std::string *origin) {
  if (origin == nullptr) return false;
  origin->clear();
  if (!read_header_(request, "Origin", origin)) {
    if (!config_.allow_missing_origin) {
      (void) send_http_error(request, kHttp403, "Origin required");
      return false;
    }
    return true;
  }
  const bool exact = std::find(config_.allowed_origins.begin(), config_.allowed_origins.end(), *origin) !=
                     config_.allowed_origins.end();
  if (!exact && !(config_.allow_http_loopback_origins && http_loopback_origin(*origin))) {
    (void) send_http_error(request, kHttp403, "Origin rejected");
    return false;
  }
  return true;
}

void EspIdfDirectHttpService::set_response_headers_(httpd_req_t *request,
                                                     const std::string &origin) const {
  if (!origin.empty()) {
    (void) httpd_resp_set_hdr(request, "Access-Control-Allow-Origin", origin.c_str());
  }
  (void) httpd_resp_set_hdr(request, "Vary", "Origin");
  std::string private_network;
  if (read_header_(request, "Access-Control-Request-Private-Network", &private_network) &&
      private_network == "true") {
    (void) httpd_resp_set_hdr(request, "Access-Control-Allow-Private-Network", "true");
  }
}

esp_err_t EspIdfDirectHttpService::send_error_(httpd_req_t *request,
                                                const char *status,
                                                const char *message,
                                                const std::string &origin) const {
  set_response_headers_(request, origin);
  return send_http_error(request, status, message);
}

bool EspIdfDirectHttpService::read_header_(httpd_req_t *request,
                                           const char *name,
                                           std::string *value) const {
  if (request == nullptr || name == nullptr || value == nullptr) return false;
  const size_t length = httpd_req_get_hdr_value_len(request, name);
  if (length == 0U || length >= kHeaderBufferSize) return false;
  std::array<char, kHeaderBufferSize> buffer{};
  if (httpd_req_get_hdr_value_str(request, name, buffer.data(), buffer.size()) != ESP_OK) return false;
  value->assign(buffer.data(), length);
  return true;
}

bool EspIdfDirectHttpService::read_bearer_(httpd_req_t *request, std::string *value) const {
  std::string authorization;
  if (!read_header_(request, "Authorization", &authorization)) return false;
  static constexpr char kPrefix[] = "Bearer ";
  if (authorization.compare(0U, sizeof(kPrefix) - 1U, kPrefix) != 0 ||
      authorization.size() != sizeof(kPrefix) - 1U + ESPECTRE_RAW_CSI_SESSION_ID_BYTES * 2U) {
    return false;
  }
  std::string token = authorization.substr(sizeof(kPrefix) - 1U);
  for (char &character : token) {
    const unsigned char value_char = static_cast<unsigned char>(character);
    if (!std::isxdigit(value_char)) return false;
    character = static_cast<char>(std::tolower(value_char));
  }
  *value = std::move(token);
  return true;
}

bool EspIdfDirectHttpService::mutation_allowed_locked_(const std::string &method, uint64_t now_us) {
  if (read_only_method(method)) return true;
  if (mutation_window_started_us_ == 0U || now_us - mutation_window_started_us_ >= kMutationWindowUs) {
    mutation_window_started_us_ = now_us;
    mutation_count_ = 0U;
  }
  if (mutation_count_ >= config_.max_mutations_per_minute) return false;
  mutation_count_ += 1U;
  return true;
}

bool EspIdfDirectHttpService::enqueue_event_locked_(EventClient *client, OutboundEvent event) {
  if (client == nullptr) return false;
  if (event.replaceable_telemetry) {
    const auto replace = std::find_if(client->outbound.rbegin(), client->outbound.rend(),
                                      [&event](const OutboundEvent &queued) {
                                        return queued.replaceable_telemetry && queued.event_name == event.event_name;
                                      });
    if (replace != client->outbound.rend()) {
      *replace = std::move(event);
      return true;
    }
  }
  if (client->outbound.size() >= config_.outbound_queue_depth) {
    if (event.replaceable_telemetry) {
      diagnostics_.dropped_telemetry_events += 1U;
      return false;
    }
    const auto stale = std::find_if(client->outbound.begin(), client->outbound.end(),
                                    [](const OutboundEvent &queued) {
                                      return queued.replaceable_telemetry;
                                    });
    if (stale == client->outbound.end()) return false;
    client->outbound.erase(stale);
    diagnostics_.dropped_telemetry_events += 1U;
  }
  client->outbound.push_back(std::move(event));
  return true;
}

EspIdfDirectHttpService::PendingRequest *EspIdfDirectHttpService::find_deferred_locked_(uint64_t token) {
  const auto found = std::find_if(deferred_.begin(), deferred_.end(),
                                  [token](const PendingRequest &request) { return request.token == token; });
  return found == deferred_.end() ? nullptr : &*found;
}

bool EspIdfDirectHttpService::finish_request_(PendingRequest request, const std::string &response) {
  if (request.request == nullptr) return false;
  set_response_headers_(request.request, request.origin);
  (void) httpd_resp_set_type(request.request, "application/json; charset=utf-8");
  (void) httpd_resp_set_hdr(request.request, "Cache-Control", "no-store");
  const esp_err_t result = httpd_resp_send(request.request, response.data(), response.size());
  (void) httpd_req_async_handler_complete(request.request);
  if (result != ESP_OK && lock_()) {
    diagnostics_.send_failures += 1U;
    unlock_();
  }
  return result == ESP_OK;
}

void EspIdfDirectHttpService::service_event_streams_() {
  struct Send {
    int fd{-1};
    httpd_req_t *request{nullptr};
    std::string payload;
  };
  std::vector<Send> sends;
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (lock_()) {
    sends.reserve(event_clients_.size());
    for (EventClient &client : event_clients_) {
      if (!client.outbound.empty()) {
        sends.push_back(Send{client.fd, client.request, std::move(client.outbound.front().payload)});
        client.outbound.pop_front();
      } else if (now_us - client.last_send_us >= kEventHeartbeatUs) {
        sends.push_back(Send{client.fd, client.request, ": ping\n\n"});
      }
    }
    unlock_();
  }
  size_t previous_count = event_client_count();
  for (const Send &send : sends) {
    const esp_err_t result = httpd_resp_send_chunk(send.request, send.payload.data(), send.payload.size());
    if (lock_()) {
      const auto client = std::find_if(event_clients_.begin(), event_clients_.end(),
                                       [&send](const EventClient &candidate) {
                                         return candidate.fd == send.fd;
                                       });
      if (client != event_clients_.end()) {
        if (result == ESP_OK) {
          client->last_send_us = now_us;
          client->consecutive_send_failures = 0U;
        } else {
          diagnostics_.send_failures += 1U;
          client->consecutive_send_failures += 1U;
          if (client->consecutive_send_failures >= kMaxConsecutiveSendFailures) {
            diagnostics_.slow_client_disconnects += 1U;
            (void) httpd_req_async_handler_complete(client->request);
            event_clients_.erase(client);
          }
        }
      }
      unlock_();
    }
  }
  const size_t current_count = event_client_count();
  if (previous_count != current_count) notify_client_count_(current_count);
}

bool EspIdfDirectHttpService::service_raw_stream_() {
  if (xSemaphoreTake(raw_send_mutex_, portMAX_DELAY) != pdTRUE) return false;
  RawCsiSessionConfig config;
  httpd_req_t *request = nullptr;
  if (!lock_()) {
    xSemaphoreGive(raw_send_mutex_);
    return false;
  }
  if (!raw_session_active_.load(std::memory_order_acquire) || !raw_session_.binary_bound ||
      raw_session_.request == nullptr) {
    unlock_();
    xSemaphoreGive(raw_send_mutex_);
    return false;
  }
  config = raw_session_.config;
  request = raw_session_.request;
  unlock_();

  size_t length = 0U;
  size_t records = 0U;
  uint64_t last_sequence = 0U;
  RawSampleSlot sample;
  while (records < kRawBatchRecords && pop_raw_sample_(&sample)) {
    const uint64_t sequence = sample.stream_sequence;
    last_sequence = sequence;
    const uint64_t fresh_total =
        raw_fresh_record_total_.load(std::memory_order_relaxed) + records + 1U;
    RawCsiHttpFramePrefixV2 prefix{};
    prefix.magic = ESPECTRE_RAW_CSI_RESPONSE_MAGIC;
    prefix.version = ESPECTRE_RAW_CSI_PROTOCOL_VERSION;
    prefix.record_version = ESPECTRE_RAW_CSI_RECORD_VERSION;
    prefix.header_len = sizeof(prefix);
    std::memcpy(prefix.session_id, config.session_id, sizeof(prefix.session_id));
    prefix.stream_sequence = sequence;
    prefix.record_len = static_cast<uint16_t>(sizeof(RawCsiRecordHeaderV8) + sample.metadata.csi_len);
    prefix.flags = 0U;
    prefix.fresh_record_total = fresh_total;
    prefix.raw_drop_total = raw_drop_total_.load(std::memory_order_relaxed);
    prefix.raw_send_backpressure_total = raw_send_backpressure_total_.load(std::memory_order_relaxed);
    std::memcpy(raw_send_buffer_.data() + length, &prefix, sizeof(prefix));
    length += sizeof(prefix);

    RawCsiRecordHeaderV8 header{};
    header.magic = RAW_CSI_RECORD_MAGIC;
    header.version = RAW_CSI_RECORD_VERSION_V8;
    header.header_len = sizeof(header);
    header.chip = static_cast<uint8_t>(config.chip);
    header.flags = sample.metadata.record_flags;
    header.seq_num = static_cast<uint32_t>(std::min<uint64_t>(sequence, UINT32_MAX));
    header.num_subcarriers = static_cast<uint16_t>(sample.metadata.csi_len / 2U);
    header.csi_len_bytes = sample.metadata.csi_len;
    header.device_id = config.device_id;
    header.device_ticks_us = sample.metadata.captured_at_us;
    header.wifi_rx_ts_us = sample.metadata.wifi_rx_ts_us;
    header.wifi_rx_start_ts_ns = sample.metadata.wifi_rx_start_ts_ns;
    header.channel = sample.metadata.channel;
    header.rssi_dbm = sample.metadata.rssi_dbm;
    header.noise_floor_dbm = sample.metadata.noise_floor_dbm;
    header.transport_backpressure_total = prefix.raw_send_backpressure_total;
    header.fresh_record_total = static_cast<uint32_t>(std::min<uint64_t>(fresh_total, UINT32_MAX));
    header.request_accepted_total =
        static_cast<uint32_t>(std::min<uint64_t>(sequence, UINT32_MAX));
    header.phy_mode = static_cast<uint8_t>(sample.metadata.phy_mode);
    header.ltf_type = static_cast<uint8_t>(sample.metadata.ltf_type);
    header.channel_width = static_cast<uint8_t>(sample.metadata.channel_width);
    std::memcpy(raw_send_buffer_.data() + length, &header, sizeof(header));
    length += sizeof(header);
    std::memcpy(raw_send_buffer_.data() + length, sample.csi.data(), sample.metadata.csi_len);
    length += sample.metadata.csi_len;
    records += 1U;
  }
  if (records == 0U) {
    xSemaphoreGive(raw_send_mutex_);
    return false;
  }

  const esp_err_t result = httpd_resp_send_chunk(request,
                                                  reinterpret_cast<const char *>(raw_send_buffer_.data()),
                                                  length);
  if (result != ESP_OK) {
    raw_send_backpressure_total_.fetch_add(1U, std::memory_order_relaxed);
    raw_drop_total_.fetch_add(records, std::memory_order_relaxed);
    xSemaphoreGive(raw_send_mutex_);
    (void) stop_raw_session(RawCsiStopReason::SLOW_CLIENT);
    return false;
  }
  raw_fresh_record_total_.fetch_add(records, std::memory_order_relaxed);
  if (lock_()) {
    raw_session_.last_send_us = static_cast<uint64_t>(esp_timer_get_time());
    raw_session_.stream_sequence = last_sequence;
    unlock_();
  }
  xSemaphoreGive(raw_send_mutex_);
  return true;
}

void EspIdfDirectHttpService::service_raw_timeouts_() {
  bool bind_timeout = false;
  const uint64_t now_us = static_cast<uint64_t>(esp_timer_get_time());
  if (lock_()) {
    bind_timeout = raw_session_active_.load(std::memory_order_acquire) &&
                   !raw_session_.binary_bound &&
                   now_us - raw_session_.opened_at_us >= kRawBindTimeoutUs;
    unlock_();
  }
  if (bind_timeout) (void) stop_raw_session(RawCsiStopReason::BIND_TIMEOUT);
}

void EspIdfDirectHttpService::dispatch_pending_callbacks_() {
  size_t client_count = 0U;
  if (pending_client_count_event_.take(client_count) && client_count_callback_) {
    client_count_callback_(client_count);
  }

  RawSessionStoppedCallback stopped_callback;
  RawCsiStopReason stop_reason = RawCsiStopReason::INTERNAL_ERROR;
  if (lock_()) {
    stopped_callback = std::move(pending_raw_stopped_callback_);
    if (stopped_callback) {
      stop_reason = pending_raw_stop_reason_;
    }
    unlock_();
  }
  if (stopped_callback) stopped_callback(stop_reason);
}

void EspIdfDirectHttpService::worker_loop_() {
  if (server_ == nullptr) return;
  service_event_streams_();
#if !defined(ESP_PLATFORM)
  (void) service_raw_stream_();
#endif
}

bool EspIdfDirectHttpService::pop_raw_sample_(RawSampleSlot *sample) {
  if (sample == nullptr) return false;
  const uint64_t head = raw_sample_head_.load(std::memory_order_relaxed);
  const uint64_t tail = raw_sample_tail_.load(std::memory_order_acquire);
  if (head == tail) return false;
  const RawSampleSlot &slot = raw_samples_[head % raw_samples_.size()];
  sample->metadata = slot.metadata;
  std::memcpy(sample->csi.data(), slot.csi.data(), slot.metadata.csi_len);
  sample->metadata.csi = sample->csi.data();
  sample->stream_sequence = slot.stream_sequence;
  raw_sample_head_.store(head + 1U, std::memory_order_release);
  return true;
}

void EspIdfDirectHttpService::reset_raw_session_locked_() {
  raw_session_ = {};
  raw_sample_head_.store(0U, std::memory_order_relaxed);
  raw_sample_tail_.store(0U, std::memory_order_relaxed);
}

void EspIdfDirectHttpService::notify_client_count_(size_t count) {
  pending_client_count_event_.post(count);
}

bool EspIdfDirectHttpService::lock_() const {
  return mutex_ != nullptr && xSemaphoreTake(mutex_, pdMS_TO_TICKS(10)) == pdTRUE;
}

void EspIdfDirectHttpService::unlock_() const { xSemaphoreGive(mutex_); }

}  // namespace espectre
