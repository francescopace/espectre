/*
 * ESPectre - ESP-IDF Direct HTTP Service
 *
 * Bounded local HTTP, SSE, and binary streaming transport.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <atomic>
#include <deque>
#include <string>
#include <vector>

#include <esp_http_server.h>
#include <freertos/FreeRTOS.h>
#include <freertos/semphr.h>
#include <freertos/task.h>

#if !defined(ESP_PLATFORM) && !defined(portMUX_INITIALIZER_UNLOCKED)
using portMUX_TYPE = int;
#define portMUX_INITIALIZER_UNLOCKED 0
#define portENTER_CRITICAL(mux) (void)(mux)
#define portEXIT_CRITICAL(mux) (void)(mux)
#endif

#include "direct_http_service.h"
#include "pending_event.h"

namespace espectre {

class EspIdfDirectHttpService final : public IDirectHttpService {
 public:
  EspIdfDirectHttpService();
  ~EspIdfDirectHttpService() override;

  bool setup(const DirectHttpServiceConfig &config,
             RequestHandler request_handler,
             ClientCountCallback client_count_callback) override;
  bool setup_deferred(const DirectHttpServiceConfig &config,
                      DeferredRequestHandler request_handler,
                      ClientCountCallback client_count_callback) override;
  bool complete_deferred_response(uint64_t request_token, std::string response) override;
  void loop() override;
  void shutdown() override;
  bool running() const override;
  size_t event_client_count() const override;
  bool publish_event(const std::string &event_name,
                     const std::string &data_json,
                     bool replaceable_telemetry) override;
  DirectHttpServiceDiagnostics diagnostics() const override;
  bool start_raw_session(const RawCsiSessionConfig &config,
                         RawSessionStoppedCallback stopped_callback) override;
  bool stop_raw_session(RawCsiStopReason reason) override;
  bool offer_raw_packet(const RawCsiPacketView &packet) override;
  RawCsiSessionDiagnostics raw_diagnostics() const override;

 private:
  struct OutboundEvent {
    std::string payload;
    std::string event_name;
    bool replaceable_telemetry{false};
  };

  struct EventClient {
    httpd_req_t *request{nullptr};
    int fd{-1};
    uint8_t consecutive_send_failures{0U};
    uint64_t last_send_us{0U};
    std::deque<OutboundEvent> outbound;
  };

  struct PendingRequest {
    uint64_t token{0U};
    httpd_req_t *request{nullptr};
    DirectRequest direct;
    std::string origin;
  };

  struct CompletedResponse {
    PendingRequest request;
    std::string response;
  };

  struct RawSampleSlot {
    RawCsiPacketView metadata{};
    std::array<int8_t, RAW_CSI_MAX_PAYLOAD_BYTES> csi{};
    uint64_t stream_sequence{0U};
  };

  struct RawSessionState {
    RawCsiSessionConfig config{};
    RawSessionStoppedCallback stopped_callback{};
    httpd_req_t *request{nullptr};
    int fd{-1};
    bool binary_bound{false};
    uint64_t generation{0U};
    uint64_t opened_at_us{0U};
    uint64_t last_send_us{0U};
    uint64_t stream_sequence{0U};
    std::string origin;
  };

  static esp_err_t request_uri_handler_(httpd_req_t *request);
  static esp_err_t events_handler_(httpd_req_t *request);
  static esp_err_t raw_handler_(httpd_req_t *request);
  static esp_err_t options_handler_(httpd_req_t *request);
  static void worker_entry_(void *context);
  static void raw_worker_entry_(void *context);

  esp_err_t handle_request_(httpd_req_t *request);
  esp_err_t handle_events_(httpd_req_t *request);
  esp_err_t handle_raw_(httpd_req_t *request);
  esp_err_t handle_options_(httpd_req_t *request);
  bool validate_origin_(httpd_req_t *request, std::string *origin);
  void set_response_headers_(httpd_req_t *request, const std::string &origin) const;
  esp_err_t send_error_(httpd_req_t *request,
                        const char *status,
                        const char *message,
                        const std::string &origin) const;
  bool read_header_(httpd_req_t *request, const char *name, std::string *value) const;
  bool read_bearer_(httpd_req_t *request, std::string *value) const;
  bool mutation_allowed_locked_(const std::string &method, uint64_t now_us);
  bool enqueue_event_locked_(EventClient *client, OutboundEvent event);
  void enqueue_completed_response_locked_(PendingRequest request, std::string response);
  bool enqueue_completed_response_(PendingRequest request, std::string response);
  void release_request_(PendingRequest request);
  bool finish_request_(PendingRequest request, const std::string &response);
  void service_event_streams_();
  bool service_raw_stream_();
  void service_raw_timeouts_();
  void dispatch_pending_callbacks_();
  void worker_loop_();
  bool pop_raw_sample_(RawSampleSlot *sample);
  void reset_raw_session_locked_();
  void notify_client_count_(size_t count);
  bool lock_() const;
  void unlock_() const;

  mutable SemaphoreHandle_t mutex_{nullptr};
  SemaphoreHandle_t raw_send_mutex_{nullptr};
  httpd_handle_t server_{nullptr};
  DirectHttpServiceConfig config_{};
  RequestHandler request_handler_{};
  DeferredRequestHandler deferred_request_handler_{};
  ClientCountCallback client_count_callback_{};
  PendingEvent<size_t> pending_client_count_event_{};
  std::vector<EventClient> event_clients_;
  size_t pending_event_connections_{0U};
  std::deque<PendingRequest> inbound_;
  std::vector<PendingRequest> deferred_;
  std::deque<CompletedResponse> completed_;
  DirectHttpServiceDiagnostics diagnostics_{};
  uint64_t next_request_token_{1U};
  uint64_t mutation_window_started_us_{0U};
  uint16_t mutation_count_{0U};
  uint64_t next_raw_session_generation_{1U};
  std::atomic<bool> worker_running_{false};
  TaskHandle_t worker_task_{nullptr};
  std::atomic<bool> raw_worker_running_{false};
  [[maybe_unused]] TaskHandle_t raw_worker_task_{nullptr};

  static constexpr size_t kRawQueueDepth = 16U;
  static constexpr size_t kRawBatchRecords = 4U;
  static constexpr size_t kRawEncodedFrameMaximumSize =
      sizeof(RawCsiHttpFramePrefixV2) + sizeof(RawCsiRecordHeaderV8) + RAW_CSI_MAX_PAYLOAD_BYTES;
  std::atomic<bool> raw_session_active_{false};
  std::atomic<uint32_t> raw_producer_active_{0U};
  std::array<RawSampleSlot, kRawQueueDepth> raw_samples_{};
  std::atomic<uint64_t> raw_sample_head_{0U};
  std::atomic<uint64_t> raw_sample_tail_{0U};
  std::atomic<uint64_t> raw_offer_sequence_{0U};
  std::array<uint8_t, kRawBatchRecords * kRawEncodedFrameMaximumSize> raw_send_buffer_{};
  RawSessionState raw_session_{};
  RawSessionStoppedCallback pending_raw_stopped_callback_{};
  RawCsiStopReason pending_raw_stop_reason_{RawCsiStopReason::INTERNAL_ERROR};
  std::atomic<uint64_t> raw_drop_total_{0U};
  std::atomic<uint64_t> raw_send_backpressure_total_{0U};
  std::atomic<uint64_t> raw_fresh_record_total_{0U};
};

}  // namespace espectre
