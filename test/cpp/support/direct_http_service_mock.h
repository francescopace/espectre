/*
 * ESPectre - Direct HTTP Service Mock
 *
 * Test double for the Native frontend Direct transport boundary.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <string>
#include <vector>

#include "direct_http_service.h"

namespace espectre {
namespace direct_http_service_mock {

struct PublishedEvent {
  std::string event_name;
  std::string data_json;
  bool replaceable_telemetry{false};
};

struct State {
  bool setup_result{true};
  bool running{false};
  bool shutdown_called{false};
  int setup_calls{0};
  size_t client_count{0U};
  DirectHttpServiceConfig last_config;
  DirectHttpServiceDiagnostics diagnostics;
  std::vector<PublishedEvent> published_events;
  IDirectHttpService::RequestHandler request_handler;
  IDirectHttpService::DeferredRequestHandler deferred_request_handler;
  IDirectHttpService::ClientCountCallback client_count_callback;
  uint64_t last_completed_token{0U};
  std::string last_deferred_response;
  bool raw_session_active{false};
  bool raw_start_result{true};
  RawCsiSessionConfig raw_config{};
  RawCsiSessionDiagnostics raw_diagnostics{};
  IDirectHttpService::RawSessionRequestedCallback raw_session_requested_callback;
  IDirectHttpService::RawSessionStoppedCallback raw_stopped_callback;
  size_t raw_offer_calls{0U};
};

extern State state;

void reset();

class MockDirectHttpService : public IDirectHttpService {
 public:
  bool setup(const DirectHttpServiceConfig &config,
             RequestHandler request_handler,
             ClientCountCallback client_count_callback) override;
  bool setup_deferred(const DirectHttpServiceConfig &config,
                      DeferredRequestHandler request_handler,
                      ClientCountCallback client_count_callback) override;
  bool complete_deferred_response(uint64_t connection_token, std::string response) override;
  void loop() override;
  void shutdown() override;
  bool running() const override;
  size_t event_client_count() const override;
  bool publish_event(const std::string &event_name,
                     const std::string &data_json,
                     bool replaceable_telemetry) override;
  DirectHttpServiceDiagnostics diagnostics() const override;
  void set_raw_session_requested_callback(RawSessionRequestedCallback callback) override;
  bool start_raw_session(const RawCsiSessionConfig &config,
                         RawSessionStoppedCallback stopped_callback) override;
  bool stop_raw_session(RawCsiStopReason reason) override;
  bool offer_raw_packet(const RawCsiPacketView &packet) override;
  RawCsiSessionDiagnostics raw_diagnostics() const override;

  std::string emit_request(const DirectRequest &request);
  DeferredRequestResult emit_deferred_request(uint64_t connection_token,
                                              const DirectRequest &request);
  void emit_client_count(size_t client_count);
  bool emit_raw_session_request(std::string *message = nullptr);
};

}  // namespace direct_http_service_mock
}  // namespace espectre
