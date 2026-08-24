/*
 * ESPectre - Direct WebSocket Service Mock
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

#include "direct_websocket_service.h"

namespace espectre {
namespace direct_websocket_service_mock {

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
  DirectWebSocketServiceConfig last_config;
  DirectWebSocketServiceDiagnostics diagnostics;
  std::vector<PublishedEvent> published_events;
  IDirectWebSocketService::RequestHandler request_handler;
  IDirectWebSocketService::DeferredRequestHandler deferred_request_handler;
  IDirectWebSocketService::ClientCountCallback client_count_callback;
  uint64_t last_completed_token{0U};
  std::string last_deferred_response;
};

extern State state;

void reset();

class MockDirectWebSocketService : public IDirectWebSocketService {
 public:
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

  std::string emit_request(const DirectWebSocketRequest &request);
  DeferredRequestResult emit_deferred_request(uint64_t connection_token,
                                              const DirectWebSocketRequest &request);
  void emit_client_count(size_t client_count);
};

}  // namespace direct_websocket_service_mock
}  // namespace espectre
