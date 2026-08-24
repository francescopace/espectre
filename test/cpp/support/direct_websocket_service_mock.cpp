/*
 * ESPectre - Direct WebSocket Service Mock
 *
 * Test double for the Native frontend Direct transport boundary.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "direct_websocket_service_mock.h"

namespace espectre {
namespace direct_websocket_service_mock {

State state{};

void reset() { state = State{}; }

bool MockDirectWebSocketService::setup(const DirectWebSocketServiceConfig &config,
                                       RequestHandler request_handler,
                                       ClientCountCallback client_count_callback) {
  state.setup_calls += 1;
  state.last_config = config;
  state.request_handler = std::move(request_handler);
  state.client_count_callback = std::move(client_count_callback);
  state.running = state.setup_result;
  return state.setup_result;
}

void MockDirectWebSocketService::loop() {}

void MockDirectWebSocketService::shutdown() {
  state.shutdown_called = true;
  state.running = false;
  state.client_count = 0U;
}

bool MockDirectWebSocketService::running() const { return state.running; }

size_t MockDirectWebSocketService::client_count() const { return state.client_count; }

bool MockDirectWebSocketService::publish_event(const std::string &event_name,
                                               const std::string &data_json,
                                               bool replaceable_telemetry) {
  if (!state.running || state.client_count == 0U) {
    return false;
  }
  state.published_events.push_back(PublishedEvent{event_name, data_json, replaceable_telemetry});
  return true;
}

DirectWebSocketServiceDiagnostics MockDirectWebSocketService::diagnostics() const { return state.diagnostics; }

std::string MockDirectWebSocketService::emit_request(const DirectWebSocketRequest &request) {
  return state.request_handler ? state.request_handler(request) : std::string{};
}

void MockDirectWebSocketService::emit_client_count(size_t client_count) {
  state.client_count = client_count;
  if (state.client_count_callback) {
    state.client_count_callback(client_count);
  }
}

}  // namespace direct_websocket_service_mock
}  // namespace espectre
