/*
 * ESPectre - Direct HTTP Service Mock
 *
 * Test double for the Native frontend Direct transport boundary.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "direct_http_service_mock.h"

namespace espectre {
namespace direct_http_service_mock {

State state{};

void reset() { state = State{}; }

bool MockDirectHttpService::setup(const DirectHttpServiceConfig &config,
                                       RequestHandler request_handler,
                                       ClientCountCallback client_count_callback) {
  state.setup_calls += 1;
  state.last_config = config;
  state.request_handler = std::move(request_handler);
  state.deferred_request_handler = {};
  state.client_count_callback = std::move(client_count_callback);
  state.running = state.setup_result;
  return state.setup_result;
}

bool MockDirectHttpService::setup_deferred(const DirectHttpServiceConfig &config,
                                                DeferredRequestHandler request_handler,
                                                ClientCountCallback client_count_callback) {
  state.setup_calls += 1;
  state.last_config = config;
  state.request_handler = {};
  state.deferred_request_handler = std::move(request_handler);
  state.client_count_callback = std::move(client_count_callback);
  state.running = state.setup_result;
  return state.setup_result;
}

bool MockDirectHttpService::complete_deferred_response(uint64_t connection_token,
                                                            std::string response) {
  if (!state.running) {
    return false;
  }
  state.last_completed_token = connection_token;
  state.last_deferred_response = std::move(response);
  return true;
}

void MockDirectHttpService::loop() {}

void MockDirectHttpService::shutdown() {
  state.shutdown_called = true;
  state.running = false;
  state.client_count = 0U;
}

bool MockDirectHttpService::running() const { return state.running; }

size_t MockDirectHttpService::event_client_count() const { return state.client_count; }

bool MockDirectHttpService::publish_event(const std::string &event_name,
                                               const std::string &data_json,
                                               bool replaceable_telemetry) {
  if (!state.running || state.client_count == 0U) {
    return false;
  }
  state.published_events.push_back(PublishedEvent{event_name, data_json, replaceable_telemetry});
  return true;
}

DirectHttpServiceDiagnostics MockDirectHttpService::diagnostics() const { return state.diagnostics; }

bool MockDirectHttpService::start_raw_session(
    const RawCsiSessionConfig &config,
    RawSessionStoppedCallback stopped_callback) {
  if (!state.raw_start_result || state.raw_session_active) return false;
  state.raw_config = config;
  state.raw_stopped_callback = std::move(stopped_callback);
  state.raw_session_active = true;
  state.raw_diagnostics.active = true;
  return true;
}

bool MockDirectHttpService::stop_raw_session(RawCsiStopReason reason) {
  if (!state.raw_session_active) return false;
  state.raw_session_active = false;
  state.raw_diagnostics.active = false;
  auto callback = std::move(state.raw_stopped_callback);
  if (callback) callback(reason);
  return true;
}

bool MockDirectHttpService::offer_raw_packet(const RawCsiPacketView &packet) {
  if (!state.raw_session_active || packet.csi == nullptr) return false;
  state.raw_offer_calls += 1U;
  return true;
}

RawCsiSessionDiagnostics MockDirectHttpService::raw_diagnostics() const {
  return state.raw_diagnostics;
}

std::string MockDirectHttpService::emit_request(const DirectRequest &request) {
  if (state.request_handler) return state.request_handler(request);
  if (state.deferred_request_handler) {
    return state.deferred_request_handler(1U, request).response;
  }
  return {};
}

IDirectHttpService::DeferredRequestResult MockDirectHttpService::emit_deferred_request(
    uint64_t connection_token,
    const DirectRequest &request) {
  return state.deferred_request_handler
             ? state.deferred_request_handler(connection_token, request)
             : IDirectHttpService::DeferredRequestResult{};
}

void MockDirectHttpService::emit_client_count(size_t client_count) {
  state.client_count = client_count;
  if (state.client_count_callback) {
    state.client_count_callback(client_count);
  }
}

}  // namespace direct_http_service_mock
}  // namespace espectre
