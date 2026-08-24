/*
 * ESPectre - Direct WebSocket Service Boundary
 *
 * Transport boundary for the local, versioned Direct WebSocket endpoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "direct_websocket_protocol.h"

namespace espectre {

struct DirectWebSocketServiceConfig {
  std::vector<std::string> allowed_origins;
  uint16_t port{80U};
  size_t max_clients{2U};
  size_t outbound_queue_depth{8U};
  uint16_t max_mutations_per_minute{60U};
  bool allow_missing_origin{false};
  bool allow_http_loopback_origins{false};
};

struct DirectWebSocketServiceDiagnostics {
  size_t client_limit{0U};
  size_t queue_capacity{0U};
  uint32_t accepted_connections{0U};
  uint32_t rejected_connections{0U};
  uint32_t malformed_frames{0U};
  uint32_t oversized_frames{0U};
  uint32_t rate_limited_requests{0U};
  uint32_t dropped_telemetry_events{0U};
  uint32_t send_failures{0U};
  uint32_t slow_client_disconnects{0U};
  size_t queued_messages{0U};
};

/** Local WebSocket endpoint shared by ESPectre firmware frontends. */
class IDirectWebSocketService {
 public:
  using RequestHandler = std::function<std::string(const DirectWebSocketRequest &request)>;
  using ClientCountCallback = std::function<void(size_t client_count)>;

  virtual ~IDirectWebSocketService() = default;

  /** Configure and start the endpoint. Safe to call again after shutdown. */
  virtual bool setup(const DirectWebSocketServiceConfig &config,
                     RequestHandler request_handler,
                     ClientCountCallback client_count_callback) = 0;
  /** Pump deferred receive, dispatch, and send work from the frontend task. */
  virtual void loop() = 0;
  /** Stop accepting clients, close sockets, and release queued messages. */
  virtual void shutdown() = 0;
  virtual bool running() const = 0;
  virtual size_t client_count() const = 0;

  /**
   * Queue a normalized event for every connected client.
   *
   * Telemetry events may replace an older queued event with the same name.
   * State transitions and command responses must never be replaced by
   * telemetry. Returns false when no client can accept the event.
   */
  virtual bool publish_event(const std::string &event_name,
                             const std::string &data_json,
                             bool replaceable_telemetry) = 0;
  virtual DirectWebSocketServiceDiagnostics diagnostics() const = 0;
};

}  // namespace espectre
