/*
 * ESPectre - Direct HTTP Service Boundary
 *
 * Transport boundary for local HTTP carriage of canonical ESPectre messages.
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

#include "direct_http_protocol.h"
#include "raw_csi.h"

namespace espectre {

/**
 * Settings for an IDirectHttpService.
 *
 * setup() rejects a configuration without allowed origins unless
 * `allow_missing_origin` is set, and zero limits.
 */
struct DirectHttpServiceConfig {
  /** Configuration for ESPectre's production and validation portals. */
  static DirectHttpServiceConfig for_first_party_portals() {
    DirectHttpServiceConfig config;
    config.allowed_origins = {
        "https://espectre.dev",
        "https://www.espectre.dev",
        "https://test.espectre.dev",
    };
    return config;
  }

  /** Browser origins allowed to call the API, matched exactly, such as `https://espectre.dev`. */
  std::vector<std::string> allowed_origins;
  /** Device identity used in error payloads the service builds itself. */
  uint64_t device_id{0U};
  uint16_t port{ESPECTRE_DIRECT_HTTP_PORT};
  /** Concurrent event stream clients, 1 or 2. */
  size_t max_event_clients{2U};
  /** Requests received but not yet answered, across all clients. */
  size_t max_pending_requests{4U};
  /** Events queued per event stream client. */
  size_t outbound_queue_depth{8U};
  /** Requests accepted per second, reads included. */
  uint16_t max_requests_per_second{20U};
  /** State-changing requests accepted per minute. */
  uint16_t max_mutations_per_minute{60U};
  /** Accept requests without an `Origin` header, such as those from the CLI. */
  bool allow_missing_origin{false};
  /** Also accept `http://localhost` and loopback addresses as origins, for local development. */
  bool allow_http_loopback_origins{false};
  /** Optional frontend routes. The immutable catalog must outlive this service. */
  const EspectreProtocolExtension *protocol_extension{nullptr};
};

/** Counters of an IDirectHttpService since its last setup(). */
struct DirectHttpServiceDiagnostics {
  /** `DirectHttpServiceConfig::max_event_clients`. */
  size_t event_client_limit{0U};
  /** `DirectHttpServiceConfig::outbound_queue_depth`. */
  size_t queue_capacity{0U};
  /** Event stream connections accepted. */
  uint32_t accepted_connections{0U};
  /** Event stream connections refused, usually because the client limit was reached. */
  uint32_t rejected_connections{0U};
  /** Requests with an invalid route, query, or body. */
  uint32_t malformed_requests{0U};
  /** Requests larger than `ESPECTRE_DIRECT_MAX_REQUEST_SIZE`. */
  uint32_t oversized_requests{0U};
  /** Requests refused by the per-second or per-minute limits. */
  uint32_t rate_limited_requests{0U};
  /** Replaceable telemetry events dropped because a client queue was full. */
  uint32_t dropped_motion_events{0U};
  /** Failed sends to event stream clients. */
  uint32_t send_failures{0U};
  /** Events currently queued across clients. */
  size_t queued_messages{0U};
};

/** Local HTTP endpoints shared by ESPectre firmware frontends. */
class IDirectHttpService {
 public:
  /** Answer a request synchronously with a canonical JSON response. Runs from loop(). */
  using RequestHandler = std::function<std::string(const DirectRequest &request)>;
  /** Reports whether the response reached the client. */
  using ResponseSentCallback = std::function<void(bool sent)>;
  /** What a DeferredRequestHandler decided for one request. */
  struct DeferredRequestResult {
    /** True to answer later with complete_deferred_response(); `response` is then ignored. */
    bool deferred{false};
    /** Immediate response when `deferred` is false. */
    std::string response;
    /** Runs on the frontend task after the response send attempt completes. */
    ResponseSentCallback response_sent_callback{};
  };
  /** Answer now or later; keep `request_token` to complete the request. Runs from loop(). */
  using DeferredRequestHandler =
      std::function<DeferredRequestResult(uint64_t request_token, const DirectRequest &request)>;
  /** Reports the number of connected event stream clients after it changes. */
  using ClientCountCallback = std::function<void(size_t event_client_count)>;
  /** Open raw collection for `GET /csi`; return false with a reason to refuse it. */
  using RawSessionRequestedCallback = std::function<bool(std::string *message)>;
  /** Reports why a raw session ended. */
  using RawSessionStoppedCallback = std::function<void(RawCsiStopReason reason)>;

  virtual ~IDirectHttpService() = default;

  /** Configure and start the endpoint. Safe to call again after shutdown. */
  virtual bool setup(const DirectHttpServiceConfig &config,
                     RequestHandler request_handler,
                     ClientCountCallback client_count_callback) = 0;
  /**
   * Configure a handler that may complete a request later.
   *
   * The default preserves source compatibility for external transports that
   * implement only synchronous Direct requests. A successful deferred handler
   * must eventually call complete_deferred_response() with the opaque token.
   */
  virtual bool setup_deferred(const DirectHttpServiceConfig &config,
                              DeferredRequestHandler request_handler,
                              ClientCountCallback client_count_callback) {
    (void) config;
    (void) request_handler;
    (void) client_count_callback;
    return false;
  }
  /** Queue a deferred response only if the originating connection is live. */
  virtual bool complete_deferred_response(uint64_t request_token, std::string response) {
    (void) request_token;
    (void) response;
    return false;
  }
  /**
   * Pump deferred receive, dispatch, send work, and application callbacks from
   * the frontend task. Request, client-count, and raw-stop callbacks are never
   * delivered from HTTP server or streaming worker tasks.
   */
  virtual void loop() = 0;
  /** Stop accepting clients, close sockets, and release queued messages. */
  virtual void shutdown() = 0;
  /** True between a successful setup and shutdown(). */
  virtual bool running() const = 0;
  /** Connected event stream clients. */
  virtual size_t event_client_count() const = 0;

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
  /** Current counters. */
  virtual DirectHttpServiceDiagnostics diagnostics() const = 0;

  /** Register the frontend-task callback that opens collection for GET /csi. */
  virtual void set_raw_session_requested_callback(RawSessionRequestedCallback callback) {
    (void) callback;
  }

  /** Begin one owner-bound raw session on the service's binary endpoint. */
  virtual bool start_raw_session(const RawCsiSessionConfig &config,
                                 RawSessionStoppedCallback stopped_callback) {
    (void) config;
    (void) stopped_callback;
    return false;
  }
  /**
   * Stop the active raw session and close its binary socket.
   *
   * The stopped callback is delivered by loop(), or synchronously while
   * shutdown() completes on the owning frontend task.
   */
  virtual bool stop_raw_session(RawCsiStopReason reason) {
    (void) reason;
    return false;
  }
  /** Copy one callback-scoped sample into the transport's bounded raw slots. */
  virtual bool offer_raw_packet(const RawCsiPacketView &packet) {
    (void) packet;
    return false;
  }
  /** Counters of the current or last raw session; zeros when unsupported. */
  virtual RawCsiSessionDiagnostics raw_diagnostics() const { return {}; }
};

}  // namespace espectre
