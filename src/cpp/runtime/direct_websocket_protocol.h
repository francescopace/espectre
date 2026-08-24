/*
 * ESPectre - Direct WebSocket Protocol
 *
 * Versioned request, response, error, and event envelopes for the local
 * Direct WebSocket transport.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <string>

namespace espectre {

struct EspectreCommand;

inline constexpr const char *ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT = "/espectre/v1/ws";
inline constexpr const char *ESPECTRE_DIRECT_WEBSOCKET_SUBPROTOCOL = "espectre.v1";
inline constexpr unsigned ESPECTRE_DIRECT_ENVELOPE_VERSION = 1U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_REQUEST_FRAME_SIZE = 4096U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_RESPONSE_FRAME_SIZE = 8192U;
// Compatibility alias for integrations that use the original request bound.
inline constexpr size_t ESPECTRE_DIRECT_MAX_FRAME_SIZE = ESPECTRE_DIRECT_MAX_REQUEST_FRAME_SIZE;
inline constexpr size_t ESPECTRE_DIRECT_MAX_REQUEST_ID_SIZE = 64U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_METHOD_SIZE = 64U;

struct DirectWebSocketRequest {
  std::string id;
  std::string method;
  /** Validated JSON object containing method parameters. */
  std::string params{"{}"};
};

bool parse_direct_websocket_request(const std::string &payload,
                                    DirectWebSocketRequest *request,
                                    std::string *error = nullptr);
bool direct_websocket_request_to_command(const DirectWebSocketRequest &request,
                                         EspectreCommand *command,
                                         std::string *error = nullptr);

std::string direct_websocket_success_response(const std::string &id, const std::string &result_json = "{}");
std::string direct_websocket_error_response(const std::string &id,
                                            const char *code,
                                            const char *message);
std::string direct_websocket_event(const char *event, const std::string &data_json);

}  // namespace espectre
