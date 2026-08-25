/*
 * ESPectre - Direct HTTP Protocol
 *
 * Versioned request, response, error, and event envelopes for local HTTP.
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

inline constexpr const char *ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT = "/espectre/v1/request";
inline constexpr const char *ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT = "/espectre/v1/events";
inline constexpr const char *ESPECTRE_DIRECT_HTTP_TRANSPORT = "http";
inline constexpr const char *ESPECTRE_DIRECT_DISCOVERY_TXT_VERSION = "2";
inline constexpr unsigned ESPECTRE_DIRECT_ENVELOPE_VERSION = 1U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_REQUEST_SIZE = 4096U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_RESPONSE_SIZE = 8192U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_REQUEST_ID_SIZE = 64U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_METHOD_SIZE = 64U;

struct DirectRequest {
  std::string id;
  std::string method;
  /** Validated JSON object containing method parameters. */
  std::string params{"{}"};
  /** Bearer token supplied by the HTTP transport; never parsed from JSON. */
  std::string authorization;
};

bool parse_direct_http_request(const std::string &payload,
                               DirectRequest *request,
                               std::string *error = nullptr);
bool direct_http_request_to_command(const DirectRequest &request,
                                    EspectreCommand *command,
                                    std::string *error = nullptr);

std::string direct_http_success_response(const std::string &id, const std::string &result_json = "{}");
std::string direct_http_error_response(const std::string &id,
                                       const char *code,
                                       const char *message);
std::string direct_http_event(const char *event, const std::string &data_json);

}  // namespace espectre
