/*
 * ESPectre - Direct HTTP Protocol
 *
 * HTTP framing for the transport-neutral ESPectre message model.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "espectre_protocol.h"

namespace espectre {

/** ESPectre service port: low 16 bits of U+1F47B GHOST (0xF47B). */
inline constexpr uint16_t ESPECTRE_DIRECT_HTTP_PORT = 0xF47BU;  // 62587
/** Path prefix of every versioned Direct resource. */
inline constexpr const char *ESPECTRE_DIRECT_HTTP_BASE_ENDPOINT = "/espectre/v1";
/** Server-sent events stream. */
inline constexpr const char *ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT = "/espectre/v1/events";
/** Transport name reported in capabilities and discovery. */
inline constexpr const char *ESPECTRE_DIRECT_HTTP_TRANSPORT = "http";
/** Largest request body accepted, in bytes. */
inline constexpr size_t ESPECTRE_DIRECT_MAX_REQUEST_SIZE = ESPECTRE_COMMAND_MAX_PAYLOAD_SIZE;
/** Largest response body produced, in bytes. */
inline constexpr size_t ESPECTRE_DIRECT_MAX_RESPONSE_SIZE = 8192U;

/** A Direct HTTP request mapped onto a canonical command. */
struct DirectRequest {
  /** Correlation id; empty, because the HTTP response itself is the correlation. */
  std::string command_id;
  /** Canonical command name selected by the route. */
  std::string command;
  /** Syntactically valid JSON object containing command parameters. */
  std::string params{"{}"};
  /** Request path, retained for resource-aware response handling. */
  std::string path;
  /** HTTP method selected by the transport route. */
  std::string http_method;
  /** Whether the accepted operation completes asynchronously. */
  bool asynchronous{false};
};

/**
 * Map an HTTP method, path, and body onto a canonical command request.
 *
 * Looks the route up in the SDK registry, then in `extension`. Only
 * `GET /espectre/v1/diagnostics` accepts a query, `?fields=`. The body must be
 * a JSON object within `ESPECTRE_DIRECT_MAX_REQUEST_SIZE`; parameter values are
 * validated later, by direct_http_request_to_command().
 *
 * @return false for an unknown route, a malformed query, or an invalid body,
 *         with a reason in `error`.
 */
bool parse_direct_http_request(const std::string &http_method,
                               const std::string &path,
                               const std::string &payload,
                               DirectRequest *request,
                               std::string *error = nullptr,
                               const EspectreProtocolExtension *extension = nullptr);
/**
 * Validate a parsed Direct request and build the command, as
 * parse_espectre_command_request() does for every transport.
 */
bool direct_http_request_to_command(const DirectRequest &request,
                                    EspectreCommand *command,
                                    std::string *error = nullptr,
                                    const EspectreProtocolExtension *extension = nullptr);

/** How each canonical message maps onto Direct HTTP routes and MQTT topics. */
std::string espectre_transport_mapping_payload();
/** Combined message-model and transport-mapping catalog for protocol inspection. */
std::string espectre_protocol_catalog_payload(const EspectreProtocolExtension *extension = nullptr);

}  // namespace espectre
