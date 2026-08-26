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
inline constexpr const char *ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT = "/espectre/v1/request";
inline constexpr const char *ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT = "/espectre/v1/events";
inline constexpr const char *ESPECTRE_DIRECT_HTTP_TRANSPORT = "http";
inline constexpr size_t ESPECTRE_DIRECT_MAX_REQUEST_SIZE = 4096U;
inline constexpr size_t ESPECTRE_DIRECT_MAX_RESPONSE_SIZE = 8192U;
/** @deprecated Use `ESPECTRE_COMMAND_ID_MAX_LENGTH`; Direct uses the canonical limit. */
[[deprecated("use ESPECTRE_COMMAND_ID_MAX_LENGTH")]] inline constexpr size_t
    ESPECTRE_DIRECT_MAX_REQUEST_ID_SIZE = ESPECTRE_COMMAND_ID_MAX_LENGTH;
/** @deprecated Command names are validated by the canonical registry. */
[[deprecated("command names are validated by the canonical registry")]] inline constexpr size_t
    ESPECTRE_DIRECT_MAX_METHOD_SIZE = 64U;

struct DirectRequest {
  std::string command_id;
  std::string command;
  /** Syntactically valid JSON object containing command parameters. */
  std::string params{"{}"};
  /** Bearer token supplied by the HTTP transport; never parsed from JSON. */
  std::string authorization;
  /** Canonical message version supplied by the requester. */
  std::string protocol_version{ESPECTRE_PROTOCOL_VERSION};
};

bool parse_direct_http_request(const std::string &payload,
                               DirectRequest *request,
                               std::string *error = nullptr);
bool direct_http_request_to_command(const DirectRequest &request,
                                    EspectreCommand *command,
                                    std::string *error = nullptr);

/** Executable Direct/MQTT mapping owned and tested by C++. */
std::string espectre_transport_mapping_payload();
/** Combined message-model and transport-mapping catalog for protocol inspection. */
std::string espectre_protocol_catalog_payload();

}  // namespace espectre
