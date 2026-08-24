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
#include "direct_websocket_protocol.h"

#include <cctype>
#include <vector>

#include "espectre_protocol.h"
#include "protocol_json.h"

namespace espectre {

namespace {

bool identifier_accepted(const std::string &value, size_t max_size, bool method) {
  if (value.empty() || value.size() > max_size) {
    return false;
  }
  for (const unsigned char ch : value) {
    const bool accepted = std::isalnum(ch) || ch == '_' || ch == '-' || ch == '.' || (!method && ch == ':');
    if (!accepted) {
      return false;
    }
  }
  return true;
}

bool validated_json_object(const std::string &payload) {
  std::vector<JsonObjectField> fields;
  return parse_json_object_fields(payload, &fields, nullptr);
}

std::string envelope_prefix(const char *type) {
  std::string out;
  out.reserve(48U);
  out = "{\"v\":";
  out += std::to_string(ESPECTRE_DIRECT_ENVELOPE_VERSION);
  append_json_pair(&out, "type", type != nullptr ? type : "", false);
  return out;
}

}  // namespace

bool parse_direct_websocket_request(const std::string &payload,
                                    DirectWebSocketRequest *request,
                                    std::string *error) {
  if (request == nullptr) {
    if (error != nullptr) {
      *error = "request output is required";
    }
    return false;
  }
  DirectWebSocketRequest parsed;
  const auto reject = [&](const char *message) {
    if (error != nullptr) {
      *error = message;
    }
    *request = parsed;
    return false;
  };
  if (payload.empty()) {
    return reject("empty Direct frame");
  }
  if (payload.size() > ESPECTRE_DIRECT_MAX_FRAME_SIZE) {
    return reject("Direct frame exceeds the size limit");
  }

  std::vector<JsonObjectField> fields;
  std::string json_error;
  if (!parse_json_object_fields(payload, &fields, &json_error)) {
    if (error != nullptr) {
      *error = json_error.empty() ? "invalid Direct JSON envelope" : json_error;
    }
    *request = parsed;
    return false;
  }

  const JsonObjectField *version = find_json_object_field(fields, "v");
  if (version == nullptr || version->type != JsonValueType::NUMBER || version->value != "1") {
    return reject("unsupported Direct envelope version");
  }
  const JsonObjectField *type = find_json_object_field(fields, "type");
  if (type == nullptr || type->type != JsonValueType::STRING || type->value != "request") {
    return reject("Direct client frames must have type request");
  }
  const JsonObjectField *id = find_json_object_field(fields, "id");
  if (id == nullptr || id->type != JsonValueType::STRING ||
      !identifier_accepted(id->value, ESPECTRE_DIRECT_MAX_REQUEST_ID_SIZE, false)) {
    return reject("invalid Direct request id");
  }
  parsed.id = id->value;
  const JsonObjectField *method = find_json_object_field(fields, "method");
  if (method == nullptr || method->type != JsonValueType::STRING ||
      !identifier_accepted(method->value, ESPECTRE_DIRECT_MAX_METHOD_SIZE, true)) {
    return reject("invalid Direct request method");
  }
  parsed.method = method->value;
  const JsonObjectField *params = find_json_object_field(fields, "params");
  if (params != nullptr) {
    if (params->type != JsonValueType::OBJECT) {
      return reject("Direct request params must be an object");
    }
    parsed.params = params->value;
  }
  *request = std::move(parsed);
  return true;
}

bool direct_websocket_request_to_command(const DirectWebSocketRequest &request,
                                         EspectreCommand *command,
                                         std::string *error) {
  return parse_espectre_command_request(request.id, request.method, request.params, command, error);
}

std::string direct_websocket_success_response(const std::string &id, const std::string &result_json) {
  const std::string result = validated_json_object(result_json) ? result_json : "{}";
  std::string out = envelope_prefix("response");
  append_json_pair(&out, "id", id.c_str(), false);
  out += ",\"ok\":true,\"result\":";
  out += result;
  out += "}";
  return out;
}

std::string direct_websocket_error_response(const std::string &id, const char *code, const char *message) {
  std::string out = envelope_prefix("response");
  append_json_pair(&out, "id", id.c_str(), false);
  out += ",\"ok\":false,\"error\":{";
  append_json_pair(&out, "code", code != nullptr ? code : "internal_error", true);
  append_json_pair(&out, "message", message != nullptr ? message : "", false);
  out += "}}";
  return out;
}

std::string direct_websocket_event(const char *event, const std::string &data_json) {
  const std::string data = validated_json_object(data_json) ? data_json : "{}";
  std::string out = envelope_prefix("event");
  append_json_pair(&out, "event", event != nullptr ? event : "", false);
  out += ",\"data\":";
  out += data;
  out += "}";
  return out;
}

}  // namespace espectre
