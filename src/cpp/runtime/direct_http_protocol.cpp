/*
 * ESPectre - Direct HTTP Protocol
 *
 * HTTP framing for the transport-neutral ESPectre message model.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "direct_http_protocol.h"

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

}  // namespace

bool parse_direct_http_request(const std::string &payload,
                               DirectRequest *request,
                               std::string *error) {
  if (request == nullptr) {
    if (error != nullptr) {
      *error = "request output is required";
    }
    return false;
  }
  DirectRequest parsed;
  const auto reject = [&](const char *message) {
    if (error != nullptr) {
      *error = message;
    }
    *request = parsed;
    return false;
  };
  if (payload.empty()) {
    return reject("empty Direct request");
  }
  if (payload.size() > ESPECTRE_DIRECT_MAX_REQUEST_SIZE) {
    return reject("Direct request exceeds the size limit");
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

  const JsonObjectField *version = find_json_object_field(fields, "protocol_version");
  if (version == nullptr || version->type != JsonValueType::STRING ||
      version->value != ESPECTRE_PROTOCOL_VERSION) {
    return reject("unsupported ESPectre protocol version");
  }
  const JsonObjectField *id = find_json_object_field(fields, "command_id");
  if (id == nullptr || id->type != JsonValueType::STRING ||
      !identifier_accepted(id->value, ESPECTRE_DIRECT_MAX_REQUEST_ID_SIZE, false)) {
    return reject("invalid ESPectre command_id");
  }
  parsed.id = id->value;
  const JsonObjectField *method = find_json_object_field(fields, "command");
  if (method == nullptr || method->type != JsonValueType::STRING ||
      !identifier_accepted(method->value, ESPECTRE_DIRECT_MAX_METHOD_SIZE, true)) {
    return reject("invalid ESPectre command");
  }
  parsed.method = method->value;

  EspectreCommand command;
  if (!parse_espectre_command(payload, &command, &json_error)) {
    if (error != nullptr) *error = json_error;
    *request = parsed;
    return false;
  }

  parsed.params = "{";
  bool first = true;
  for (const JsonObjectField &field : fields) {
    if (field.name == "protocol_version" || field.name == "command_id" || field.name == "command") {
      continue;
    }
    if (!first) parsed.params += ',';
    append_json_string(&parsed.params, field.name.c_str());
    parsed.params += ':';
    if (field.type == JsonValueType::STRING) {
      append_json_string(&parsed.params, field.value.c_str());
    } else {
      parsed.params += field.value;
    }
    first = false;
  }
  parsed.params += '}';
  *request = std::move(parsed);
  return true;
}

bool direct_http_request_to_command(const DirectRequest &request,
                                    EspectreCommand *command,
                                    std::string *error) {
  return parse_espectre_command_request(request.id, request.method, request.params, command, error);
}

std::string espectre_transport_mapping_payload() {
  std::string out{"{"};
  out += "\"direct\":{\"request\":{\"framing\":\"http_post\"";
  append_json_pair(&out, "path", ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT);
  append_json_pair(&out, "message", "request");
  out += "},\"result\":{\"framing\":\"http_response_body\",\"message\":\"result\"},"
         "\"events\":{\"framing\":\"sse\"";
  append_json_pair(&out, "path", ESPECTRE_DIRECT_HTTP_EVENTS_ENDPOINT);
  append_json_pair(&out, "name", "event");
  append_json_pair(&out, "data", "event_payload");
  out += "}},\"mqtt\":{\"request\":{\"topic_suffix\":\"commands/request\",\"message\":\"request\"},"
         "\"result\":{\"topic_suffix\":\"commands/result\",\"message\":\"result\"},"
         "\"events\":{\"topic_suffix\":\"{event}\",\"message\":\"event_payload\"}}}";
  return out;
}

std::string espectre_protocol_catalog_payload() {
  return "{\"message_model\":" + espectre_message_catalog_payload() +
         ",\"transport_mapping\":" + espectre_transport_mapping_payload() + "}";
}

}  // namespace espectre
