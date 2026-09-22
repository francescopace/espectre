/*
 * ESPectre - Protocol JSON
 *
 * JSON helpers for shared ESPectre protocol payloads.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace espectre {

enum class JsonValueType : uint8_t {
  STRING = 0,
  NUMBER,
  BOOLEAN,
  NULL_VALUE,
  OBJECT,
  ARRAY,
};

struct JsonObjectField {
  std::string name;
  JsonValueType type{JsonValueType::NULL_VALUE};
  /** Decoded contents for strings, or the exact JSON token for every other type. */
  std::string value;
};

/** Read-only JSON input that may span separately allocated buffers. */
class JsonInput {
 public:
  virtual ~JsonInput() = default;
  /** Return the number of available bytes. */
  virtual size_t size() const = 0;
  /** Read a byte at an offset strictly less than size(). */
  virtual char operator[](size_t offset) const = 0;
};

/** Validated field whose raw JSON token remains in the caller-owned input. */
struct JsonFieldView {
  /** Decoded field name. */
  std::string name;
  /** Kind of the referenced JSON token. */
  JsonValueType type{JsonValueType::NULL_VALUE};
  /** Token offset relative to the parsed input range. */
  size_t begin{0U};
  /** Token size, including quotes and escapes for strings. */
  size_t length{0U};
};

/** Validate an object range, rejecting duplicate fields, without copying its values. */
bool parse_json_object_views(const JsonInput &input, size_t offset, size_t length,
                             std::vector<JsonFieldView> *fields, std::string *error = nullptr);
/** Validate an array of objects and retain field offsets relative to the array range. */
bool parse_json_array_object_views(const JsonInput &input, size_t offset, size_t length,
                                   std::vector<std::vector<JsonFieldView>> *objects,
                                   std::string *error = nullptr);
/** Decode one complete JSON string token from the specified input range. */
bool parse_json_string_value(const JsonInput &input, size_t offset, size_t length,
                             std::string *value, std::string *error = nullptr);

void append_json_string(std::string *out, const char *value);
void append_json_pair(std::string *out, const char *key, const char *value, bool first = false);
bool has_json_key(const std::string &payload, const char *key);
std::string extract_json_string(const std::string &payload, const char *key);
std::string extract_json_number_token(const std::string &payload, const char *key);
bool decode_urlencoded_component(const std::string &encoded, std::string *decoded, std::string *error = nullptr);
std::string encode_urlencoded_component(const std::string &value);
bool parse_urlencoded_key_value_pairs(const std::string &payload,
                                      std::vector<std::pair<std::string, std::string>> *pairs,
                                      std::string *error = nullptr);
/** Parse and validate one complete JSON object, rejecting duplicate field names. */
bool parse_json_object_fields(const std::string &payload,
                              std::vector<JsonObjectField> *fields,
                              std::string *error = nullptr);
/** Parse a complete array of objects, rejecting invalid or non-object entries. */
bool parse_json_array_objects(const std::string &payload,
                              std::vector<std::vector<JsonObjectField>> *objects,
                              std::string *error = nullptr);
/** Parse a complete array containing only JSON strings. */
bool parse_json_array_strings(const std::string &payload, std::vector<std::string> *strings,
                              std::string *error = nullptr);
const JsonObjectField *find_json_object_field(const std::vector<JsonObjectField> &fields, const char *name);

}  // namespace espectre
