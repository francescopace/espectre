/*
 * ESPectre - Protocol JSON
 *
 * JSON helpers for shared ESPectre protocol payloads.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <string>
#include <utility>
#include <vector>

namespace espectre {

void append_json_string(std::string *out, const char *value);
void append_json_pair(std::string *out, const char *key, const char *value, bool first = false);
bool has_json_key(const std::string &payload, const char *key);
std::string extract_json_string(const std::string &payload, const char *key);
std::string extract_json_number_token(const std::string &payload, const char *key);
bool decode_urlencoded_component(const std::string &encoded, std::string *decoded, std::string *error = nullptr);
bool parse_urlencoded_key_value_pairs(const std::string &payload,
                                      std::vector<std::pair<std::string, std::string>> *pairs,
                                      std::string *error = nullptr);

}  // namespace espectre
