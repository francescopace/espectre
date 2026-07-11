#include "protocol_json.h"

#include <cctype>
#include <cstdint>

namespace espectre {

void append_json_string(std::string *out, const char *value) {
  if (out == nullptr) {
    return;
  }
  out->push_back('"');
  if (value != nullptr) {
    for (const char *p = value; *p != '\0'; ++p) {
      switch (*p) {
        case '"':
          out->append("\\\"");
          break;
        case '\\':
          out->append("\\\\");
          break;
        case '\n':
          out->append("\\n");
          break;
        case '\r':
          out->append("\\r");
          break;
        case '\t':
          out->append("\\t");
          break;
        default:
          out->push_back(*p);
          break;
      }
    }
  }
  out->push_back('"');
}

std::string json_pair_string(const char *key, const char *value, bool first) {
  std::string out;
  if (!first) {
    out.append(",");
  }
  append_json_string(&out, key);
  out.append(":");
  append_json_string(&out, value);
  return out;
}

std::string extract_json_string(const std::string &payload, const char *key) {
  if (key == nullptr || key[0] == '\0') {
    return {};
  }
  const std::string needle = std::string("\"") + key + "\"";
  const size_t key_pos = payload.find(needle);
  if (key_pos == std::string::npos) {
    return {};
  }
  const size_t colon = payload.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return {};
  }
  const size_t first_quote = payload.find('"', colon + 1);
  if (first_quote == std::string::npos) {
    return {};
  }
  std::string value;
  bool escaped = false;
  for (size_t i = first_quote + 1; i < payload.size(); ++i) {
    const char ch = payload[i];
    if (escaped) {
      value.push_back(ch);
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"') {
      return value;
    }
    value.push_back(ch);
  }
  return {};
}

std::string extract_json_number_token(const std::string &payload, const char *key) {
  if (key == nullptr || key[0] == '\0') {
    return {};
  }
  const std::string needle = std::string("\"") + key + "\"";
  const size_t key_pos = payload.find(needle);
  if (key_pos == std::string::npos) {
    return {};
  }
  const size_t colon = payload.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return {};
  }
  size_t begin = payload.find_first_not_of(" \t\r\n", colon + 1);
  if (begin == std::string::npos) {
    return {};
  }
  size_t end = begin;
  while (end < payload.size()) {
    const char ch = payload[end];
    if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
      ++end;
      continue;
    }
    break;
  }
  return payload.substr(begin, end - begin);
}

bool decode_urlencoded_component(const std::string &encoded, std::string *decoded, std::string *error) {
  if (decoded == nullptr) {
    if (error != nullptr) {
      *error = "decoded output is required";
    }
    return false;
  }
  decoded->clear();
  decoded->reserve(encoded.size());
  for (size_t i = 0; i < encoded.size(); ++i) {
    const char ch = encoded[i];
    if (ch == '+') {
      decoded->push_back(' ');
      continue;
    }
    if (ch != '%') {
      decoded->push_back(ch);
      continue;
    }
    if (i + 2 >= encoded.size()) {
      if (error != nullptr) {
        *error = "truncated escape sequence";
      }
      return false;
    }
    const char hi = encoded[i + 1];
    const char lo = encoded[i + 2];
    if (!std::isxdigit(static_cast<unsigned char>(hi)) || !std::isxdigit(static_cast<unsigned char>(lo))) {
      if (error != nullptr) {
        *error = "invalid escape sequence";
      }
      return false;
    }
    const auto decode_hex = [](char value) -> uint8_t {
      if (value >= '0' && value <= '9') {
        return static_cast<uint8_t>(value - '0');
      }
      if (value >= 'a' && value <= 'f') {
        return static_cast<uint8_t>(10 + (value - 'a'));
      }
      return static_cast<uint8_t>(10 + (value - 'A'));
    };
    const uint8_t decoded_byte = static_cast<uint8_t>((decode_hex(hi) << 4U) | decode_hex(lo));
    decoded->push_back(static_cast<char>(decoded_byte));
    i += 2;
  }
  return true;
}

bool parse_urlencoded_key_value_pairs(const std::string &payload,
                                      std::vector<std::pair<std::string, std::string>> *pairs,
                                      std::string *error) {
  if (pairs == nullptr) {
    if (error != nullptr) {
      *error = "pairs output is required";
    }
    return false;
  }
  pairs->clear();
  if (payload.empty()) {
    if (error != nullptr) {
      *error = "missing payload";
    }
    return false;
  }

  size_t begin = 0;
  while (begin <= payload.size()) {
    const size_t end = payload.find('&', begin);
    const std::string token = payload.substr(begin, end == std::string::npos ? std::string::npos : end - begin);
    if (token.empty()) {
      if (error != nullptr) {
        *error = "empty key-value token";
      }
      return false;
    }
    const size_t eq = token.find('=');
    if (eq == std::string::npos || eq == 0) {
      if (error != nullptr) {
        *error = "invalid key-value token";
      }
      return false;
    }
    std::string key;
    std::string value;
    if (!decode_urlencoded_component(token.substr(0, eq), &key, error) ||
        !decode_urlencoded_component(token.substr(eq + 1), &value, error)) {
      return false;
    }
    if (key.empty()) {
      if (error != nullptr) {
        *error = "empty key";
      }
      return false;
    }
    pairs->emplace_back(std::move(key), std::move(value));
    if (end == std::string::npos) {
      break;
    }
    begin = end + 1;
  }
  return !pairs->empty();
}

}  // namespace espectre
