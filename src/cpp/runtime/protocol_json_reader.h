/*
 * ESPectre - Internal Protocol JSON Reader
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <algorithm>
#include <cctype>
#include "protocol_json.h"

namespace espectre {
namespace detail {

template <typename Input>
class JsonInputView {
 public:
  explicit JsonInputView(const Input &input, size_t offset = 0U, size_t length = std::string::npos)
      : input_(input), offset_(std::min(offset, input.size())),
        size_(std::min(length, input.size() - offset_)) {}
  size_t size() const { return size_; }
  char operator[](size_t offset) const { return input_[offset_ + offset]; }
  std::string substr(size_t begin, size_t length) const {
    std::string result;
    result.reserve(length);
    for (size_t i = 0U; i < length; ++i) result.push_back((*this)[begin + i]);
    return result;
  }
  int compare(size_t begin, size_t length, const std::string &expected) const {
    if (begin > size_ || length > size_ - begin || length != expected.size()) return -1;
    for (size_t i = 0U; i < length; ++i) {
      if ((*this)[begin + i] != expected[i]) return -1;
    }
    return 0;
  }
 private:
  const Input &input_;
  size_t offset_;
  size_t size_;
};

template <typename Input>
class JsonReader {
 public:
  explicit JsonReader(const Input &input, size_t offset = 0U, size_t length = std::string::npos)
      : input_(input, offset, length) {}

  bool parse_object(std::vector<JsonObjectField> *fields, std::string *error) {
    error_ = error;
    if (fields == nullptr) {
      return fail_("fields output is required");
    }
    fields->clear();
    skip_space_();
    if (!parse_object_(fields, 0U)) {
      return false;
    }
    skip_space_();
    if (position_ != input_.size()) {
      return fail_("unexpected data after JSON object");
    }
    return true;
  }

  bool parse_array_strings(std::vector<std::string> *strings, std::string *error) {
    error_ = error;
    if (strings == nullptr) return fail_("strings output is required");
    strings->clear();
    skip_space_();
    if (!consume_('[')) return fail_("expected JSON array");
    skip_space_();
    if (!consume_(']')) {
      while (true) {
        std::string value;
        if (!parse_string_(&value)) return false;
        strings->push_back(std::move(value));
        skip_space_();
        if (consume_(']')) break;
        if (!consume_(',')) return fail_("expected comma in JSON array");
        skip_space_();
      }
    }
    skip_space_();
    if (position_ != input_.size()) {
      return fail_("unexpected data after JSON array");
    }
    return true;
  }

  bool parse_object_views(std::vector<JsonFieldView> *fields, std::string *error) {
    error_ = error;
    fields->clear();
    skip_space_();
    if (!parse_object_(nullptr, 0U, fields)) return false;
    skip_space_();
    if (position_ != input_.size()) {
      return fail_("unexpected data after JSON object");
    }
    return true;
  }

  bool parse_array_object_views(std::vector<std::vector<JsonFieldView>> *objects, std::string *error) {
    error_ = error;
    objects->clear();
    skip_space_();
    if (!parse_array_(0U, objects)) return false;
    skip_space_();
    if (position_ != input_.size()) {
      return fail_("unexpected data after JSON array");
    }
    return true;
  }

  bool parse_string_value(std::string *value, std::string *error) {
    error_ = error;
    skip_space_();
    if (!parse_string_(value)) return false;
    skip_space_();
    if (position_ != input_.size()) {
      return fail_("unexpected data after JSON string");
    }
    return true;
  }

 private:
  static bool hex_value_(char ch, uint32_t *value) {
    if (value == nullptr) {
      return false;
    }
    if (ch >= '0' && ch <= '9') {
      *value = static_cast<uint32_t>(ch - '0');
      return true;
    }
    if (ch >= 'a' && ch <= 'f') {
      *value = static_cast<uint32_t>(10 + ch - 'a');
      return true;
    }
    if (ch >= 'A' && ch <= 'F') {
      *value = static_cast<uint32_t>(10 + ch - 'A');
      return true;
    }
    return false;
  }

  static bool append_utf8_(uint32_t codepoint, std::string *out) {
    if (out == nullptr || codepoint > 0x10FFFFU || (codepoint >= 0xD800U && codepoint <= 0xDFFFU)) {
      return false;
    }
    if (codepoint <= 0x7FU) {
      out->push_back(static_cast<char>(codepoint));
    } else if (codepoint <= 0x7FFU) {
      out->push_back(static_cast<char>(0xC0U | (codepoint >> 6U)));
      out->push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else if (codepoint <= 0xFFFFU) {
      out->push_back(static_cast<char>(0xE0U | (codepoint >> 12U)));
      out->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      out->push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    } else {
      out->push_back(static_cast<char>(0xF0U | (codepoint >> 18U)));
      out->push_back(static_cast<char>(0x80U | ((codepoint >> 12U) & 0x3FU)));
      out->push_back(static_cast<char>(0x80U | ((codepoint >> 6U) & 0x3FU)));
      out->push_back(static_cast<char>(0x80U | (codepoint & 0x3FU)));
    }
    return true;
  }

  bool fail_(const char *message) {
    if (error_ != nullptr) {
      *error_ = message != nullptr ? message : "invalid JSON";
    }
    return false;
  }

  void skip_space_() {
    while (position_ < input_.size()) {
      const char ch = input_[position_];
      if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') {
        break;
      }
      ++position_;
    }
  }

  bool consume_(char expected) {
    if (position_ >= input_.size() || input_[position_] != expected) {
      return false;
    }
    ++position_;
    return true;
  }

  bool parse_hex_quad_(uint32_t *value) {
    if (value == nullptr || position_ + 4U > input_.size()) {
      return false;
    }
    uint32_t parsed = 0U;
    for (size_t i = 0U; i < 4U; ++i) {
      uint32_t nibble = 0U;
      if (!hex_value_(input_[position_ + i], &nibble)) {
        return false;
      }
      parsed = (parsed << 4U) | nibble;
    }
    position_ += 4U;
    *value = parsed;
    return true;
  }

  bool parse_string_(std::string *value) {
    if (!consume_('"')) {
      return fail_("expected JSON string");
    }
    std::string parsed;
    while (position_ < input_.size()) {
      const unsigned char ch = static_cast<unsigned char>(input_[position_++]);
      if (ch == '"') {
        if (value != nullptr) {
          *value = std::move(parsed);
        }
        return true;
      }
      if (ch < 0x20U) {
        return fail_("unescaped control character in JSON string");
      }
      if (ch != '\\') {
        if (value != nullptr) parsed.push_back(static_cast<char>(ch));
        continue;
      }
      if (position_ >= input_.size()) {
        return fail_("truncated JSON string escape");
      }
      const char escape = input_[position_++];
      switch (escape) {
        case '"':
        case '\\':
        case '/':
          if (value != nullptr) parsed.push_back(escape);
          break;
        case 'b':
          if (value != nullptr) parsed.push_back('\b');
          break;
        case 'f':
          if (value != nullptr) parsed.push_back('\f');
          break;
        case 'n':
          if (value != nullptr) parsed.push_back('\n');
          break;
        case 'r':
          if (value != nullptr) parsed.push_back('\r');
          break;
        case 't':
          if (value != nullptr) parsed.push_back('\t');
          break;
        case 'u': {
          uint32_t codepoint = 0U;
          if (!parse_hex_quad_(&codepoint)) {
            return fail_("invalid JSON unicode escape");
          }
          if (codepoint >= 0xD800U && codepoint <= 0xDBFFU) {
            if (position_ + 6U > input_.size() || input_[position_] != '\\' || input_[position_ + 1U] != 'u') {
              return fail_("missing JSON low surrogate");
            }
            position_ += 2U;
            uint32_t low = 0U;
            if (!parse_hex_quad_(&low) || low < 0xDC00U || low > 0xDFFFU) {
              return fail_("invalid JSON low surrogate");
            }
            codepoint = 0x10000U + ((codepoint - 0xD800U) << 10U) + (low - 0xDC00U);
          } else if (codepoint >= 0xDC00U && codepoint <= 0xDFFFU) {
            return fail_("unexpected JSON low surrogate");
          }
          if (value != nullptr && !append_utf8_(codepoint, &parsed)) {
            return fail_("invalid JSON unicode codepoint");
          }
          break;
        }
        default:
          return fail_("invalid JSON string escape");
      }
    }
    return fail_("unterminated JSON string");
  }

  bool parse_number_() {
    const size_t begin = position_;
    if (position_ < input_.size() && input_[position_] == '-') {
      ++position_;
    }
    if (position_ >= input_.size()) {
      return fail_("invalid JSON number");
    }
    if (input_[position_] == '0') {
      ++position_;
      if (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        return fail_("invalid leading zero in JSON number");
      }
    } else if (input_[position_] >= '1' && input_[position_] <= '9') {
      while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
      }
    } else {
      return fail_("invalid JSON number");
    }
    if (position_ < input_.size() && input_[position_] == '.') {
      ++position_;
      const size_t fraction = position_;
      while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
      }
      if (position_ == fraction) {
        return fail_("invalid JSON fraction");
      }
    }
    if (position_ < input_.size() && (input_[position_] == 'e' || input_[position_] == 'E')) {
      ++position_;
      if (position_ < input_.size() && (input_[position_] == '+' || input_[position_] == '-')) {
        ++position_;
      }
      const size_t exponent = position_;
      while (position_ < input_.size() && std::isdigit(static_cast<unsigned char>(input_[position_]))) {
        ++position_;
      }
      if (position_ == exponent) {
        return fail_("invalid JSON exponent");
      }
    }
    return position_ > begin;
  }

  bool parse_literal_(const char *literal) {
    if (literal == nullptr) {
      return false;
    }
    const std::string expected(literal);
    if (input_.compare(position_, expected.size(), expected) != 0) {
      return fail_("invalid JSON literal");
    }
    position_ += expected.size();
    return true;
  }

  bool parse_array_(size_t depth, std::vector<std::vector<JsonFieldView>> *views = nullptr) {
    if (depth > 16U) {
      return fail_("JSON nesting limit exceeded");
    }
    if (!consume_('[')) {
      return fail_("expected JSON array");
    }
    skip_space_();
    if (consume_(']')) {
      return true;
    }
    while (true) {
      if (views != nullptr) {
        std::vector<JsonFieldView> object_views;
        if (!parse_object_(nullptr, depth + 1U, &object_views)) return false;
        views->push_back(std::move(object_views));
      } else if (!parse_value_(nullptr, nullptr, depth + 1U)) {
        return false;
      }
      skip_space_();
      if (consume_(']')) {
        return true;
      }
      if (!consume_(',')) {
        return fail_("expected comma in JSON array");
      }
      skip_space_();
    }
  }

  bool parse_object_(std::vector<JsonObjectField> *fields, size_t depth,
                     std::vector<JsonFieldView> *views = nullptr) {
    if (depth > 16U) {
      return fail_("JSON nesting limit exceeded");
    }
    if (!consume_('{')) {
      return fail_("expected JSON object");
    }
    skip_space_();
    if (consume_('}')) {
      return true;
    }
    while (true) {
      std::string name;
      if (!parse_string_(&name)) {
        return false;
      }
      if (fields != nullptr) {
        for (const auto &field : *fields) {
          if (field.name == name) {
            return fail_("duplicate JSON object field");
          }
        }
      }
      if (views != nullptr) {
        for (const auto &field : *views) {
          if (field.name == name) return fail_("duplicate JSON object field");
        }
      }
      skip_space_();
      if (!consume_(':')) {
        return fail_("expected colon in JSON object");
      }
      skip_space_();
      JsonObjectField field;
      field.name = std::move(name);
      const size_t value_begin = position_;
      if (!parse_value_(fields != nullptr || views != nullptr ? &field.type : nullptr,
                        fields != nullptr ? &field.value : nullptr,
                        depth + 1U)) {
        return false;
      }
      if (views != nullptr) {
        views->push_back({field.name, field.type, value_begin, position_ - value_begin});
      }
      if (fields != nullptr) {
        fields->push_back(std::move(field));
      }
      skip_space_();
      if (consume_('}')) {
        return true;
      }
      if (!consume_(',')) {
        return fail_("expected comma in JSON object");
      }
      skip_space_();
    }
  }

  bool parse_value_(JsonValueType *type, std::string *value, size_t depth) {
    if (position_ >= input_.size()) {
      return fail_("missing JSON value");
    }
    const size_t begin = position_;
    const char ch = input_[position_];
    if (ch == '"') {
      if (type != nullptr) {
        *type = JsonValueType::STRING;
      }
      return parse_string_(value);
    }
    bool accepted = false;
    JsonValueType parsed_type = JsonValueType::NULL_VALUE;
    if (ch == '{') {
      parsed_type = JsonValueType::OBJECT;
      accepted = parse_object_(nullptr, depth);
    } else if (ch == '[') {
      parsed_type = JsonValueType::ARRAY;
      accepted = parse_array_(depth);
    } else if (ch == 't') {
      parsed_type = JsonValueType::BOOLEAN;
      accepted = parse_literal_("true");
    } else if (ch == 'f') {
      parsed_type = JsonValueType::BOOLEAN;
      accepted = parse_literal_("false");
    } else if (ch == 'n') {
      parsed_type = JsonValueType::NULL_VALUE;
      accepted = parse_literal_("null");
    } else if (ch == '-' || std::isdigit(static_cast<unsigned char>(ch))) {
      parsed_type = JsonValueType::NUMBER;
      accepted = parse_number_();
    } else {
      return fail_("invalid JSON value");
    }
    if (!accepted) {
      return false;
    }
    if (type != nullptr) {
      *type = parsed_type;
    }
    if (value != nullptr) {
      *value = input_.substr(begin, position_ - begin);
    }
    return true;
  }

  JsonInputView<Input> input_;
  size_t position_{0U};
  std::string *error_{nullptr};
};

}  // namespace detail
}  // namespace espectre
