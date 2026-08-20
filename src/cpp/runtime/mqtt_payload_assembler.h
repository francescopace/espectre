/*
 * ESPectre - MQTT Payload Assembler
 *
 * Assembles MQTT payloads for shared protocol publications.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <cstddef>
#include <cstring>
#include <string_view>

namespace espectre {

class MqttPayloadAssembler {
 public:
  enum class Result {
    INCOMPLETE,
    COMPLETE,
    INVALID,
  };

  static constexpr size_t MAX_PAYLOAD_SIZE = 2048U;

  Result append(const char *data, size_t data_len, size_t total_len, size_t offset) {
    if (data == nullptr || data_len == 0U || total_len == 0U || total_len > MAX_PAYLOAD_SIZE ||
        offset > total_len || data_len > total_len - offset) {
      reset();
      return Result::INVALID;
    }

    if (offset == 0U) {
      std::memcpy(payload_.data(), data, data_len);
      payload_size_ = data_len;
      expected_total_len_ = total_len;
      next_offset_ = data_len;
    } else if (expected_total_len_ != total_len || offset != next_offset_) {
      reset();
      return Result::INVALID;
    } else {
      std::memcpy(payload_.data() + payload_size_, data, data_len);
      payload_size_ += data_len;
      next_offset_ += data_len;
    }

    if (next_offset_ < expected_total_len_) {
      return Result::INCOMPLETE;
    }
    return next_offset_ == expected_total_len_ ? Result::COMPLETE : Result::INVALID;
  }

  std::string_view payload() const { return {payload_.data(), payload_size_}; }

  void reset() {
    payload_size_ = 0U;
    expected_total_len_ = 0U;
    next_offset_ = 0U;
  }

 private:
  std::array<char, MAX_PAYLOAD_SIZE> payload_{};
  size_t payload_size_{0U};
  size_t expected_total_len_{0U};
  size_t next_offset_{0U};
};

}  // namespace espectre
