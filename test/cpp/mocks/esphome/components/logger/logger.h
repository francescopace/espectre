/*
 * ESPectre - ESPHome Logger Mock
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/core/log.h"

#include <cstdint>

namespace esphome {
namespace logger {

class Logger {
 public:
  inline uint8_t level_for(const char *tag);
  void set_log_level(uint8_t level) { current_level_ = level; }

 protected:
  uint8_t current_level_{ESPHOME_LOG_LEVEL_INFO};
};

inline Logger *global_logger = nullptr;

}  // namespace logger
}  // namespace esphome
