/*
 * ESPectre - Mock ESPHome HAL
 *
 * Minimal mock for esphome/core/hal.h used in testing.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

namespace esphome {

inline uint32_t mock_millis_value = 0;

// Mock millis() function
inline uint32_t millis() { return mock_millis_value; }

// Mock micros() function
inline uint32_t micros() { return mock_millis_value * 1000; }

inline void set_mock_millis(uint32_t value) { mock_millis_value = value; }
inline void advance_mock_millis(uint32_t delta) { mock_millis_value += delta; }
inline void reset_mock_millis() { mock_millis_value = 0; }

// Mock delay() function
inline void delay(uint32_t ms) { (void)ms; }

// Mock delayMicroseconds() function
inline void delayMicroseconds(uint32_t us) { (void)us; }

// Mock yield() function
inline void yield() {}

}  // namespace esphome

