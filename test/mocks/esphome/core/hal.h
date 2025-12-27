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

// Mock millis() function
inline uint32_t millis() { return 0; }

// Mock micros() function
inline uint32_t micros() { return 0; }

// Mock delay() function
inline void delay(uint32_t ms) { (void)ms; }

// Mock delayMicroseconds() function
inline void delayMicroseconds(uint32_t us) { (void)us; }

// Mock yield() function
inline void yield() {}

}  // namespace esphome

