/*
 * ESPectre - Detector Limits
 *
 * Shared detector window size and calibration buffer limits.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>

namespace espectre {

constexpr uint16_t DETECTOR_DEFAULT_WINDOW_SIZE = 100;
constexpr uint16_t DETECTOR_MIN_WINDOW_SIZE = 10;
constexpr uint16_t DETECTOR_MAX_WINDOW_SIZE = 200;

constexpr uint16_t CALIBRATION_NUM_WINDOWS = 10;
constexpr uint16_t CALIBRATION_DEFAULT_BUFFER_SIZE =
    DETECTOR_DEFAULT_WINDOW_SIZE * CALIBRATION_NUM_WINDOWS;

}  // namespace espectre
