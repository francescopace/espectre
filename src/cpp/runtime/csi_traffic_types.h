/*
 * ESPectre - CSI Traffic Types
 *
 * Platform-agnostic CSI traffic mode shared between the runtime interface
 * and the ESP-IDF traffic service implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

namespace espectre {

enum class CsiTrafficMode {
  INTERNAL,
  EXTERNAL,
  PACING,
  DISABLED,
};

}  // namespace espectre
