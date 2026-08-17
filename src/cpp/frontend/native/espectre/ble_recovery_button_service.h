/*
 * ESPectre - Native BLE Recovery Button Service
 *
 * Detects a non-blocking long press and requests Native BLE setup mode.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <functional>

namespace espectre {

class BleRecoveryButtonService {
 public:
  using RecoveryCallback = std::function<void()>;

  explicit BleRecoveryButtonService(uint32_t hold_ms, RecoveryCallback callback);

  void update(bool pressed, uint32_t now_ms);

 private:
  uint32_t hold_ms_{0U};
  RecoveryCallback callback_{};
  uint32_t pressed_at_ms_{0U};
  bool tracking_press_{false};
  bool fired_for_press_{false};
};

}  // namespace espectre
