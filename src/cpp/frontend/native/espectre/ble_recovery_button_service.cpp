/*
 * ESPectre - Native BLE Recovery Button Service
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "ble_recovery_button_service.h"

#include <utility>

namespace espectre {

BleRecoveryButtonService::BleRecoveryButtonService(uint32_t hold_ms, RecoveryCallback callback)
    : hold_ms_(hold_ms), callback_(std::move(callback)) {}

void BleRecoveryButtonService::update(bool pressed, uint32_t now_ms) {
  if (!pressed) {
    tracking_press_ = false;
    fired_for_press_ = false;
    return;
  }
  if (!tracking_press_) {
    tracking_press_ = true;
    pressed_at_ms_ = now_ms;
    return;
  }
  if (fired_for_press_ || static_cast<uint32_t>(now_ms - pressed_at_ms_) < hold_ms_) {
    return;
  }
  fired_for_press_ = true;
  if (callback_) {
    callback_();
  }
}

}  // namespace espectre
