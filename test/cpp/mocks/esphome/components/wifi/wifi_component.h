/*
 * ESPectre - Mock ESPHome Wi-Fi Roaming Control
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace esphome::wifi {

class WiFiComponent {
 public:
  void request_roaming_suppression() { ++suppression_count_; }
  void release_roaming_suppression() {
    if (suppression_count_ != 0U) --suppression_count_;
  }
  bool roaming_allowed() const { return suppression_count_ == 0U; }

 private:
  unsigned suppression_count_{0U};
};

inline WiFiComponent *global_wifi_component = nullptr;

}  // namespace esphome::wifi
