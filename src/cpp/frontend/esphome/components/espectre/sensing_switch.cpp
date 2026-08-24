/*
 * ESPectre - Sensing Switch Component
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "sensing_switch.h"

#include "espectre.h"

namespace esphome {
namespace espectre_component {

void ESpectreSensingSwitch::dump_config() {
  LOG_SWITCH("", "ESPectre Sensing Enabled", this);
}

void ESpectreSensingSwitch::write_state(bool state) {
  if (this->parent_ != nullptr) {
    (void) this->parent_->set_sensing_runtime(state);
    this->republish_state();
  }
}

void ESpectreSensingSwitch::republish_state() {
  if (this->parent_ != nullptr) {
    this->publish_state(this->parent_->is_sensing_enabled());
  }
}

}  // namespace espectre_component
}  // namespace esphome
