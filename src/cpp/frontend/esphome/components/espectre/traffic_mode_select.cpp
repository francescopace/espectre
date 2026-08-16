/*
 * ESPectre - Traffic Mode Select
 *
 * ESPHome select entity for runtime CSI traffic and generator mode choice.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "traffic_mode_select.h"

#include "espectre.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre_component {

void ESpectreTrafficModeSelect::dump_config() {
  LOG_SELECT("", this->csi_traffic_mode_ ? "ESPectre CSI Traffic Mode" : "ESPectre Traffic Generator Mode", this);
}

void ESpectreTrafficModeSelect::control(const std::string &value) {
  if (this->parent_ == nullptr) {
    return;
  }
  if (this->csi_traffic_mode_) {
    this->parent_->set_csi_traffic_mode_runtime(value);
  } else {
    this->parent_->set_traffic_generator_mode_runtime(value);
  }
}

void ESpectreTrafficModeSelect::republish_state() {
  if (this->parent_ == nullptr) {
    return;
  }
  if (this->csi_traffic_mode_) {
    this->publish_state(::espectre::csi_traffic_mode_name(this->parent_->runtime_.config().csi_traffic_mode));
  } else {
    this->publish_state(::espectre::traffic_mode_name(this->parent_->runtime_.config().traffic_generator_mode));
  }
}

}  // namespace espectre_component
}  // namespace esphome
