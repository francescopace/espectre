/*
 * ESPectre - Traffic Mode Select
 *
 * ESPHome select entity for runtime CSI traffic and generator mode choice.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/components/select/select.h"
#include "esphome/core/component.h"

namespace esphome {
namespace espectre_component {

class ESpectreComponent;

class ESpectreTrafficModeSelect : public select::Select, public Component {
 public:
  void dump_config() override;
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }
  void set_csi_traffic_mode(bool value) { this->csi_traffic_mode_ = value; }
  void republish_state();

 protected:
  void control(const std::string &value) override;
  ESpectreComponent *parent_{nullptr};
  bool csi_traffic_mode_{false};
};

}  // namespace espectre_component
}  // namespace esphome
