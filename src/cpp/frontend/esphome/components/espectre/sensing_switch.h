/*
 * ESPectre - Sensing Switch Component
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/components/switch/switch.h"
#include "esphome/core/component.h"

namespace esphome {
namespace espectre_component {

class ESpectreComponent;

class ESpectreSensingSwitch : public switch_::Switch, public Component {
 public:
  void dump_config() override;
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }
  void republish_state();

 protected:
  void write_state(bool state) override;

  ESpectreComponent *parent_{nullptr};
};

}  // namespace espectre_component
}  // namespace esphome
