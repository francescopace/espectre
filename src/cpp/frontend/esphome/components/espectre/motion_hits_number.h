/*
 * ESPectre - Motion Hits Number
 *
 * ESPHome number entity for runtime motion-on and motion-off hit counts.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/components/number/number.h"
#include "esphome/core/component.h"

namespace esphome {
namespace espectre_component {

class ESpectreComponent;

class ESpectreMotionHitsNumber : public number::Number, public Component {
 public:
  void dump_config() override;
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }
  void set_motion_on(bool motion_on) { this->motion_on_ = motion_on; }
  void republish_state();

 protected:
  void control(float value) override;

  ESpectreComponent *parent_{nullptr};
  bool motion_on_{true};
};

}  // namespace espectre_component
}  // namespace esphome
