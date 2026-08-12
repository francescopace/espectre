/*
 * ESPectre - Calibrate Switch Component
 *
 * ESPHome switch component that reflects calibration state and triggers
 * recalibration.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/core/component.h"
#include "esphome/components/switch/switch.h"

namespace esphome {
namespace espectre_component {

// Forward declaration
class ESpectreComponent;

class ESpectreCalibrateSwitch : public switch_::Switch, public Component {
 public:
  void setup() override;
  void dump_config() override;
  
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }
  
  /// Update switch state from parent (called when calibration starts/stops)
  void set_calibrating(bool calibrating);
  
 protected:
  void write_state(bool state) override;
  
  ESpectreComponent *parent_{nullptr};
};

}  // namespace espectre_component
}  // namespace esphome

