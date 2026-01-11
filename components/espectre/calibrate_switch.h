/*
 * ESPectre - Calibrate Switch Component
 * 
 * ESPHome switch component for triggering band recalibration from Home Assistant.
 * Switch is ON during calibration, OFF when idle.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esphome/core/component.h"
#include "esphome/components/switch/switch.h"

namespace esphome {
namespace espectre {

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

}  // namespace espectre
}  // namespace esphome

