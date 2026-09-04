/*
 * ESPectre - Threshold Number Component
 *
 * ESPHome number component for viewing and updating the runtime detection
 * threshold.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/core/component.h"
#include "esphome/components/number/number.h"
#include "runtime_sensing_schema.h"

namespace esphome {
namespace espectre_component {

// Forward declaration
class ESpectreComponent;

class ESpectreThresholdNumber : public number::Number, public Component {
 public:
  void setup() override;
  void dump_config() override;
  
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }
  
  // Re-publish current threshold value to Home Assistant
  // Called when API connection is ready to ensure HA receives the saved value
  void republish_state();
  void update_detector_range(::espectre::DetectionAlgorithm algorithm);
  
 protected:
  void control(float value) override;
  
  ESpectreComponent *parent_{nullptr};
};

}  // namespace espectre_component
}  // namespace esphome
