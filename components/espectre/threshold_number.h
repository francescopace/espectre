/*
 * ESPectre - Threshold Number Component
 * 
 * ESPHome number component for adjusting motion detection threshold from Home Assistant.
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include "esphome/core/component.h"
#include "esphome/components/number/number.h"

namespace esphome {
namespace espectre {

// Forward declaration
class ESpectreComponent;

class ESpectreThresholdNumber : public number::Number, public Component {
 public:
  void setup() override;
  void dump_config() override;
  
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }
  
 protected:
  void control(float value) override;
  
  ESpectreComponent *parent_{nullptr};
};

}  // namespace espectre
}  // namespace esphome

