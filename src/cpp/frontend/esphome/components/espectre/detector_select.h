/*
 * ESPectre - Detector Select
 *
 * ESPHome select entity for runtime detector algorithm choice.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "esphome/components/select/select.h"
#include "esphome/core/component.h"

namespace esphome {
namespace espectre_component {

class ESpectreComponent;

class ESpectreDetectorSelect : public select::Select, public Component {
 public:
  void dump_config() override;
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }
  void republish_state();

 protected:
  void control(const std::string &value) override;
  ESpectreComponent *parent_{nullptr};
};

}  // namespace espectre_component
}  // namespace esphome
