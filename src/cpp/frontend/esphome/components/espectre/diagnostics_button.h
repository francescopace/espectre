/*
 * ESPectre - Diagnostics Button Component
 *
 * Publishes the latest cached runtime diagnostics on demand.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include "esphome/components/button/button.h"
#include "esphome/core/component.h"

namespace esphome {
namespace espectre_component {

class ESpectreComponent;

class ESpectreDiagnosticsButton : public button::Button, public Component {
 public:
  void dump_config() override;
  void set_parent(ESpectreComponent *parent) { this->parent_ = parent; }

 protected:
  void press_action() override;

  ESpectreComponent *parent_{nullptr};
};

}  // namespace espectre_component
}  // namespace esphome
