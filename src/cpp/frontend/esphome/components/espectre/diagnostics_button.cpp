/*
 * ESPectre - Diagnostics Button Component
 *
 * Publishes the latest cached runtime diagnostics on demand.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "diagnostics_button.h"

#include "espectre.h"

namespace esphome {
namespace espectre_component {

void ESpectreDiagnosticsButton::dump_config() {
  LOG_BUTTON("", "ESPectre Diagnostics Refresh", this);
}

void ESpectreDiagnosticsButton::press_action() {
  if (this->parent_ != nullptr) {
    this->parent_->publish_diagnostics_on_demand();
  }
}

}  // namespace espectre_component
}  // namespace esphome
