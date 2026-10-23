/*
 * ESPectre - Sample Spacing Select
 *
 * ESPHome select entity for the runtime CSI sample spacing rule.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "sample_spacing_select.h"

#include "espectre.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre_component {

void ESpectreSampleSpacingSelect::dump_config() {
  LOG_SELECT("", "ESPectre CSI Sample Spacing", this);
}

void ESpectreSampleSpacingSelect::control(const std::string &value) {
  if (this->parent_ == nullptr) {
    return;
  }
  if (!this->parent_->set_csi_sample_spacing_runtime(value)) this->republish_state();
}

void ESpectreSampleSpacingSelect::republish_state() {
  if (this->parent_ == nullptr) {
    return;
  }
  this->publish_state(::espectre::csi_sample_spacing_name(this->parent_->runtime_.config().csi_sample_spacing));
}

}  // namespace espectre_component
}  // namespace esphome
