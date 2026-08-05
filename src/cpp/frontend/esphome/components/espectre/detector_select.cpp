/*
 * ESPectre - Detector Select
 *
 * ESPHome select entity for runtime detector algorithm choice.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "detector_select.h"

#include "espectre.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre_component {

void ESpectreDetectorSelect::dump_config() { LOG_SELECT("", "ESPectre Detector", this); }

void ESpectreDetectorSelect::control(const std::string &value) {
  if (this->parent_ != nullptr) {
    this->parent_->set_detection_algorithm_runtime(value);
  }
}

void ESpectreDetectorSelect::republish_state() {
  if (this->parent_ != nullptr) {
    this->publish_state(::espectre::detection_algorithm_name(
        this->parent_->runtime_.config().detection_algorithm));
  }
}

}  // namespace espectre_component
}  // namespace esphome
