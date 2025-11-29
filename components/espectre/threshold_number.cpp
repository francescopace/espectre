/*
 * ESPectre - Threshold Number Component Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "threshold_number.h"
#include "espectre_component.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *const TAG_THRESHOLD = "espectre.threshold";

void ESpectreThresholdNumber::setup() {
  // Initialize with current threshold value from parent
  if (this->parent_ != nullptr) {
    float current = this->parent_->get_threshold();
    this->publish_state(current);
    ESP_LOGI(TAG_THRESHOLD, "Threshold number initialized: %.2f", current);
  }
}

void ESpectreThresholdNumber::dump_config() {
  LOG_NUMBER("", "ESPectre Threshold", this);
}

void ESpectreThresholdNumber::control(float value) {
  // Called when user changes value from HA
  if (this->parent_ != nullptr) {
    this->parent_->set_threshold_runtime(value);
    this->publish_state(value);
    ESP_LOGI(TAG_THRESHOLD, "Threshold changed from HA: %.2f", value);
  }
}

}  // namespace espectre
}  // namespace esphome

