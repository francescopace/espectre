/*
 * ESPectre - Threshold Number Component Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "threshold_number.h"
#include "espectre.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *const TAG_THRESHOLD = "espectre.threshold";

void ESpectreThresholdNumber::setup() {
  // Don't publish state here - parent will call republish_state() when ready.
  // Calling publish_state() too early (before API/WiFi is connected) can cause
  // crashes or "unknown" state in Home Assistant.
  // The parent calls republish_state() on first sensor update (after API is connected).
}

void ESpectreThresholdNumber::dump_config() {
  LOG_NUMBER("", "ESPectre Threshold", this);
}

void ESpectreThresholdNumber::control(float value) {
  // Called when user changes value from HA
  // set_threshold_runtime handles everything: update, save, and publish
  if (this->parent_ != nullptr) {
    this->parent_->set_threshold_runtime(value);
  }
}

void ESpectreThresholdNumber::republish_state() {
  // Re-publish current threshold to Home Assistant
  // This ensures HA receives the saved value after API connection is established
  if (this->parent_ != nullptr) {
    float current = this->parent_->get_threshold();
    this->publish_state(current);
    ESP_LOGI(TAG_THRESHOLD, "Threshold re-published to HA: %.2f", current);
  }
}

}  // namespace espectre
}  // namespace esphome

