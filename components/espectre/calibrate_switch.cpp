/*
 * ESPectre - Calibrate Switch Component Implementation
 * 
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "calibrate_switch.h"
#include "espectre.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre {

static const char *const TAG_CALIBRATE = "espectre.calibrate";

void ESpectreCalibrateSwitch::setup() {
  // Don't publish initial state here.
  // The parent will call set_calibrating(true) when calibration starts,
  // and set_calibrating(false) when it completes.
  // This ensures the switch reflects the real calibration state.
}

void ESpectreCalibrateSwitch::dump_config() {
  LOG_SWITCH("", "ESPectre Calibrate", this);
}

void ESpectreCalibrateSwitch::set_calibrating(bool calibrating) {
  // Update switch state and publish to Home Assistant
  this->publish_state(calibrating);
  
  // Note: ESPHome doesn't support runtime disable of switches.
  // User clicks during calibration are ignored in write_state().
}

void ESpectreCalibrateSwitch::write_state(bool state) {
  if (this->parent_ == nullptr) {
    return;
  }
  
  // During calibration, ignore all user interactions
  // The switch will auto-update when calibration completes
  if (this->parent_->is_calibrating()) {
    // Re-publish current state (ON) to reject the user's click
    this->publish_state(true);
    ESP_LOGD(TAG_CALIBRATE, "Calibration in progress - ignoring switch interaction");
    return;
  }
  
  if (state) {
    // User wants to turn ON (start calibration)
    this->parent_->trigger_recalibration();
    // Don't publish state here - parent will call set_calibrating(true)
  } else {
    // User wants to turn OFF while not calibrating - just acknowledge
    this->publish_state(false);
  }
}

}  // namespace espectre
}  // namespace esphome

