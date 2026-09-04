/*
 * ESPectre - Motion Hits Number
 *
 * ESPHome number entity for runtime motion-on and motion-off hit counts.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "motion_hits_number.h"

#include <cmath>

#include "espectre.h"
#include "esphome/core/log.h"

namespace esphome {
namespace espectre_component {

namespace {

const char *const TAG_MOTION_HITS = "espectre.motion_hits";

}  // namespace

void ESpectreMotionHitsNumber::dump_config() {
  LOG_NUMBER("", this->motion_on_ ? "ESPectre Motion On Hits" : "ESPectre Motion Off Hits", this);
}

void ESpectreMotionHitsNumber::control(float value) {
  if (this->parent_ == nullptr) {
    return;
  }
  const uint8_t rounded = static_cast<uint8_t>(std::lround(value));
  const uint8_t motion_on_hits = this->motion_on_ ? rounded : this->parent_->get_motion_on_hits();
  const uint8_t motion_off_hits = this->motion_on_ ? this->parent_->get_motion_off_hits() : rounded;
  if (!this->parent_->set_motion_hits_runtime(motion_on_hits, motion_off_hits)) this->republish_state();
}

void ESpectreMotionHitsNumber::republish_state() {
  if (this->parent_ == nullptr) {
    return;
  }
  const float current = this->motion_on_ ? this->parent_->get_motion_on_hits() : this->parent_->get_motion_off_hits();
  this->publish_state(current);
  ESP_LOGD(TAG_MOTION_HITS, "%s re-published to HA: %.0f", this->motion_on_ ? "Motion on hits" : "Motion off hits", current);
}

}  // namespace espectre_component
}  // namespace esphome
