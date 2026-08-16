/*
 * ESPectre - Home Assistant Intensity Mapping
 *
 * Converts the detector movement metric and threshold into the shared
 * Home Assistant intensity percent used by ESPHome, Native MQTT, and
 * Micro-ESPectre MQTT.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace espectre {

/**
 * Map movement and threshold onto a 0-100 intensity percent.
 *
 * 50% is the decision threshold; 100% is twice the threshold.
 */
inline float ha_intensity_percent(float movement_metric, float threshold) {
  if (!(threshold > 0.0f)) {
    return 0.0f;
  }
  const float intensity = (movement_metric / threshold) * 50.0f;
  if (intensity > 100.0f) {
    return 100.0f;
  }
  if (intensity < 0.0f) {
    return 0.0f;
  }
  return intensity;
}

}  // namespace espectre
