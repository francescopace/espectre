/*
 * ESPectre - Detector Public Types
 *
 * Stable detector state and probability-threshold constants shared by the
 * core-only and full-runtime SDK surfaces.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

namespace espectre {

/** Debounced detector state. */
enum class MotionState {
  IDLE,
  MOTION,
};

constexpr float LIGHTWEIGHT_DEFAULT_THRESHOLD = 0.6621854538596202f;
constexpr float LIGHTWEIGHT_MIN_THRESHOLD = 0.0f;
constexpr float LIGHTWEIGHT_MAX_THRESHOLD = 1.0f;
constexpr float LIGHTWEIGHT_STARTUP_THRESHOLD_FACTOR = 1.0f;

constexpr float HIGH_ACCURACY_DEFAULT_THRESHOLD = 0.5f;
constexpr float HIGH_ACCURACY_MIN_THRESHOLD = 0.0f;
constexpr float HIGH_ACCURACY_MAX_THRESHOLD = 1.0f;
constexpr float HIGH_ACCURACY_METRIC_SCALE = 1.0f;

}  // namespace espectre
