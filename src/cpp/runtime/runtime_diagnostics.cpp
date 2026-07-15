/*
 * ESPectre - Runtime Diagnostics
 *
 * Runtime diagnostics snapshot helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "runtime_diagnostics.h"

#include <cstdio>

#include "runtime_config_utils.h"

namespace espectre {

void visit_runtime_diagnostics(const RuntimeConfig &config,
                               const RuntimeSnapshot &snapshot,
                               runtime_diagnostic_visitor_t visitor) {
  if (!visitor) {
    return;
  }

  char value[64];

  std::snprintf(value, sizeof(value), "%.6f (%s)", snapshot.threshold, threshold_mode_name(config.threshold_mode));
  visitor("threshold", value);
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.segmentation_window_size));
  visitor("window", value);
  visitor("detector", snapshot.detector_name);
  visitor("classic_recovery_vote", config.classic_recovery_vote_enabled ? "on" : "off");
  visitor("subcarriers", subcarrier_source_name(snapshot.subcarrier_source));
  visitor("lowpass", config.lowpass_enabled ? "on" : "off");
  std::snprintf(value, sizeof(value), "%.1f", config.lowpass_cutoff);
  visitor("lowpass_cutoff", value);
  visitor("hampel", config.hampel_enabled ? "on" : "off");
  if (config.hampel_enabled) {
    std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.hampel_window));
    visitor("hampel_window", value);
    std::snprintf(value, sizeof(value), "%.1f", config.hampel_threshold);
    visitor("hampel_threshold", value);
  }
  visitor("traffic_mode", traffic_mode_name(config.traffic_generator_mode));
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.traffic_generator_rate));
  visitor("traffic_rate", value);
  visitor("traffic_adaptive", config.traffic_generator_adaptive ? "on" : "off");
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.publish_interval));
  visitor("publish_interval", value);
  std::snprintf(value, sizeof(value), "%u", static_cast<unsigned>(config.evaluation_interval));
  visitor("evaluation_interval", value);
  std::snprintf(value,
                sizeof(value),
                "%u/%u",
                static_cast<unsigned>(config.motion_on_hits),
                static_cast<unsigned>(config.motion_off_hits));
  visitor("motion_hits", value);
  std::snprintf(value, sizeof(value), "%.6f", snapshot.startup_threshold);
  visitor("startup_threshold", value);
}

}  // namespace espectre
