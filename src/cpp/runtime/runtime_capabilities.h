/*
 * ESPectre - Runtime Capabilities
 *
 * Capability flags advertised by a runtime implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

namespace espectre {

struct RuntimeCapabilities {
  bool supports_runtime_threshold_updates{true};
  bool supports_runtime_detector_selection{false};
  bool supports_manual_recalibration{true};
  bool supports_ble_telemetry{true};
  bool supports_extended_diagnostics{true};
  bool supports_traffic_control{true};
};

}  // namespace espectre
