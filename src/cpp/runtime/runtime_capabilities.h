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

/**
 * What a runtime actually offers its frontend.
 *
 * Every flag defaults to false on purpose: ESPECTRE_PROTOCOL.md makes this
 * block the contract clients read, so a capability has to be declared rather
 * than inherited from a permissive default. Previously only the stream runtime
 * declared anything and the sensing runtime shipped whatever the struct
 * happened to default to.
 *
 * `supports_ble_telemetry` describes the runtime side of the surface: whether
 * it drives the live-telemetry callback at all. Whether that reaches a BLE
 * characteristic is the frontend's own business, and only Native forwards it.
 */
struct RuntimeCapabilities {
  bool supports_runtime_threshold_updates{false};
  bool supports_runtime_detector_selection{false};
  bool supports_manual_recalibration{false};
  bool supports_ble_telemetry{false};
  bool supports_extended_diagnostics{false};
  bool supports_traffic_control{false};
};

}  // namespace espectre
