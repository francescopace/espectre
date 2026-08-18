/*
 * ESPectre - Runtime Capabilities
 *
 * Capability flags advertised by a runtime implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
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
 * it drives the live-telemetry callback at all. Native uses that callback for
 * MQTT sensing telemetry and Home Assistant Movement Score, not for BLE notify.
 */
struct RuntimeCapabilities {
  /** `set_threshold_runtime()` is honored. */
  bool supports_runtime_threshold_updates{false};
  /** `set_motion_hits_runtime()` is honored; otherwise the controller refuses it. */
  bool supports_runtime_motion_hits_updates{false};
  /**
   * `set_detection_algorithm_runtime()` is honored.
   *
   * Driven by `RuntimeConfig::runtime_detector_selection_enabled`, since
   * switching detectors also means persisting and restoring the choice.
   */
  bool supports_runtime_detector_selection{false};
  /** `trigger_recalibration()` is honored; otherwise the controller refuses it. */
  bool supports_manual_recalibration{false};
  /**
   * The runtime drives `IRuntimeListener::on_live_telemetry()` at all.
   *
   * Native uses that callback for MQTT sensing telemetry and Home Assistant
   * Movement Score. It does not forward live sensing onto BLE.
   */
  bool supports_ble_telemetry{false};
  /** The runtime reports the extended diagnostics block used by stats payloads. */
  bool supports_extended_diagnostics{false};
  /** The runtime owns CSI traffic generation and can be asked to retune it. */
  bool supports_traffic_control{false};
};

}  // namespace espectre
