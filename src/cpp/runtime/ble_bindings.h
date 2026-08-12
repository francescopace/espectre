/*
 * ESPectre - BLE Bindings Interface
 *
 * Thin boundary between frontend adapters and the BLE transport stack.
 * Host-side tests provide a mock implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace espectre {

/**
 * The BLE GATT seam.
 *
 * Implement it to expose the ESPectre BLE control surface over a stack other
 * than NimBLE, or to stub BLE out entirely. Three implementations ship:
 * `NimbleBleBindings` (`ble_bindings_nimble.h`), `NoopBleBindings`
 * (`ble_bindings_noop.h`) for builds without Bluetooth, and a host mock under
 * `test/cpp/support/`.
 *
 * The service and characteristic UUIDs are fixed in `ble_protocol.h`, and the
 * command grammar is specified in `docs/ESPECTRE_PROTOCOL.md`. Keeping to both
 * is what lets the web BLE client in `docs/web/assets/js/espectre-ble.js` talk to your
 * firmware unchanged.
 *
 * @par Threading
 * Callbacks come from the BLE stack's own task, not from your loop task. Keep
 * them short, and prefer queueing the request and applying it from your loop
 * over driving the runtime directly from the BLE context.
 */
class IBleBindings {
 public:
  /** A central connected or disconnected. */
  using ConnectionStateCallback = std::function<void(bool connected)>;
  /** A command was written to the control characteristic, delivered verbatim. */
  using ControlWriteCallback = std::function<void(const std::string &)>;
  /** A central subscribed to or unsubscribed from telemetry notifications. */
  using TelemetrySubscriptionCallback = std::function<void(bool subscribed)>;

  virtual ~IBleBindings() = default;

  /**
   * Bring up the stack, register the GATT service, and start advertising.
   *
   * Install the callbacks and the device name first: an implementation may
   * begin advertising before this returns.
   *
   * @return false when the stack cannot start, for example in a build without
   *         Bluetooth enabled.
   */
  virtual bool setup() = 0;
  /** Advance stack work from the frontend loop. Empty for event-driven stacks. */
  virtual void loop() = 0;
  /** Stop advertising, drop connections, and release the stack. Safe to repeat. */
  virtual void shutdown() = 0;

  /** Install the connection-state handler. Set it before `setup()`. */
  virtual void set_connection_state_callback(ConnectionStateCallback callback) = 0;
  /** Install the control-command handler. Set it before `setup()`. */
  virtual void set_control_write_callback(ControlWriteCallback callback) = 0;
  /** Install the telemetry subscription handler. Set it before `setup()`. */
  virtual void set_telemetry_subscription_callback(TelemetrySubscriptionCallback callback) = 0;
  /**
   * Set the advertised device name.
   *
   * Call it before `setup()`. `espectre_device_name()` builds the conventional
   * name so devices stay identifiable in a scan.
   */
  virtual void set_device_name(const char *name) = 0;

  /**
   * Notify the telemetry characteristic.
   *
   * A no-op while no central is connected, so callers do not have to track
   * connection state to avoid wasted work.
   *
   * @param payload Copied before returning; the buffer is yours again after.
   * @param payload_len Length of `payload` in bytes.
   */
  virtual void publish_telemetry(const uint8_t *payload, size_t payload_len) = 0;
  /**
   * Replace the whole sysinfo block, for a full refresh after a status query.
   *
   * Sysinfo is a one-value characteristic, so implementations paced the lines
   * out one notification at a time. Replacing discards anything still pending.
   */
  virtual void replace_sysinfo_lines(std::vector<std::string> lines) = 0;
  /** Append one sysinfo line, for incremental updates between refreshes. */
  virtual void publish_sysinfo_line(const char *line) = 0;
  /**
   * Report a runtime-owned fault to the BLE surface.
   *
   * Advisory: no shipped implementation forwards it to the central today, so
   * treat it as a hook for your own bindings rather than a delivery guarantee.
   */
  virtual void report_fault(const char *message) = 0;
};

}  // namespace espectre
