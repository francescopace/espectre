/*
 * ESPectre - No-op BLE Bindings
 *
 * Lightweight BLE binding used by QEMU smoke builds where the BLE controller
 * path is not emulated, but frontend boot should still continue.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <vector>
#include <utility>

#include "ble_bindings.h"

namespace espectre {

class NoopBleBindings : public IBleBindings {
 public:
  bool setup() override { return true; }
  void loop() override {}
  void shutdown() override {}

  void set_connection_state_callback(ConnectionStateCallback callback) override { connection_state_callback_ = std::move(callback); }
  void set_control_write_callback(ControlWriteCallback callback) override { control_write_callback_ = std::move(callback); }
  void set_telemetry_subscription_callback(TelemetrySubscriptionCallback callback) override {
    telemetry_subscription_callback_ = std::move(callback);
  }
  void set_device_name(const char *name) override { (void) name; }

  void publish_telemetry(const uint8_t *payload, size_t payload_len) override {
    (void) payload;
    (void) payload_len;
  }
  void replace_sysinfo_lines(std::vector<std::string> lines) override { (void) lines; }
  void publish_sysinfo_line(const char *line) override { (void) line; }
  void report_fault(const char *message) override { (void) message; }

 private:
  ConnectionStateCallback connection_state_callback_{};
  ControlWriteCallback control_write_callback_{};
  TelemetrySubscriptionCallback telemetry_subscription_callback_{};
};

}  // namespace espectre
