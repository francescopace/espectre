/*
 * ESPectre - BLE Bindings Mock
 *
 * Test double for the BLE bindings boundary used by native frontend tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ble_bindings.h"

namespace espectre {
namespace ble_bindings_mock {

struct TelemetryPublish {
  std::vector<uint8_t> payload;
};

struct State {
  bool setup_result{true};
  bool shutdown_called{false};
  std::vector<bool> connection_events;
  std::vector<std::string> control_commands;
  std::vector<TelemetryPublish> telemetry_events;
  std::vector<std::string> sysinfo_lines;
  std::vector<std::string> faults;
  std::vector<std::string> device_names;
  IBleBindings::ConnectionStateCallback connection_callback;
  IBleBindings::ControlWriteCallback control_callback;
  IBleBindings::TelemetrySubscriptionCallback telemetry_subscription_callback;
};

extern State state;

void reset();

class MockBleBindings : public IBleBindings {
 public:
  bool setup() override;
  void loop() override;
  void shutdown() override;
  void set_connection_state_callback(ConnectionStateCallback callback) override;
  void set_control_write_callback(ControlWriteCallback callback) override;
  void set_telemetry_subscription_callback(TelemetrySubscriptionCallback callback) override;
  void set_device_name(const char *name) override;
  void publish_telemetry(const uint8_t *payload, size_t payload_len) override;
  void replace_sysinfo_lines(std::vector<std::string> lines) override;
  void publish_sysinfo_line(const char *line) override;
  void report_fault(const char *message) override;

  void emit_connection(bool connected);
  void emit_control(const std::string &command);
  void emit_telemetry_subscription(bool subscribed);
};

}  // namespace ble_bindings_mock
}  // namespace espectre
