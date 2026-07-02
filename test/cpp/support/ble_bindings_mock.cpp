#include "ble_bindings_mock.h"

namespace esphome {
namespace espectre {
namespace ble_bindings_mock {

State state{};

void reset() { state = State{}; }

bool MockBleBindings::setup() { return state.setup_result; }

void MockBleBindings::shutdown() { state.shutdown_called = true; }

void MockBleBindings::set_connection_state_callback(ConnectionStateCallback callback) {
  state.connection_callback = std::move(callback);
}

void MockBleBindings::set_control_write_callback(ControlWriteCallback callback) {
  state.control_callback = std::move(callback);
}

void MockBleBindings::set_telemetry_subscription_callback(TelemetrySubscriptionCallback callback) {
  state.telemetry_subscription_callback = std::move(callback);
}

void MockBleBindings::set_device_name(const char *name) {
  state.device_names.emplace_back(name != nullptr ? name : "");
}

void MockBleBindings::publish_telemetry(const uint8_t *payload, size_t payload_len) {
  TelemetryPublish publish;
  publish.payload.assign(payload, payload + payload_len);
  state.telemetry_events.push_back(std::move(publish));
}

void MockBleBindings::publish_sysinfo_line(const char *line) {
  state.sysinfo_lines.emplace_back(line != nullptr ? line : "");
}

void MockBleBindings::report_fault(const char *message) {
  if (message != nullptr) {
    state.faults.emplace_back(message);
  }
}

void MockBleBindings::emit_connection(bool connected) {
  state.connection_events.push_back(connected);
  if (state.connection_callback) {
    state.connection_callback(connected);
  }
}

void MockBleBindings::emit_control(const std::string &command) {
  state.control_commands.push_back(command);
  if (state.control_callback) {
    state.control_callback(command);
  }
}

void MockBleBindings::emit_telemetry_subscription(bool subscribed) {
  if (state.telemetry_subscription_callback) {
    state.telemetry_subscription_callback(subscribed);
  }
}

}  // namespace ble_bindings_mock
}  // namespace espectre
}  // namespace esphome
