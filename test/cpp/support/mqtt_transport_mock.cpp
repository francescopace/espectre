#include "mqtt_transport_mock.h"

namespace esphome {
namespace espectre {
namespace mqtt_transport_mock {

State state{};

void reset() { state = State{}; }

bool MockMqttTransport::setup(const EspectreDeviceConfig &config) {
  state.setup_calls += 1;
  state.last_config = config;
  return state.setup_result;
}

void MockMqttTransport::loop() {}

void MockMqttTransport::shutdown() { state.shutdown_called = true; }

bool MockMqttTransport::connected() const { return state.connected; }

bool MockMqttTransport::publish(const std::string &topic, const std::string &payload, bool retain) {
  state.publishes.push_back(Publish{topic, payload, retain});
  return true;
}

bool MockMqttTransport::publish_suffix(const char *suffix, const std::string &payload, bool retain) {
  if (suffix == nullptr) {
    return false;
  }
  return publish(espectre_topic(state.last_config, suffix), payload, retain);
}

void MockMqttTransport::set_command_callback(CommandCallback callback) { state.command_callback = std::move(callback); }

void MockMqttTransport::set_connection_callback(ConnectionCallback callback) {
  state.connection_callback = std::move(callback);
}

void MockMqttTransport::emit_command(const std::string &payload) {
  if (state.command_callback) {
    state.command_callback(payload);
  }
}

void MockMqttTransport::emit_connection(bool connected) {
  state.connected = connected;
  if (state.connection_callback) {
    state.connection_callback(connected);
  }
}

}  // namespace mqtt_transport_mock
}  // namespace espectre
}  // namespace esphome
