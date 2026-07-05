#pragma once

#include <string>
#include <vector>

#include "mqtt_transport.h"

namespace esphome {
namespace espectre {
namespace mqtt_transport_mock {

struct Publish {
  std::string topic;
  std::string payload;
  bool retain{false};
};

struct State {
  bool setup_result{true};
  bool connected{true};
  bool shutdown_called{false};
  int setup_calls{0};
  EspectreDeviceConfig last_config;
  std::vector<Publish> publishes;
  IMqttTransport::CommandCallback command_callback;
  IMqttTransport::ConnectionCallback connection_callback;
};

extern State state;

void reset();

class MockMqttTransport : public IMqttTransport {
 public:
  bool setup(const EspectreDeviceConfig &config) override;
  void loop() override;
  void shutdown() override;
  bool connected() const override;
  bool publish(const std::string &topic, const std::string &payload, bool retain) override;
  bool publish_suffix(const char *suffix, const std::string &payload, bool retain) override;
  void set_command_callback(CommandCallback callback) override;
  void set_connection_callback(ConnectionCallback callback) override;

  void emit_command(const std::string &payload);
  void emit_connection(bool connected);
};

}  // namespace mqtt_transport_mock
}  // namespace espectre
}  // namespace esphome
