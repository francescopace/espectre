/*
 * ESPectre - MQTT Transport Mock
 *
 * Test double for the MQTT transport boundary used by native frontend
 * tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <string>
#include <vector>

#include "mqtt_transport.h"

namespace espectre {
namespace mqtt_transport_mock {

struct Publish {
  std::string topic;
  std::string payload;
  bool retain{false};
};

struct Subscription {
  std::string topic;
  IMqttTransport::MessageCallback callback;
};

struct State {
  bool setup_result{true};
  bool connected{true};
  bool shutdown_called{false};
  int setup_calls{0};
  EspectreDeviceConfig last_config;
  std::vector<Publish> publishes;
  std::vector<Subscription> subscriptions;
  IMqttTransport::CommandCallback command_callback;
  IMqttTransport::ConnectionCallback connection_callback;
  MqttTransportDiagnostics diagnostics;
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
  bool subscribe(const std::string &topic, MessageCallback callback) override;
  void set_command_callback(CommandCallback callback) override;
  void set_connection_callback(ConnectionCallback callback) override;
  MqttTransportDiagnostics diagnostics() const override;

  void emit_command(const std::string &payload);
  void emit_message(const std::string &topic, const std::string &payload);
  void emit_connection(bool connected);
};

}  // namespace mqtt_transport_mock
}  // namespace espectre
