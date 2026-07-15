/*
 * ESPectre - ESP-IDF MQTT Transport
 *
 * ESP-IDF MQTT transport implementation for the native sensing frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <string>

#include "mqtt_transport.h"
#include "mqtt_payload_assembler.h"
#include "mqtt_client.h"

namespace espectre {

class EspIdfMqttTransport : public IMqttTransport {
 public:
  bool setup(const EspectreDeviceConfig &config) override;
  void loop() override;
  void shutdown() override;
  bool connected() const override { return connected_; }
  bool publish(const std::string &topic, const std::string &payload, bool retain) override;
  bool publish_suffix(const char *suffix, const std::string &payload, bool retain) override;
  void set_command_callback(CommandCallback callback) override;
  void set_connection_callback(ConnectionCallback callback) override;

 private:
  static void event_handler_(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data);
  void handle_event_(esp_mqtt_event_handle_t event);
  void subscribe_commands_();

  esp_mqtt_client_handle_t client_{nullptr};
  CommandCallback command_callback_{};
  ConnectionCallback connection_callback_{};
  MqttPayloadAssembler command_payload_assembler_{};
  std::string broker_uri_{};
  std::string mqtt_username_{};
  std::string mqtt_password_{};
  std::string topic_base_{};
  std::string publish_topic_{};
  std::string command_topic_{};
  std::string last_will_topic_{};
  std::string last_will_payload_{};
  bool connected_{false};
};

}  // namespace espectre
