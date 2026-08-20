/*
 * ESPectre - ESP-IDF MQTT Transport
 *
 * ESP-IDF MQTT transport implementation for the native sensing frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <array>
#include <atomic>
#include <string>
#include <vector>

#include "mqtt_transport.h"
#include "mqtt_payload_assembler.h"
#include "pending_event.h"
#include "pending_queue.h"
#include "mqtt_client.h"

namespace espectre {

class EspIdfMqttTransport : public IMqttTransport {
 public:
  bool setup(const EspectreDeviceConfig &config) override;
  void loop() override;
  void shutdown() override;
  bool connected() const override { return connected_.load(std::memory_order_relaxed); }
  bool publish(const std::string &topic, const std::string &payload, bool retain) override;
  bool publish_suffix(const char *suffix, const std::string &payload, bool retain) override;
  bool subscribe(const std::string &topic, MessageCallback callback) override;
  void set_command_callback(CommandCallback callback) override;
  void set_connection_callback(ConnectionCallback callback) override;

 private:
  struct TopicSubscription {
    std::string topic;
    MessageCallback callback;
  };

  struct PendingMessage {
    std::array<char, 256U> topic{};
    std::array<char, MqttPayloadAssembler::MAX_PAYLOAD_SIZE + 1U> payload{};
    uint16_t topic_len{0U};
    uint16_t payload_len{0U};
  };

  static void event_handler_(void *handler_args, esp_event_base_t base, int32_t event_id, void *event_data);
  void handle_event_(esp_mqtt_event_handle_t event);
  bool enqueue_message_(const char *topic, size_t topic_len, const char *payload, size_t payload_len);
  void dispatch_message_(const PendingMessage &message);
  void reset_message_slots_();
  bool subscribe_topic_(const std::string &topic);
  void subscribe_registered_topics_();

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
  std::vector<TopicSubscription> subscriptions_{};
  static constexpr size_t kPendingMessageCapacity = 4U;
  PendingEvent<bool> connection_event_{};
  std::array<PendingMessage, kPendingMessageCapacity> message_slots_{};
  PendingQueue<uint8_t, kPendingMessageCapacity> free_message_slots_{};
  PendingQueue<uint8_t, kPendingMessageCapacity> ready_message_slots_{};
  std::atomic<bool> connected_{false};
  std::atomic<uint32_t> dropped_messages_{0U};
};

}  // namespace espectre
