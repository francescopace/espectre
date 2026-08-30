/*
 * ESPectre - ESP-IDF MQTT Transport
 *
 * ESP-IDF MQTT transport implementation for the native sensing frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "mqtt_transport_esp_idf.h"

#include <algorithm>
#include <cstdio>
#include <cstring>

#include "espectre_log.h"

namespace espectre {

namespace {

[[maybe_unused]] static const char *const TAG = "espectre.mqtt";

bool has_uri_scheme(const std::string &value) {
  return value.find("://") != std::string::npos;
}

bool uri_authority_has_port(const std::string &authority) {
  if (authority.empty()) {
    return false;
  }
  if (authority.front() == '[') {
    const size_t close = authority.find(']');
    return close != std::string::npos && close + 1 < authority.size() && authority[close + 1] == ':';
  }
  const size_t first_colon = authority.find(':');
  const size_t last_colon = authority.rfind(':');
  return first_colon != std::string::npos && first_colon == last_colon;
}

std::string append_port_to_uri(const std::string &uri, uint16_t port) {
  const size_t scheme_pos = uri.find("://");
  if (scheme_pos == std::string::npos) {
    return uri;
  }

  const size_t authority_start = scheme_pos + 3U;
  const size_t suffix_start = uri.find_first_of("/?#", authority_start);
  const std::string authority = uri.substr(
      authority_start, suffix_start == std::string::npos ? std::string::npos : suffix_start - authority_start);
  if (authority.empty() || uri_authority_has_port(authority)) {
    return uri;
  }

  char port_suffix[8];
  std::snprintf(port_suffix, sizeof(port_suffix), ":%u", static_cast<unsigned>(port));
  if (suffix_start == std::string::npos) {
    return uri + port_suffix;
  }
  return uri.substr(0, suffix_start) + port_suffix + uri.substr(suffix_start);
}

std::string make_broker_uri(const EspectreDeviceConfig &config) {
  if (has_uri_scheme(config.mqtt_host)) {
    return append_port_to_uri(config.mqtt_host, config.mqtt_port);
  }
  char uri[192];
  std::snprintf(uri, sizeof(uri), "mqtt://%s:%u", config.mqtt_host.c_str(), static_cast<unsigned>(config.mqtt_port));
  return uri;
}

std::string make_topic_base(const EspectreDeviceConfig &config) {
  std::string topic = config.topic_prefix.empty() ? ESPECTRE_TOPIC_PREFIX : config.topic_prefix;
  if (!topic.empty() && topic.back() == '/') {
    topic.pop_back();
  }
  topic.push_back('/');
  topic.append(espectre_effective_device_id(config));
  topic.push_back('/');
  return topic;
}

}  // namespace

bool EspIdfMqttTransport::setup(const EspectreDeviceConfig &config) {
  if (config.mqtt_host.empty()) {
    return false;
  }

  if (client_ != nullptr) {
    shutdown();
  }
  subscriptions_.clear();
  reset_message_slots_();
  pending_publishes_.clear();
  diagnostics_ = {};
  connected_once_ = false;

  broker_uri_ = make_broker_uri(config);
  mqtt_username_ = config.mqtt_username;
  mqtt_password_ = config.mqtt_password;
  topic_base_ = make_topic_base(config);
  publish_topic_.reserve(topic_base_.size() + 24U);
  command_topic_ = topic_base_ + "commands/request";
  last_will_topic_ = topic_base_ + "status";
  last_will_payload_ = espectre_status_payload(config, false, 0);
  esp_mqtt_client_config_t mqtt_config{};
  mqtt_config.broker.address.uri = broker_uri_.c_str();
  if (!mqtt_username_.empty()) {
    mqtt_config.credentials.username = mqtt_username_.c_str();
  }
  if (!mqtt_password_.empty()) {
    mqtt_config.credentials.authentication.password = mqtt_password_.c_str();
  }
  mqtt_config.session.last_will.topic = last_will_topic_.c_str();
  mqtt_config.session.last_will.msg = last_will_payload_.c_str();
  mqtt_config.session.last_will.msg_len = 0;
  mqtt_config.session.last_will.qos = 0;
  mqtt_config.session.last_will.retain = true;
  mqtt_config.outbox.limit = kMqttOutboxLimitBytes;

  client_ = esp_mqtt_client_init(&mqtt_config);
  if (client_ == nullptr) {
    ESPECTRE_LOGE(TAG, "esp_mqtt_client_init failed");
    return false;
  }
  esp_mqtt_client_register_event(client_, MQTT_EVENT_ANY, &EspIdfMqttTransport::event_handler_, this);
  const esp_err_t err = esp_mqtt_client_start(client_);
  if (err != ESP_OK) {
    ESPECTRE_LOGE(TAG, "esp_mqtt_client_start failed: %s", esp_err_to_name(err));
    esp_mqtt_client_destroy(client_);
    client_ = nullptr;
    return false;
  }
  ESPECTRE_LOGI(TAG, "MQTT transport connecting to %s", broker_uri_.c_str());
  return true;
}

void EspIdfMqttTransport::loop() {
  bool connected = false;
  if (connection_event_.take(connected)) {
    if (connected) {
      if (connected_once_) {
        diagnostics_.reconnects += 1U;
      }
      connected_once_ = true;
      subscribe_registered_topics_();
    }
    if (connection_callback_) {
      connection_callback_(connected);
    }
  }

  uint8_t slot_index = 0U;
  while (ready_message_slots_.take(slot_index)) {
    if (slot_index < message_slots_.size()) {
      dispatch_message_(message_slots_[slot_index]);
      (void)free_message_slots_.post(slot_index);
    }
  }
  drain_publish_queue_();
}

void EspIdfMqttTransport::shutdown() {
  if (client_ != nullptr) {
    esp_mqtt_client_stop(client_);
    esp_mqtt_client_destroy(client_);
    client_ = nullptr;
  }
  connected_.store(false, std::memory_order_relaxed);
  command_payload_assembler_.reset();
  connection_event_.clear();
  reset_message_slots_();
  pending_publishes_.clear();
  diagnostics_.queued_publishes = 0U;
}

bool EspIdfMqttTransport::publish(const std::string &topic, const std::string &payload, bool retain) {
  if (client_ == nullptr || !connected_.load(std::memory_order_relaxed) || topic.empty()) {
    return false;
  }
  const bool home_assistant_state = topic.compare(0U, topic_base_.size(), topic_base_) == 0 &&
                                    topic.compare(topic_base_.size(), 3U, "ha/") == 0;
  return enqueue_publish_(topic, payload, retain, retain || home_assistant_state);
}

bool EspIdfMqttTransport::publish_suffix(const char *suffix, const std::string &payload, bool retain) {
  if (client_ == nullptr || !connected_.load(std::memory_order_relaxed) || suffix == nullptr || suffix[0] == '\0') {
    return false;
  }
  publish_topic_.assign(topic_base_);
  publish_topic_.append(suffix);
  const bool command_result = std::strcmp(suffix, "commands/result") == 0;
  return enqueue_publish_(publish_topic_, payload, retain, !command_result);
}

MqttTransportDiagnostics EspIdfMqttTransport::diagnostics() const {
  MqttTransportDiagnostics snapshot = diagnostics_;
  snapshot.queue_capacity = kPendingPublishCapacity;
  snapshot.outbox_capacity_bytes = kMqttOutboxLimitBytes;
  snapshot.queued_publishes = pending_publishes_.size();
  return snapshot;
}

bool EspIdfMqttTransport::subscribe(const std::string &topic, MessageCallback callback) {
  if (topic.empty() || !callback) {
    return false;
  }
  for (auto &subscription : subscriptions_) {
    if (subscription.topic == topic) {
      subscription.callback = std::move(callback);
      return connected_.load(std::memory_order_relaxed) ? subscribe_topic_(topic) : true;
    }
  }
  subscriptions_.push_back(TopicSubscription{topic, std::move(callback)});
  return connected_.load(std::memory_order_relaxed) ? subscribe_topic_(topic) : true;
}

void EspIdfMqttTransport::set_command_callback(CommandCallback callback) {
  command_callback_ = std::move(callback);
}

void EspIdfMqttTransport::set_connection_callback(ConnectionCallback callback) {
  connection_callback_ = std::move(callback);
}

void EspIdfMqttTransport::event_handler_(void *handler_args,
                                              esp_event_base_t base,
                                              int32_t event_id,
                                              void *event_data) {
  (void) base;
  (void) event_id;
  auto *transport = static_cast<EspIdfMqttTransport *>(handler_args);
  if (transport != nullptr) {
    transport->handle_event_(static_cast<esp_mqtt_event_handle_t>(event_data));
  }
}

void EspIdfMqttTransport::handle_event_(esp_mqtt_event_handle_t event) {
  if (event == nullptr) {
    return;
  }
  switch (event->event_id) {
    case MQTT_EVENT_CONNECTED:
      connected_.store(true, std::memory_order_relaxed);
      connection_event_.post(true);
      ESPECTRE_LOGI(TAG, "MQTT connected");
      break;
    case MQTT_EVENT_DISCONNECTED:
      connected_.store(false, std::memory_order_relaxed);
      command_payload_assembler_.reset();
      connection_event_.post(false);
      ESPECTRE_LOGW(TAG, "MQTT disconnected");
      break;
    case MQTT_EVENT_DATA:
      if (event->topic == nullptr || event->topic_len <= 0 || event->data == nullptr || event->data_len <= 0) {
        break;
      }
      {
        const size_t topic_len = static_cast<size_t>(event->topic_len);
        const bool is_command = topic_len == command_topic_.size() &&
            std::memcmp(event->topic, command_topic_.data(), topic_len) == 0;
        if (is_command) {
          const auto result = command_payload_assembler_.append(
              event->data,
              static_cast<size_t>(event->data_len),
              static_cast<size_t>(event->total_data_len),
              static_cast<size_t>(event->current_data_offset));
          if (result == MqttPayloadAssembler::Result::COMPLETE) {
            const std::string_view payload = command_payload_assembler_.payload();
            (void)enqueue_message_(event->topic, topic_len, payload.data(), payload.size());
            command_payload_assembler_.reset();
          } else if (result == MqttPayloadAssembler::Result::INVALID) {
            ESPECTRE_LOGW(TAG, "Rejected invalid or oversized MQTT command payload");
          }
          break;
        }
        if (event->current_data_offset != 0 || event->data_len != event->total_data_len) {
          ESPECTRE_LOGW(TAG, "Ignoring fragmented MQTT payload on unsupported topic: %.*s",
                   event->topic_len, event->topic);
          break;
        }
        (void)enqueue_message_(event->topic, topic_len, event->data,
                               static_cast<size_t>(event->data_len));
      }
      break;
    default:
      break;
  }
}

bool EspIdfMqttTransport::enqueue_message_(const char *topic,
                                           size_t topic_len,
                                           const char *payload,
                                           size_t payload_len) {
  if (topic == nullptr || payload == nullptr || topic_len == 0U ||
      topic_len >= PendingMessage{}.topic.size() ||
      payload_len > MqttPayloadAssembler::MAX_PAYLOAD_SIZE) {
    dropped_messages_.fetch_add(1U, std::memory_order_relaxed);
    return false;
  }

  uint8_t slot_index = 0U;
  if (!free_message_slots_.take(slot_index) || slot_index >= message_slots_.size()) {
    dropped_messages_.fetch_add(1U, std::memory_order_relaxed);
    ESPECTRE_LOGW(TAG, "Dropping MQTT message because the frontend queue is full");
    return false;
  }
  PendingMessage &message = message_slots_[slot_index];
  std::memcpy(message.topic.data(), topic, topic_len);
  std::memcpy(message.payload.data(), payload, payload_len);
  message.topic[topic_len] = '\0';
  message.payload[payload_len] = '\0';
  message.topic_len = static_cast<uint16_t>(topic_len);
  message.payload_len = static_cast<uint16_t>(payload_len);
  if (!ready_message_slots_.post(slot_index)) {
    (void)free_message_slots_.post(slot_index);
    dropped_messages_.fetch_add(1U, std::memory_order_relaxed);
    ESPECTRE_LOGW(TAG, "Dropping MQTT message because the frontend queue is full");
    return false;
  }
  return true;
}

void EspIdfMqttTransport::reset_message_slots_() {
  ready_message_slots_.clear();
  free_message_slots_.clear();
  for (uint8_t index = 0U; index < message_slots_.size(); ++index) {
    (void)free_message_slots_.post(index);
  }
}

void EspIdfMqttTransport::dispatch_message_(const PendingMessage &message) {
  const std::string topic(message.topic.data(), message.topic_len);
  const std::string payload(message.payload.data(), message.payload_len);
  if (topic == command_topic_) {
    if (command_callback_) {
      command_callback_(payload);
    }
    return;
  }
  for (const auto &subscription : subscriptions_) {
    if (subscription.topic == topic && subscription.callback) {
      subscription.callback(topic, payload);
      return;
    }
  }
}

bool EspIdfMqttTransport::enqueue_publish_(std::string topic,
                                           const std::string &payload,
                                           bool retain,
                                           bool replaceable) {
  if (replaceable) {
    const auto existing = std::find_if(pending_publishes_.rbegin(),
                                       pending_publishes_.rend(),
                                       [&topic](const PendingPublish &pending) {
                                         return pending.replaceable && pending.topic == topic;
                                       });
    if (existing != pending_publishes_.rend()) {
      existing->payload = payload;
      existing->retain = retain;
      return true;
    }
  }
  if (pending_publishes_.size() >= kPendingPublishCapacity) {
    if (!replaceable) {
      const auto stale = std::find_if(pending_publishes_.begin(),
                                      pending_publishes_.end(),
                                      [](const PendingPublish &pending) { return pending.replaceable; });
      if (stale != pending_publishes_.end()) {
        pending_publishes_.erase(stale);
        diagnostics_.dropped_publishes += 1U;
      } else {
        diagnostics_.dropped_publishes += 1U;
        return false;
      }
    } else {
      diagnostics_.dropped_publishes += 1U;
      return false;
    }
  }
  PendingPublish pending{std::move(topic), payload, retain, replaceable};
  if (replaceable) {
    pending_publishes_.push_back(std::move(pending));
  } else {
    const auto first_replaceable = std::find_if(pending_publishes_.begin(),
                                                pending_publishes_.end(),
                                                [](const PendingPublish &queued) { return queued.replaceable; });
    pending_publishes_.insert(first_replaceable, std::move(pending));
  }
  diagnostics_.queued_publishes = pending_publishes_.size();
  return true;
}

void EspIdfMqttTransport::drain_publish_queue_() {
  if (client_ == nullptr || !connected_.load(std::memory_order_relaxed) || pending_publishes_.empty()) {
    return;
  }
  PendingPublish &pending = pending_publishes_.front();
  if (pending.replaceable &&
      esp_mqtt_client_get_outbox_size(client_) >= static_cast<int>(kReplaceableOutboxHighWaterBytes)) {
    return;
  }
  const int id = esp_mqtt_client_enqueue(
      client_, pending.topic.c_str(), pending.payload.c_str(), 0, 0, pending.retain ? 1 : 0, true);
  if (id >= 0) {
    pending_publishes_.pop_front();
  } else {
    diagnostics_.publish_failures += 1U;
    if (id != -2) {
      pending_publishes_.pop_front();
      diagnostics_.dropped_publishes += 1U;
    }
  }
  diagnostics_.queued_publishes = pending_publishes_.size();
}

bool EspIdfMqttTransport::subscribe_topic_(const std::string &topic) {
  if (client_ == nullptr || topic.empty()) {
    return false;
  }
  return esp_mqtt_client_subscribe(client_, topic.c_str(), 0) >= 0;
}

void EspIdfMqttTransport::subscribe_registered_topics_() {
  if (client_ == nullptr) {
    return;
  }
  (void) subscribe_topic_(command_topic_);
  for (const auto &subscription : subscriptions_) {
    (void) subscribe_topic_(subscription.topic);
  }
}

}  // namespace espectre
