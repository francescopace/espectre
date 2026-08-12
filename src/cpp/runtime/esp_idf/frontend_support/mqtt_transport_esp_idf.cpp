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

#include <cstdio>

#include "esp_log.h"

namespace espectre {

namespace {

static const char *const TAG = "espectre.mqtt";

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
  mqtt_config.session.last_will.retain = false;

  client_ = esp_mqtt_client_init(&mqtt_config);
  if (client_ == nullptr) {
    ESP_LOGE(TAG, "esp_mqtt_client_init failed");
    return false;
  }
  esp_mqtt_client_register_event(client_, MQTT_EVENT_ANY, &EspIdfMqttTransport::event_handler_, this);
  const esp_err_t err = esp_mqtt_client_start(client_);
  if (err != ESP_OK) {
    ESP_LOGE(TAG, "esp_mqtt_client_start failed: %s", esp_err_to_name(err));
    esp_mqtt_client_destroy(client_);
    client_ = nullptr;
    return false;
  }
  ESP_LOGI(TAG, "MQTT transport connecting to %s", broker_uri_.c_str());
  return true;
}

void EspIdfMqttTransport::loop() {}

void EspIdfMqttTransport::shutdown() {
  connected_ = false;
  command_payload_assembler_.reset();
  if (client_ == nullptr) {
    return;
  }
  esp_mqtt_client_stop(client_);
  esp_mqtt_client_destroy(client_);
  client_ = nullptr;
}

bool EspIdfMqttTransport::publish(const std::string &topic, const std::string &payload, bool retain) {
  if (client_ == nullptr || !connected_) {
    return false;
  }
  const int id = esp_mqtt_client_publish(client_, topic.c_str(), payload.c_str(), 0, 0, retain ? 1 : 0);
  return id >= 0;
}

bool EspIdfMqttTransport::publish_suffix(const char *suffix, const std::string &payload, bool retain) {
  if (client_ == nullptr || !connected_ || suffix == nullptr || suffix[0] == '\0') {
    return false;
  }
  publish_topic_.assign(topic_base_);
  publish_topic_.append(suffix);
  const int id = esp_mqtt_client_publish(
      client_, publish_topic_.c_str(), payload.c_str(), 0, 0, retain ? 1 : 0);
  return id >= 0;
}

bool EspIdfMqttTransport::subscribe(const std::string &topic, MessageCallback callback) {
  if (topic.empty() || !callback) {
    return false;
  }
  for (auto &subscription : subscriptions_) {
    if (subscription.topic == topic) {
      subscription.callback = std::move(callback);
      return connected_ ? subscribe_topic_(topic) : true;
    }
  }
  subscriptions_.push_back(TopicSubscription{topic, std::move(callback)});
  return connected_ ? subscribe_topic_(topic) : true;
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
      connected_ = true;
      subscribe_registered_topics_();
      if (connection_callback_) {
        connection_callback_(true);
      }
      ESP_LOGI(TAG, "MQTT connected");
      break;
    case MQTT_EVENT_DISCONNECTED:
      connected_ = false;
      command_payload_assembler_.reset();
      if (connection_callback_) {
        connection_callback_(false);
      }
      ESP_LOGW(TAG, "MQTT disconnected");
      break;
    case MQTT_EVENT_DATA:
      if (event->topic == nullptr || event->topic_len <= 0 || event->data == nullptr || event->data_len <= 0) {
        break;
      }
      {
        const std::string topic(event->topic, static_cast<size_t>(event->topic_len));
        if (topic == command_topic_ && command_callback_) {
          const auto result = command_payload_assembler_.append(
              event->data,
              static_cast<size_t>(event->data_len),
              static_cast<size_t>(event->total_data_len),
              static_cast<size_t>(event->current_data_offset));
          if (result == MqttPayloadAssembler::Result::COMPLETE) {
            command_callback_(command_payload_assembler_.payload());
            command_payload_assembler_.reset();
          } else if (result == MqttPayloadAssembler::Result::INVALID) {
            ESP_LOGW(TAG, "Rejected invalid or oversized MQTT command payload");
          }
          break;
        }
        if (event->current_data_offset != 0 || event->data_len != event->total_data_len) {
          ESP_LOGW(TAG, "Ignoring fragmented MQTT payload on unsupported topic: %s", topic.c_str());
          break;
        }
        for (const auto &subscription : subscriptions_) {
          if (subscription.topic == topic && subscription.callback) {
            subscription.callback(topic, std::string(event->data, static_cast<size_t>(event->data_len)));
            break;
          }
        }
      }
      break;
    default:
      break;
  }
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
