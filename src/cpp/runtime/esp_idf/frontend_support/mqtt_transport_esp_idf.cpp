/*
 * ESPectre - ESP-IDF MQTT transport for the native frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "mqtt_transport_esp_idf.h"

#include <cstdio>

#include "esp_log.h"

namespace espectre {

namespace {

static const char *const TAG = "espectre.mqtt";

std::string make_broker_uri(const EspectreDeviceConfig &config) {
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
  config_ = config;
  if (config_.mqtt_host.empty()) {
    return false;
  }

  if (client_ != nullptr) {
    shutdown();
  }

  broker_uri_ = make_broker_uri(config_);
  topic_base_ = make_topic_base(config_);
  command_topic_ = topic_base_ + "commands/request";
  last_will_topic_ = topic_base_ + "status";
  last_will_payload_ = espectre_status_payload(config_, false, 0);
  esp_mqtt_client_config_t mqtt_config{};
  mqtt_config.broker.address.uri = broker_uri_.c_str();
  if (!config_.mqtt_username.empty()) {
    mqtt_config.credentials.username = config_.mqtt_username.c_str();
  }
  if (!config_.mqtt_password.empty()) {
    mqtt_config.credentials.authentication.password = config_.mqtt_password.c_str();
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
  if (suffix == nullptr || suffix[0] == '\0') {
    return false;
  }
  return publish(topic_base_ + suffix, payload, retain);
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
      subscribe_commands_();
      if (connection_callback_) {
        connection_callback_(true);
      }
      ESP_LOGI(TAG, "MQTT connected");
      break;
    case MQTT_EVENT_DISCONNECTED:
      connected_ = false;
      if (connection_callback_) {
        connection_callback_(false);
      }
      ESP_LOGW(TAG, "MQTT disconnected");
      break;
    case MQTT_EVENT_DATA:
      if (command_callback_ && event->data != nullptr && event->data_len > 0) {
        command_callback_(std::string(event->data, event->data_len));
      }
      break;
    default:
      break;
  }
}

void EspIdfMqttTransport::subscribe_commands_() {
  if (client_ == nullptr) {
    return;
  }
  esp_mqtt_client_subscribe(client_, command_topic_.c_str(), 0);
}

}  // namespace espectre
