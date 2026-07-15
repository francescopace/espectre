/*
 * ESPectre - Device Config Store
 *
 * Persists Wi-Fi and device configuration in ESP-IDF non-volatile storage.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "device_config_store.h"

#include "nvs.h"

namespace espectre {

namespace {

constexpr const char *kNamespace = "espectre";
constexpr const char *kWifiSsidKey = "wifi_ssid";
constexpr const char *kWifiPasswordKey = "wifi_pass";
constexpr const char *kWifiBssidKey = "wifi_bssid";
constexpr const char *kWifiChannelKey = "wifi_chan";
constexpr const char *kDeviceLabelKey = "device_label";
constexpr const char *kMqttHostKey = "mqtt_host";
constexpr const char *kMqttPortKey = "mqtt_port";
constexpr const char *kMqttUserKey = "mqtt_user";
constexpr const char *kMqttPasswordKey = "mqtt_pass";
constexpr const char *kTopicPrefixKey = "topic_prefix";

esp_err_t read_string(nvs_handle_t handle, const char *key, std::string *value) {
  size_t length = 0;
  esp_err_t err = nvs_get_str(handle, key, nullptr, &length);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    value->clear();
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }

  std::string buffer(length, '\0');
  err = nvs_get_str(handle, key, buffer.data(), &length);
  if (err != ESP_OK) {
    return err;
  }
  if (!buffer.empty() && buffer.back() == '\0') {
    buffer.pop_back();
  }
  *value = buffer;
  return ESP_OK;
}

esp_err_t write_string(nvs_handle_t handle, const char *key, const std::string &value) {
  if (value.empty()) {
    const esp_err_t err = nvs_erase_key(handle, key);
    return err == ESP_ERR_NVS_NOT_FOUND ? ESP_OK : err;
  }
  return nvs_set_str(handle, key, value.c_str());
}

}  // namespace

esp_err_t load_stored_wifi_config(StoredWifiConfig *config) {
  if (config == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }

  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READONLY, &handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    *config = StoredWifiConfig{};
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }

  StoredWifiConfig loaded;
  err = read_string(handle, kWifiSsidKey, &loaded.ssid);
  if (err == ESP_OK) {
    err = read_string(handle, kWifiPasswordKey, &loaded.password);
  }
  if (err == ESP_OK) {
    err = read_string(handle, kWifiBssidKey, &loaded.bssid);
  }
  uint8_t channel = 0;
  const esp_err_t channel_err = nvs_get_u8(handle, kWifiChannelKey, &channel);
  if (err == ESP_OK && channel_err != ESP_OK && channel_err != ESP_ERR_NVS_NOT_FOUND) {
    err = channel_err;
  }
  nvs_close(handle);

  if (err != ESP_OK) {
    return err;
  }

  loaded.channel = channel;
  loaded.has_saved_config = !loaded.ssid.empty();
  *config = loaded;
  return ESP_OK;
}

esp_err_t save_stored_wifi_config(const StoredWifiConfig &config) {
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err != ESP_OK) {
    return err;
  }

  err = write_string(handle, kWifiSsidKey, config.ssid);
  if (err == ESP_OK) {
    err = write_string(handle, kWifiPasswordKey, config.password);
  }
  if (err == ESP_OK) {
    err = write_string(handle, kWifiBssidKey, config.bssid);
  }
  if (err == ESP_OK) {
    err = nvs_set_u8(handle, kWifiChannelKey, config.channel);
  }
  if (err == ESP_OK) {
    err = nvs_commit(handle);
  }
  nvs_close(handle);
  return err;
}

esp_err_t clear_stored_wifi_config() {
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }

  esp_err_t result = nvs_erase_key(handle, kWifiSsidKey);
  if (result == ESP_ERR_NVS_NOT_FOUND) {
    result = ESP_OK;
  }
  esp_err_t err2 = nvs_erase_key(handle, kWifiPasswordKey);
  if (result == ESP_OK && err2 != ESP_ERR_NVS_NOT_FOUND) {
    result = err2;
  }
  err2 = nvs_erase_key(handle, kWifiBssidKey);
  if (result == ESP_OK && err2 != ESP_ERR_NVS_NOT_FOUND) {
    result = err2;
  }
  err2 = nvs_erase_key(handle, kWifiChannelKey);
  if (result == ESP_OK && err2 != ESP_ERR_NVS_NOT_FOUND) {
    result = err2;
  }
  if (result == ESP_OK) {
    result = nvs_commit(handle);
  }
  nvs_close(handle);
  return result;
}

esp_err_t load_stored_device_config(EspectreDeviceConfig *config, bool *has_saved_config) {
  if (config == nullptr) {
    return ESP_ERR_INVALID_ARG;
  }

  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READONLY, &handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    if (has_saved_config != nullptr) {
      *has_saved_config = false;
    }
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }

  EspectreDeviceConfig loaded;
  err = read_string(handle, kDeviceLabelKey, &loaded.device_label);
  if (err == ESP_OK) {
    err = read_string(handle, kMqttHostKey, &loaded.mqtt_host);
  }
  if (err == ESP_OK) {
    err = read_string(handle, kMqttUserKey, &loaded.mqtt_username);
  }
  if (err == ESP_OK) {
    err = read_string(handle, kMqttPasswordKey, &loaded.mqtt_password);
  }
  if (err == ESP_OK) {
    err = read_string(handle, kTopicPrefixKey, &loaded.topic_prefix);
  }

  uint16_t port = 0;
  const esp_err_t port_err = nvs_get_u16(handle, kMqttPortKey, &port);
  if (err == ESP_OK && port_err != ESP_OK && port_err != ESP_ERR_NVS_NOT_FOUND) {
    err = port_err;
  }

  nvs_close(handle);

  if (err != ESP_OK) {
    return err;
  }

  const bool has_config = !loaded.device_label.empty() || !loaded.mqtt_host.empty() || !loaded.mqtt_username.empty() ||
                          !loaded.mqtt_password.empty() || !loaded.topic_prefix.empty() || port_err == ESP_OK;
  if (!has_config) {
    if (has_saved_config != nullptr) {
      *has_saved_config = false;
    }
    return ESP_OK;
  }

  if (loaded.topic_prefix.empty()) {
    loaded.topic_prefix = ESPECTRE_TOPIC_PREFIX;
  }
  if (port_err == ESP_OK && port != 0) {
    loaded.mqtt_port = port;
  }
  *config = loaded;
  if (has_saved_config != nullptr) {
    *has_saved_config = true;
  }
  return ESP_OK;
}

esp_err_t save_stored_device_config(const EspectreDeviceConfig &config) {
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err != ESP_OK) {
    return err;
  }

  err = write_string(handle, kDeviceLabelKey, config.device_label);
  if (err == ESP_OK) {
    err = write_string(handle, kMqttHostKey, config.mqtt_host);
  }
  if (err == ESP_OK) {
    err = write_string(handle, kMqttUserKey, config.mqtt_username);
  }
  if (err == ESP_OK) {
    err = write_string(handle, kMqttPasswordKey, config.mqtt_password);
  }
  if (err == ESP_OK) {
    err = write_string(handle, kTopicPrefixKey, config.topic_prefix);
  }
  if (err == ESP_OK) {
    err = nvs_set_u16(handle, kMqttPortKey, config.mqtt_port);
  }
  if (err == ESP_OK) {
    err = nvs_commit(handle);
  }
  nvs_close(handle);
  return err;
}

esp_err_t clear_stored_device_config() {
  nvs_handle_t handle = 0;
  esp_err_t err = nvs_open(kNamespace, NVS_READWRITE, &handle);
  if (err == ESP_ERR_NVS_NOT_FOUND) {
    return ESP_OK;
  }
  if (err != ESP_OK) {
    return err;
  }

  const char *keys[] = {kDeviceLabelKey,
                        kMqttHostKey,
                        kMqttPortKey,
                        kMqttUserKey,
                        kMqttPasswordKey,
                        kTopicPrefixKey};
  esp_err_t result = ESP_OK;
  for (const char *key : keys) {
    const esp_err_t erase_err = nvs_erase_key(handle, key);
    if (result == ESP_OK && erase_err != ESP_OK && erase_err != ESP_ERR_NVS_NOT_FOUND) {
      result = erase_err;
    }
  }
  if (result == ESP_OK) {
    result = nvs_commit(handle);
  }
  nvs_close(handle);
  return result;
}

}  // namespace espectre
