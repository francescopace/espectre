/*
 * ESPectre BLE firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <esp_err.h>
#include <esp_log.h>
#include <esp_mac.h>
#include <nvs_flash.h>

#include "ble_bindings_nimble.h"
#include "ble_frontend.h"
#include "ble_device_config_store.h"
#include "mqtt_transport_esp_idf.h"
#include "standalone_wifi_manager.h"

static const char *TAG = "espectre.ble.app";

namespace {

constexpr int kWifiConnectMaxRetry = 8;

esphome::espectre::BleFrontend *g_frontend = nullptr;
esphome::espectre::StandaloneWifiManager g_wifi_manager;
esphome::espectre::StoredWifiConfig g_wifi_config;
std::string g_wifi_ssid;
std::string g_wifi_password;
std::string g_wifi_bssid;

void sync_frontend_wifi_info() {
  if (g_frontend == nullptr) {
    return;
  }
  esphome::espectre::BleFrontend::WifiProvisioningInfo info;
  info.ssid = g_wifi_config.ssid;
  info.bssid = g_wifi_config.bssid;
  info.channel = g_wifi_config.channel;
  info.has_saved_config = g_wifi_config.has_saved_config;
  info.password_set = !g_wifi_config.password.empty();
  g_frontend->set_wifi_provisioning_info(info);

  esphome::espectre::EspectreDeviceInfo device_info;
  device_info.frontend = "ble";
  device_info.firmware_version = "unknown";
  device_info.chip = CONFIG_IDF_TARGET;

  esphome::espectre::StandaloneWifiInfo wifi_info;
  if (g_wifi_manager.get_info(&wifi_info)) {
    device_info.network.ip_address = wifi_info.ip_address;
    device_info.network.mac_address = wifi_info.mac_address;
    device_info.network.channel = wifi_info.channel;
  }
  g_frontend->set_device_info(device_info);
}

esphome::espectre::RuntimeConfig make_runtime_config() {
  esphome::espectre::RuntimeConfig config;
  return config;
}

std::string derive_protocol_device_id() {
  uint8_t mac[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (esp_read_mac(mac, ESP_MAC_WIFI_STA) != ESP_OK) {
    return esphome::espectre::ESPECTRE_DEFAULT_DEVICE_ID;
  }

  char device_id[sizeof("espectre-ffffffffffff")] = {0};
  std::snprintf(device_id,
                sizeof(device_id),
                "espectre-%02x%02x%02x%02x%02x%02x",
                mac[0],
                mac[1],
                mac[2],
                mac[3],
                mac[4],
                mac[5]);
  return device_id;
}

esphome::espectre::EspectreDeviceConfig make_device_config() {
  esphome::espectre::EspectreDeviceConfig config;
  config.device_id = derive_protocol_device_id();
  config.device_name = CONFIG_ESPECTRE_DEVICE_NAME;
  config.mqtt_host = CONFIG_ESPECTRE_MQTT_HOST;
  config.mqtt_port = CONFIG_ESPECTRE_MQTT_PORT;
  config.topic_prefix = CONFIG_ESPECTRE_TOPIC_PREFIX;
#if defined(CONFIG_ESPECTRE_MQTT_ENABLED)
  config.mqtt_enabled = !config.mqtt_host.empty();
#else
  config.mqtt_enabled = false;
#endif
  esphome::espectre::EspectreDeviceConfig stored_config;
  bool has_stored_config = false;
  const esp_err_t load_err = esphome::espectre::load_stored_device_config(&stored_config, &has_stored_config);
  if (load_err == ESP_OK && has_stored_config) {
    ESP_LOGI(TAG, "Using ESPectre Protocol config provisioned over BLE");
    config = stored_config;
  } else if (load_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to load BLE-provisioned device config: %s", esp_err_to_name(load_err));
  }
  config.device_id = derive_protocol_device_id();
  if (config.device_name.empty()) {
    config.device_name = CONFIG_ESPECTRE_DEVICE_NAME;
  }
  return config;
}

void espectre_loop_task(void *arg) {
  (void) arg;
  while (true) {
    if (g_frontend != nullptr) {
      g_frontend->loop();
    }
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}

bool init_wifi_station() {
  constexpr int kConfiguredWifiChannel = CONFIG_ESPECTRE_WIFI_CHANNEL;
  static_assert(kConfiguredWifiChannel >= 0 && kConfiguredWifiChannel <= 14, "invalid Wi-Fi channel");

  esphome::espectre::StoredWifiConfig stored_config;
  const esp_err_t load_err = esphome::espectre::load_stored_wifi_config(&stored_config);
  if (load_err == ESP_OK && stored_config.has_saved_config) {
    g_wifi_config = stored_config;
    ESP_LOGI(TAG, "Using Wi-Fi credentials provisioned over BLE");
  } else {
    if (load_err != ESP_OK) {
      ESP_LOGW(TAG, "Failed to load BLE-provisioned Wi-Fi config: %s", esp_err_to_name(load_err));
    }
    g_wifi_config.ssid = CONFIG_ESPECTRE_WIFI_SSID;
    g_wifi_config.password = CONFIG_ESPECTRE_WIFI_PASSWORD;
    g_wifi_config.bssid = CONFIG_ESPECTRE_WIFI_BSSID;
    g_wifi_config.channel = static_cast<uint8_t>(kConfiguredWifiChannel);
    g_wifi_config.has_saved_config = false;
  }

  g_wifi_ssid = g_wifi_config.ssid;
  g_wifi_password = g_wifi_config.password;
  g_wifi_bssid = g_wifi_config.bssid;
  sync_frontend_wifi_info();

  esphome::espectre::StandaloneWifiConfig wifi_config;
  wifi_config.ssid = g_wifi_ssid.c_str();
  wifi_config.password = g_wifi_password.c_str();
  wifi_config.bssid = g_wifi_bssid.c_str();
  wifi_config.channel = g_wifi_config.channel;
  wifi_config.max_retry = kWifiConnectMaxRetry;
  wifi_config.manage_csi_lifecycle = false;
  return g_wifi_manager.setup(wifi_config, sync_frontend_wifi_info, sync_frontend_wifi_info) == ESP_OK;
}

bool parse_wifi_channel(const std::string &value, uint8_t *channel) {
  if (channel == nullptr || value.empty()) {
    return false;
  }
  char *end_ptr = nullptr;
  const long parsed = std::strtol(value.c_str(), &end_ptr, 10);
  if (end_ptr == value.c_str() || end_ptr == nullptr || *end_ptr != '\0' || parsed < 0 || parsed > 14) {
    return false;
  }
  *channel = static_cast<uint8_t>(parsed);
  return true;
}

bool apply_wifi_config_live(std::string *message) {
  g_wifi_ssid = g_wifi_config.ssid;
  g_wifi_password = g_wifi_config.password;
  g_wifi_bssid = g_wifi_config.bssid;

  esphome::espectre::StandaloneWifiConfig wifi_config;
  wifi_config.ssid = g_wifi_ssid.c_str();
  wifi_config.password = g_wifi_password.c_str();
  wifi_config.bssid = g_wifi_bssid.c_str();
  wifi_config.channel = g_wifi_config.channel;
  wifi_config.max_retry = kWifiConnectMaxRetry;
  wifi_config.manage_csi_lifecycle = false;

  const esp_err_t err = g_wifi_manager.update_station_config(wifi_config);
  sync_frontend_wifi_info();
  if (message != nullptr) {
    *message = err == ESP_OK ? "Wi-Fi config applied" : esp_err_to_name(err);
  }
  return err == ESP_OK;
}

bool handle_wifi_provisioning_command(const std::string &command, std::string *message) {
  auto set_message = [message](const char *value) {
    if (message != nullptr) {
      *message = value;
    }
  };

  constexpr const char *kSsidPrefix = "SET_WIFI_SSID:";
  constexpr const char *kPasswordPrefix = "SET_WIFI_PASSWORD:";
  constexpr const char *kBssidPrefix = "SET_WIFI_BSSID:";
  constexpr const char *kChannelPrefix = "SET_WIFI_CHANNEL:";

  if (command.rfind(kSsidPrefix, 0) == 0) {
    const std::string ssid = command.substr(std::strlen(kSsidPrefix));
    if (ssid.empty() || ssid.size() > 32) {
      set_message("SSID must be 1..32 bytes");
      return false;
    }
    g_wifi_config.ssid = ssid;
    g_wifi_config.has_saved_config = true;
    const esp_err_t err = esphome::espectre::save_stored_wifi_config(g_wifi_config);
    sync_frontend_wifi_info();
    set_message(err == ESP_OK ? "SSID saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command.rfind(kPasswordPrefix, 0) == 0) {
    const std::string password = command.substr(std::strlen(kPasswordPrefix));
    if (password.size() > 63) {
      set_message("password must be 0..63 bytes");
      return false;
    }
    g_wifi_config.password = password;
    g_wifi_config.has_saved_config = true;
    const esp_err_t err = esphome::espectre::save_stored_wifi_config(g_wifi_config);
    sync_frontend_wifi_info();
    set_message(err == ESP_OK ? "password saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command.rfind(kBssidPrefix, 0) == 0) {
    const std::string bssid = command.substr(std::strlen(kBssidPrefix));
    if (!bssid.empty() && bssid.size() != 17) {
      set_message("BSSID must be empty or 17 chars");
      return false;
    }
    g_wifi_config.bssid = bssid;
    g_wifi_config.has_saved_config = true;
    const esp_err_t err = esphome::espectre::save_stored_wifi_config(g_wifi_config);
    sync_frontend_wifi_info();
    set_message(err == ESP_OK ? "BSSID saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command.rfind(kChannelPrefix, 0) == 0) {
    uint8_t channel = 0;
    if (!parse_wifi_channel(command.substr(std::strlen(kChannelPrefix)), &channel)) {
      set_message("channel must be 0..14");
      return false;
    }
    g_wifi_config.channel = channel;
    g_wifi_config.has_saved_config = true;
    const esp_err_t err = esphome::espectre::save_stored_wifi_config(g_wifi_config);
    sync_frontend_wifi_info();
    set_message(err == ESP_OK ? "channel saved" : esp_err_to_name(err));
    return err == ESP_OK;
  }

  if (command == "CLEAR_WIFI") {
    const esp_err_t err = esphome::espectre::clear_stored_wifi_config();
    if (err != ESP_OK) {
      set_message(esp_err_to_name(err));
      return false;
    }
    g_wifi_config = esphome::espectre::StoredWifiConfig{};
    sync_frontend_wifi_info();
    return apply_wifi_config_live(message);
  }

  if (command == "APPLY_WIFI") {
    if (g_wifi_config.ssid.empty()) {
      set_message("SSID is empty");
      return false;
    }
    const esp_err_t err = esphome::espectre::save_stored_wifi_config(g_wifi_config);
    if (err != ESP_OK) {
      set_message(esp_err_to_name(err));
      return false;
    }
    g_wifi_config.has_saved_config = true;
    sync_frontend_wifi_info();
    return apply_wifi_config_live(message);
  }

  set_message("unknown provisioning command");
  return false;
}

bool handle_device_config_change(const esphome::espectre::EspectreDeviceConfig &config, bool clear, std::string *message) {
  if (clear) {
    const esp_err_t err = esphome::espectre::clear_stored_device_config();
    if (message != nullptr) {
      *message = err == ESP_OK ? "device config cleared" : esp_err_to_name(err);
    }
    return err == ESP_OK;
  }

  const esp_err_t err = esphome::espectre::save_stored_device_config(config);
  if (message != nullptr) {
    *message = err == ESP_OK ? "device config saved" : esp_err_to_name(err);
  }
  return err == ESP_OK;
}

}  // namespace

extern "C" void app_main() {
  esp_err_t err = nvs_flash_init();
  if (err == ESP_ERR_NVS_NO_FREE_PAGES || err == ESP_ERR_NVS_NEW_VERSION_FOUND) {
    ESP_ERROR_CHECK(nvs_flash_erase());
    err = nvs_flash_init();
  }
  ESP_ERROR_CHECK(err);

  if (!init_wifi_station()) {
    ESP_LOGE(TAG, "Failed to initialize Wi-Fi station");
    return;
  }

  static esphome::espectre::NimbleBleBindings bindings;
  static esphome::espectre::EspIdfMqttTransport mqtt_transport;
  static esphome::espectre::BleFrontend frontend(&bindings, &mqtt_transport);
  frontend.set_runtime_config(make_runtime_config());
  frontend.set_device_config(make_device_config());
  g_frontend = &frontend;
  sync_frontend_wifi_info();
  frontend.set_provisioning_command_callback(handle_wifi_provisioning_command);
  frontend.set_device_config_change_callback(handle_device_config_change);

  ESP_LOGI(TAG, "ESPectre BLE smoke marker: transport configured, starting BLE frontend");
  if (!frontend.setup()) {
    ESP_LOGE(TAG, "Failed to initialize ESPectre BLE frontend");
    return;
  }

  ESP_ERROR_CHECK(g_wifi_manager.start());
  xTaskCreate(espectre_loop_task, "espectre_ble_loop", 8192, nullptr, 5, nullptr);
  ESP_LOGI(TAG, "ESPectre BLE firmware started");
}
