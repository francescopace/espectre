/*
 * ESPectre native firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <cstdio>
#include <string>

#include <esp_err.h>
#include <esp_log.h>
#include <nvs_flash.h>

#if CONFIG_BT_ENABLED
#include "ble_bindings_nimble.h"
#else
#include "ble_bindings_noop.h"
#endif
#include "native_frontend.h"
#include "device_config_store.h"
#include "device_identity.h"
#include "firmware_version.h"
#include "https_ota_service.h"
#include "mqtt_transport_esp_idf.h"
#include "standalone_wifi_manager.h"
#include "wifi_provisioning_service.h"

static const char *TAG = "espectre.native.app";

namespace {

constexpr int kWifiConnectMaxRetry = 8;

esphome::espectre::NativeFrontend *g_frontend = nullptr;
esphome::espectre::StandaloneWifiManager g_wifi_manager;
esphome::espectre::WifiProvisioningService g_wifi_provisioning(&g_wifi_manager);

void sync_frontend_wifi_info() {
  if (g_frontend == nullptr) {
    return;
  }
  esphome::espectre::NativeFrontend::WifiProvisioningInfo info;
  const esphome::espectre::StoredWifiConfig &wifi_config = g_wifi_provisioning.config();
  info.ssid = wifi_config.ssid;
  info.bssid = wifi_config.bssid;
  info.channel = wifi_config.channel;
  info.has_saved_config = wifi_config.has_saved_config;
  info.password_set = g_wifi_provisioning.password_set();
  g_frontend->set_wifi_provisioning_info(info);

  esphome::espectre::EspectreDeviceInfo device_info;
  device_info.frontend = "native";
  device_info.firmware_version = esphome::espectre::espectre_firmware_version();
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

esphome::espectre::EspectreDeviceConfig make_device_config() {
  esphome::espectre::EspectreDeviceConfig config;
  config.device_id = esphome::espectre::derive_runtime_device_id();
  config.device_label = CONFIG_ESPECTRE_DEVICE_LABEL;
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
  config.device_id = esphome::espectre::derive_runtime_device_id();
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

  esphome::espectre::WifiProvisioningDefaults defaults;
  defaults.ssid = CONFIG_ESPECTRE_WIFI_SSID;
  defaults.password = CONFIG_ESPECTRE_WIFI_PASSWORD;
  defaults.bssid = CONFIG_ESPECTRE_WIFI_BSSID;
  defaults.channel = static_cast<uint8_t>(kConfiguredWifiChannel);
  defaults.max_retry = kWifiConnectMaxRetry;
  defaults.manage_csi_lifecycle = false;

  g_wifi_provisioning.set_change_callback(sync_frontend_wifi_info);
  const esp_err_t setup_err = g_wifi_provisioning.setup_station(defaults, sync_frontend_wifi_info, sync_frontend_wifi_info);
  if (setup_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to initialize Wi-Fi provisioning service: %s", esp_err_to_name(setup_err));
    return false;
  }
  if (g_wifi_provisioning.config().has_saved_config) {
    ESP_LOGI(TAG, "Using Wi-Fi credentials provisioned over BLE");
  }
  sync_frontend_wifi_info();
  return true;
}

bool handle_wifi_provisioning_command(const std::string &command, std::string *message) {
  return g_wifi_provisioning.handle_command(command, message);
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

#if CONFIG_BT_ENABLED
  static esphome::espectre::NimbleBleBindings bindings;
#else
  static esphome::espectre::NoopBleBindings bindings;
#endif
  static esphome::espectre::EspIdfMqttTransport mqtt_transport;
  static esphome::espectre::HttpsOtaService ota_service;
  static esphome::espectre::NativeFrontend frontend(&bindings, &mqtt_transport, &ota_service);
  frontend.set_runtime_config(make_runtime_config());
  frontend.set_device_config(make_device_config());
  g_frontend = &frontend;
  sync_frontend_wifi_info();
  frontend.set_provisioning_command_callback(handle_wifi_provisioning_command);
  frontend.set_device_config_change_callback(handle_device_config_change);

  ESP_LOGI(TAG, "ESPectre native smoke marker: transport configured, starting native frontend");
  if (!frontend.setup()) {
    ESP_LOGE(TAG, "Failed to initialize ESPectre native frontend");
    return;
  }

  ESP_ERROR_CHECK(g_wifi_manager.start());
  xTaskCreate(espectre_loop_task, "espectre_native_loop", 8192, nullptr, 5, nullptr);
  ESP_LOGI(TAG, "ESPectre native firmware started");
}
