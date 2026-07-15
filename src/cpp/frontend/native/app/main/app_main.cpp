/*
 * ESPectre - Native Firmware Entrypoint
 *
 * Native firmware application entrypoint.
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
#include "espectre_banner.h"
#include "firmware_version.h"
#include "frontend_bootstrap_helpers.h"
#include "ota_service_https.h"
#include "mqtt_transport_esp_idf.h"
#include "runtime_sensing_kconfig.h"
#include "standalone_wifi_service.h"
#include "wifi_provisioning_service.h"

static const char *TAG = "espectre.native.app";

namespace {

#ifdef ESPECTRE_OTA_SNAPSHOT_BUILD
constexpr espectre::OtaReleaseChannel kOtaReleaseChannel = espectre::OtaReleaseChannel::SNAPSHOT;
#else
constexpr espectre::OtaReleaseChannel kOtaReleaseChannel = espectre::OtaReleaseChannel::STABLE;
#endif

constexpr int kWifiConnectMaxRetry = 8;

espectre::NativeFrontend *g_frontend = nullptr;
espectre::StandaloneWifiService g_wifi_manager;
espectre::WifiProvisioningService g_wifi_provisioning(&g_wifi_manager);

void sync_frontend_wifi_info() {
  if (g_frontend == nullptr) {
    return;
  }
  espectre::NativeFrontend::WifiProvisioningInfo info;
  const espectre::StoredWifiConfig &wifi_config = g_wifi_provisioning.config();
  info.ssid = wifi_config.ssid;
  info.bssid = wifi_config.bssid;
  info.channel = wifi_config.channel;
  info.has_saved_config = wifi_config.has_saved_config;
  info.password_set = g_wifi_provisioning.password_set();
  g_frontend->set_wifi_provisioning_info(info);

  espectre::EspectreDeviceInfo device_info;
  device_info.frontend = "native";
  device_info.firmware_version = espectre::espectre_firmware_version();
  device_info.chip = CONFIG_IDF_TARGET;

  espectre::StandaloneWifiInfo wifi_info;
  if (g_wifi_manager.get_info(&wifi_info)) {
    device_info.network.ip_address = wifi_info.ip_address;
    device_info.network.mac_address = wifi_info.mac_address;
    device_info.network.channel = wifi_info.channel;
  }
  g_frontend->set_device_info(device_info);
}

espectre::RuntimeConfig make_runtime_config() { return espectre::make_runtime_sensing_config_from_kconfig(); }

espectre::EspectreDeviceConfig make_device_config() {
  return espectre::load_frontend_device_config(espectre::FrontendDeviceConfigDefaults{
                                                            CONFIG_ESPECTRE_DEVICE_LABEL,
                                                            CONFIG_ESPECTRE_MQTT_HOST,
                                                            CONFIG_ESPECTRE_MQTT_PORT,
                                                            CONFIG_ESPECTRE_TOPIC_PREFIX,
                                                            espectre::derive_runtime_device_id(),
                                                        },
                                                        TAG,
                                                        "Using ESPectre Protocol config provisioned over BLE",
                                                        "Failed to load BLE-provisioned device config");
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
  const esp_err_t setup_err = espectre::setup_frontend_wifi_station(
      &g_wifi_provisioning,
      &g_wifi_manager,
      espectre::FrontendWifiStationOptions{CONFIG_ESPECTRE_WIFI_SSID,
                                                    CONFIG_ESPECTRE_WIFI_PASSWORD,
                                                    CONFIG_ESPECTRE_WIFI_BSSID,
                                                    CONFIG_ESPECTRE_WIFI_CHANNEL,
                                                    kWifiConnectMaxRetry,
                                                    false,
                                                    false,
                                                    sync_frontend_wifi_info,
                                                    sync_frontend_wifi_info,
                                                    sync_frontend_wifi_info},
      TAG,
      "Using Wi-Fi credentials provisioned over BLE");
  if (setup_err != ESP_OK) {
    ESP_LOGW(TAG, "Failed to initialize Wi-Fi provisioning service: %s", esp_err_to_name(setup_err));
    return false;
  }
  sync_frontend_wifi_info();
  return true;
}

bool handle_wifi_provisioning_command(const std::string &command, std::string *message) {
  return g_wifi_provisioning.handle_command(command, message);
}

bool handle_device_config_change(const espectre::EspectreDeviceConfig &config, bool clear, std::string *message) {
  if (clear) {
    const esp_err_t err = espectre::clear_stored_device_config();
    if (message != nullptr) {
      *message = err == ESP_OK ? "device config cleared" : esp_err_to_name(err);
    }
    return err == ESP_OK;
  }

  const esp_err_t err = espectre::save_stored_device_config(config);
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

  espectre::log_espectre_banner([](const char *line) { ESP_LOGI(TAG, "%s", line); });

  if (!init_wifi_station()) {
    ESP_LOGE(TAG, "Failed to initialize Wi-Fi station");
    return;
  }

#if CONFIG_BT_ENABLED
  static espectre::NimbleBleBindings bindings;
#else
  static espectre::NoopBleBindings bindings;
#endif
  static espectre::EspIdfMqttTransport mqtt_transport;
  static espectre::HttpsOtaService ota_service("native", CONFIG_IDF_TARGET, kOtaReleaseChannel);
  static espectre::NativeFrontend frontend(&bindings, &mqtt_transport, &ota_service);
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
