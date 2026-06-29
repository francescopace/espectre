/*
 * ESPectre BLE firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <cstdio>

#include <esp_err.h>
#include <esp_log.h>
#include <nvs_flash.h>

#include "ble_bindings_nimble.h"
#include "ble_frontend.h"
#include "standalone_wifi_manager.h"

static const char *TAG = "espectre.ble.app";

namespace {

constexpr int kWifiConnectMaxRetry = 8;

esphome::espectre::BleFrontend *g_frontend = nullptr;
esphome::espectre::StandaloneWifiManager g_wifi_manager;

esphome::espectre::RuntimeConfig make_runtime_config() {
  esphome::espectre::RuntimeConfig config;
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
  esphome::espectre::StandaloneWifiConfig wifi_config;
  wifi_config.ssid = CONFIG_ESPECTRE_WIFI_SSID;
  wifi_config.password = CONFIG_ESPECTRE_WIFI_PASSWORD;
  wifi_config.bssid = CONFIG_ESPECTRE_WIFI_BSSID;
  wifi_config.channel = static_cast<uint8_t>(kConfiguredWifiChannel);
  wifi_config.max_retry = kWifiConnectMaxRetry;
  wifi_config.manage_csi_lifecycle = false;
  return g_wifi_manager.setup(wifi_config) == ESP_OK;
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
  static esphome::espectre::BleFrontend frontend(&bindings);
  frontend.set_runtime_config(make_runtime_config());
  g_frontend = &frontend;

  ESP_LOGI(TAG, "ESPectre BLE smoke marker: transport configured, starting BLE frontend");
  if (!frontend.setup()) {
    ESP_LOGE(TAG, "Failed to initialize ESPectre BLE frontend");
    return;
  }

  ESP_ERROR_CHECK(g_wifi_manager.start());
  xTaskCreate(espectre_loop_task, "espectre_ble_loop", 8192, nullptr, 5, nullptr);
  ESP_LOGI(TAG, "ESPectre BLE firmware started");
}
