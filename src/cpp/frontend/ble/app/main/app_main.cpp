/*
 * ESPectre BLE firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include <cstdio>

#include <esp_err.h>
#include <esp_event.h>
#include <esp_log.h>
#include <esp_netif.h>
#include <esp_wifi.h>
#include <nvs_flash.h>

#include "ble_bindings_nimble.h"
#include "ble_frontend.h"

static const char *TAG = "espectre.ble.app";

namespace {

constexpr int kWifiConnectMaxRetry = 8;

esphome::espectre::BleFrontend *g_frontend = nullptr;
int g_wifi_retry_count = 0;

bool parse_bssid(const char *text, uint8_t out[6]) {
  if (text == nullptr || out == nullptr || text[0] == '\0') {
    return false;
  }

  unsigned int bytes[6] = {0U, 0U, 0U, 0U, 0U, 0U};
  if (std::sscanf(text, "%2x:%2x:%2x:%2x:%2x:%2x", &bytes[0], &bytes[1], &bytes[2], &bytes[3], &bytes[4],
                  &bytes[5]) != 6) {
    return false;
  }

  for (size_t i = 0; i < 6; i++) {
    out[i] = static_cast<uint8_t>(bytes[i]);
  }
  return true;
}

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

void wifi_event_handler(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data) {
  (void) arg;
  if (event_base != WIFI_EVENT) {
    return;
  }

  if (event_id == WIFI_EVENT_STA_START) {
    if (CONFIG_ESPECTRE_WIFI_SSID[0] == '\0') {
      ESP_LOGW(TAG, "Wi-Fi SSID is empty; configure sdkconfig credentials before using the BLE frontend");
      return;
    }
    (void) esp_wifi_connect();
    return;
  }

  if (event_id == WIFI_EVENT_STA_DISCONNECTED) {
    const auto *event = static_cast<const wifi_event_sta_disconnected_t *>(event_data);
    ESP_LOGW(TAG, "Wi-Fi disconnected: reason=%u", event != nullptr ? static_cast<unsigned>(event->reason) : 0U);
    if (CONFIG_ESPECTRE_WIFI_SSID[0] != '\0' && g_wifi_retry_count < kWifiConnectMaxRetry) {
      g_wifi_retry_count++;
      (void) esp_wifi_connect();
    }
  }
}

bool init_wifi_station() {
  ESP_ERROR_CHECK(esp_netif_init());
  const esp_err_t loop_err = esp_event_loop_create_default();
  if (loop_err != ESP_OK && loop_err != ESP_ERR_INVALID_STATE) {
    ESP_LOGE(TAG, "esp_event_loop_create_default failed: %s", esp_err_to_name(loop_err));
    return false;
  }
  if (esp_netif_create_default_wifi_sta() == nullptr) {
    ESP_LOGE(TAG, "esp_netif_create_default_wifi_sta failed");
    return false;
  }

  wifi_init_config_t wifi_cfg = WIFI_INIT_CONFIG_DEFAULT();
  ESP_ERROR_CHECK(esp_wifi_init(&wifi_cfg));
  ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
  ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
  ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));

  ESP_ERROR_CHECK(esp_event_handler_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, nullptr));

  wifi_config_t sta_cfg{};
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.ssid), sizeof(sta_cfg.sta.ssid), "%s", CONFIG_ESPECTRE_WIFI_SSID);
  std::snprintf(reinterpret_cast<char *>(sta_cfg.sta.password), sizeof(sta_cfg.sta.password), "%s",
                CONFIG_ESPECTRE_WIFI_PASSWORD);
  sta_cfg.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;
  sta_cfg.sta.sae_pwe_h2e = WPA3_SAE_PWE_BOTH;
  sta_cfg.sta.pmf_cfg.capable = true;
  sta_cfg.sta.pmf_cfg.required = false;

  if (CONFIG_ESPECTRE_WIFI_BSSID[0] != '\0') {
    if (!parse_bssid(CONFIG_ESPECTRE_WIFI_BSSID, sta_cfg.sta.bssid)) {
      ESP_LOGE(TAG, "Invalid BSSID format: %s", CONFIG_ESPECTRE_WIFI_BSSID);
      return false;
    }
    sta_cfg.sta.bssid_set = true;
  }

  ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &sta_cfg));
  return true;
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

  ESP_ERROR_CHECK(esp_wifi_start());
  xTaskCreate(espectre_loop_task, "espectre_ble_loop", 8192, nullptr, 5, nullptr);
  ESP_LOGI(TAG, "ESPectre BLE firmware started");
}
