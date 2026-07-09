/*
 * ESPectre streamer firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "stream_frontend.h"

#include <esp_log.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#if CONFIG_ESPECTRE_STREAMER_BLE_ENABLED
#include "ble_bindings_nimble.h"
#endif
#if CONFIG_ESPECTRE_STREAMER_OTA_ENABLED
#include "https_ota_service.h"
#endif
#if CONFIG_ESPECTRE_STREAMER_MQTT_ENABLED
#include "mqtt_transport_esp_idf.h"
#endif
#include "espectre_banner.h"

static const char *TAG = "espectre.streamer.app";

extern "C" void app_main() {
  esphome::espectre::log_espectre_banner([](const char *line) { ESP_LOGI(TAG, "%s", line); });
#if CONFIG_ESPECTRE_STREAMER_BLE_ENABLED
  static esphome::espectre::NimbleBleBindings ble_bindings;
  auto *ble_bindings_ptr = &ble_bindings;
#else
  esphome::espectre::IBleBindings *ble_bindings_ptr = nullptr;
#endif
#if CONFIG_ESPECTRE_STREAMER_MQTT_ENABLED
  static esphome::espectre::EspIdfMqttTransport mqtt_transport;
#endif
#if CONFIG_ESPECTRE_STREAMER_OTA_ENABLED
  static esphome::espectre::HttpsOtaService ota_service;
#endif
#if CONFIG_ESPECTRE_STREAMER_MQTT_ENABLED
  auto *mqtt_transport_ptr = &mqtt_transport;
#else
  esphome::espectre::IMqttTransport *mqtt_transport_ptr = nullptr;
#endif
#if CONFIG_ESPECTRE_STREAMER_OTA_ENABLED
  auto *ota_service_ptr = &ota_service;
#else
  esphome::espectre::IOtaService *ota_service_ptr = nullptr;
#endif
  static esphome::espectre::StreamFrontend frontend(ble_bindings_ptr, mqtt_transport_ptr, ota_service_ptr);
  if (!frontend.setup()) {
    return;
  }

  while (true) {
    frontend.loop();
    // Keep the control loop responsive so the external stimulus socket is
    // drained quickly and collector address changes propagate with low jitter.
    vTaskDelay(pdMS_TO_TICKS(2));
  }
}
