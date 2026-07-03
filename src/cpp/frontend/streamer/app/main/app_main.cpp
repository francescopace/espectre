/*
 * ESPectre streamer firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "stream_frontend.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "https_ota_service.h"
#include "mqtt_transport_esp_idf.h"

extern "C" void app_main() {
  static esphome::espectre::EspIdfMqttTransport mqtt_transport;
  static esphome::espectre::HttpsOtaService ota_service;
  static esphome::espectre::StreamFrontend frontend(&mqtt_transport, &ota_service);
  if (!frontend.setup()) {
    return;
  }

  while (true) {
    frontend.loop();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}
