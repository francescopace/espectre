/*
 * ESPectre streamer firmware entrypoint.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "stream_frontend.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern "C" void app_main() {
  static esphome::espectre::StreamFrontend frontend;
  if (!frontend.setup()) {
    return;
  }

  while (true) {
    frontend.loop();
    vTaskDelay(pdMS_TO_TICKS(10));
  }
}
