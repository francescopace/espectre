// Dummy main for PlatformIO
// Tests are in test/ directory

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

extern "C" void app_main(void) {
    // Empty - tests run separately via PlatformIO test framework
    while(1) {
        vTaskDelay(pdMS_TO_TICKS(1000));
    }
}
