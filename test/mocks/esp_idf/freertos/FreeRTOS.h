#ifndef FREERTOS_H
#define FREERTOS_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// FreeRTOS types
typedef void *TaskHandle_t;
typedef void *QueueHandle_t;
typedef void *SemaphoreHandle_t;
typedef void *TimerHandle_t;
typedef uint32_t TickType_t;
typedef uint32_t BaseType_t;
typedef uint32_t UBaseType_t;

// FreeRTOS constants
#define pdTRUE 1
#define pdFALSE 0
#define pdPASS pdTRUE
#define pdFAIL pdFALSE

#define portMAX_DELAY 0xFFFFFFFF
#define portTICK_PERIOD_MS (1000 / CONFIG_FREERTOS_HZ)

// Task priorities
#define tskIDLE_PRIORITY 0
#define configMAX_PRIORITIES 25

// Mock FreeRTOS functions (no-ops for testing)
static inline void vTaskDelay(TickType_t xTicksToDelay) { (void)xTicksToDelay; }

static inline TickType_t xTaskGetTickCount(void) { return 0; }

static inline void taskYIELD(void) {
  // No-op
}

#ifdef __cplusplus
}
#endif

#endif // FREERTOS_H
