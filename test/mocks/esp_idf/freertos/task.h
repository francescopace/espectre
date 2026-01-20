#ifndef FREERTOS_TASK_H
#define FREERTOS_TASK_H

#include "FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

// Task creation
typedef void (*TaskFunction_t)(void *);

static inline BaseType_t xTaskCreate(TaskFunction_t pvTaskCode,
                                     const char *const pcName,
                                     uint32_t usStackDepth, void *pvParameters,
                                     UBaseType_t uxPriority,
                                     TaskHandle_t *pxCreatedTask) {
  (void)pcName;
  (void)usStackDepth;
  (void)uxPriority;
  (void)pxCreatedTask;
  // For testing: execute task function synchronously instead of in a thread
  // This allows P95Calibrator/NBVICalibrator to work in native tests without real FreeRTOS
  if (pvTaskCode != NULL) {
    pvTaskCode(pvParameters);
  }
  return pdPASS;
}

// Task delay
#define pdMS_TO_TICKS(ms) ((ms) * CONFIG_FREERTOS_HZ / 1000)

// Task deletion
static inline void vTaskDelete(TaskHandle_t xTask) { (void)xTask; }

// Task state
typedef enum {
  eRunning = 0,
  eReady,
  eBlocked,
  eSuspended,
  eDeleted,
  eInvalid
} eTaskState;

static inline eTaskState eTaskGetState(TaskHandle_t xTask) {
  (void)xTask;
  // For testing, return eRunning (not deleted)
  return eRunning;
}

// Task suspend/resume
static inline void vTaskSuspend(TaskHandle_t xTaskToSuspend) {
  (void)xTaskToSuspend;
}

static inline void vTaskResume(TaskHandle_t xTaskToResume) {
  (void)xTaskToResume;
}

#ifdef __cplusplus
}
#endif

#endif // FREERTOS_TASK_H
