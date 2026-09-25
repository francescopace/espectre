/*
 * ESPectre - Mock task.h
 *
 * Host-side mock of task.h for native C++ tests.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#ifndef FREERTOS_TASK_H
#define FREERTOS_TASK_H

#include "FreeRTOS.h"

#ifdef __cplusplus
extern "C" {
#endif

// Task creation
typedef void (*TaskFunction_t)(void *);

#ifdef __cplusplus
struct FreeRtosTaskMock {
  BaseType_t create_result{pdPASS};
  bool defer_execution{false};
  // When false, vTaskSuspend records the call but eTaskGetState stays eRunning
  // until the test sets suspended. That is the window before the scheduler parks the task.
  bool reveal_suspend{true};
  bool suspended{false};
  unsigned create_calls{0U};
  unsigned delete_calls{0U};
  unsigned suspend_calls{0U};
  TaskFunction_t pending_function{nullptr};
  void *pending_argument{nullptr};
};

inline FreeRtosTaskMock g_freertos_task_mock;
#endif

static inline BaseType_t xTaskCreate(TaskFunction_t pvTaskCode,
                                     const char *const pcName,
                                     uint32_t usStackDepth, void *pvParameters,
                                     UBaseType_t uxPriority,
                                     TaskHandle_t *pxCreatedTask) {
  (void)pcName;
  (void)usStackDepth;
  (void)uxPriority;
#ifdef __cplusplus
  ++g_freertos_task_mock.create_calls;
  if (g_freertos_task_mock.create_result != pdPASS) {
    return g_freertos_task_mock.create_result;
  }
  g_freertos_task_mock.suspended = false;
  g_freertos_task_mock.suspend_calls = 0U;
  if (g_freertos_task_mock.defer_execution) {
    // A deferred task has a handle, so its owner can delete it before it runs.
    g_freertos_task_mock.pending_function = pvTaskCode;
    g_freertos_task_mock.pending_argument = pvParameters;
    if (pxCreatedTask != NULL) *pxCreatedTask = &g_freertos_task_mock;
    return pdPASS;
  }
#endif
  (void)pxCreatedTask;
  // For testing: execute task function synchronously instead of in a thread.
  if (pvTaskCode != NULL) {
    pvTaskCode(pvParameters);
  }
  return pdPASS;
}

// Task delay
#ifndef pdMS_TO_TICKS
#define pdMS_TO_TICKS(ms) ((ms) * CONFIG_FREERTOS_HZ / 1000)
#endif

static inline BaseType_t xTaskNotifyGive(TaskHandle_t xTaskToNotify) {
  (void)xTaskToNotify;
  return pdPASS;
}

static inline uint32_t ulTaskNotifyTake(BaseType_t xClearCountOnExit, TickType_t xTicksToWait) {
  (void)xClearCountOnExit;
  (void)xTicksToWait;
  return 0U;
}

// Task deletion
static inline void vTaskDelete(TaskHandle_t xTask) {
#ifdef __cplusplus
  // Deleting the deferred task by handle means it never runs.
  if (xTask != NULL && xTask == &g_freertos_task_mock) {
    ++g_freertos_task_mock.delete_calls;
    g_freertos_task_mock.pending_function = nullptr;
    g_freertos_task_mock.suspended = false;
  }
#endif
  (void)xTask;
}

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
#ifdef __cplusplus
  if (xTask == &g_freertos_task_mock && g_freertos_task_mock.suspended) {
    return eSuspended;
  }
#endif
  (void)xTask;
  return eRunning;
}

// Task suspend/resume
static inline void vTaskSuspend(TaskHandle_t xTaskToSuspend) {
#ifdef __cplusplus
  // NULL suspends the calling task. Host tests run that task on this thread.
  if (xTaskToSuspend == NULL || xTaskToSuspend == &g_freertos_task_mock) {
    ++g_freertos_task_mock.suspend_calls;
    if (g_freertos_task_mock.reveal_suspend) {
      g_freertos_task_mock.suspended = true;
    }
  }
#else
  (void)xTaskToSuspend;
#endif
}

static inline void vTaskResume(TaskHandle_t xTaskToResume) {
  (void)xTaskToResume;
}

#ifdef __cplusplus
}
#endif

#endif // FREERTOS_TASK_H
