/*
 * ESP-IDF semaphore mock for host tests.
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <stdlib.h>

#include "FreeRTOS.h"

#ifdef __cplusplus
inline bool g_freertos_fail_next_take = false;
#endif

static inline SemaphoreHandle_t xSemaphoreCreateMutex(void) { return malloc(1U); }
static inline SemaphoreHandle_t xSemaphoreCreateBinary(void) { return malloc(1U); }
static inline void vSemaphoreDelete(SemaphoreHandle_t semaphore) { free(semaphore); }
static inline BaseType_t xSemaphoreTake(SemaphoreHandle_t semaphore, TickType_t ticks) {
  (void) ticks;
#ifdef __cplusplus
  if (g_freertos_fail_next_take) {
    g_freertos_fail_next_take = false;
    return pdFALSE;
  }
#endif
  return semaphore != NULL ? pdTRUE : pdFALSE;
}
static inline BaseType_t xSemaphoreGive(SemaphoreHandle_t semaphore) {
  return semaphore != NULL ? pdTRUE : pdFALSE;
}
