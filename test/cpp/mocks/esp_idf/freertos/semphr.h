/*
 * ESP-IDF semaphore mock for host tests.
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <stdlib.h>

#include "FreeRTOS.h"

static inline SemaphoreHandle_t xSemaphoreCreateMutex(void) { return malloc(1U); }
static inline SemaphoreHandle_t xSemaphoreCreateBinary(void) { return malloc(1U); }
static inline void vSemaphoreDelete(SemaphoreHandle_t semaphore) { free(semaphore); }
static inline BaseType_t xSemaphoreTake(SemaphoreHandle_t semaphore, TickType_t ticks) {
  (void) ticks;
  return semaphore != NULL ? pdTRUE : pdFALSE;
}
static inline BaseType_t xSemaphoreGive(SemaphoreHandle_t semaphore) {
  return semaphore != NULL ? pdTRUE : pdFALSE;
}
