/*
 * ESPectre - Mock esp_attr.h
 *
 * Provides empty definitions for ESP-IDF attributes that are
 * not meaningful on native platforms.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

// IRAM_ATTR: On ESP32, places function in IRAM for ISR safety
// On native: no-op (functions are always in RAM)
#ifndef IRAM_ATTR
#define IRAM_ATTR
#endif

// Other common ESP-IDF attributes (add as needed)
#ifndef DRAM_ATTR
#define DRAM_ATTR
#endif

#ifndef RTC_DATA_ATTR
#define RTC_DATA_ATTR
#endif

#ifndef RTC_NOINIT_ATTR
#define RTC_NOINIT_ATTR
#endif

#ifndef EXT_RAM_ATTR
#define EXT_RAM_ATTR
#endif

