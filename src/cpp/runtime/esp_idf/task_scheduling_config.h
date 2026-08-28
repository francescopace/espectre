/*
 * ESPectre - ESP-IDF Task Scheduling Configuration
 *
 * Centralized build-policy defaults for ESPectre-owned FreeRTOS tasks.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>

#ifndef CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY
#define CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY 1
#endif

#ifndef CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY
#define CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY 2
#endif

#ifndef CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY
#define CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY 3
#endif

#ifndef CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY
#define CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY 1
#endif

#ifndef CONFIG_ESPECTRE_NATIVE_LOOP_TASK_PRIORITY
#define CONFIG_ESPECTRE_NATIVE_LOOP_TASK_PRIORITY 5
#endif

namespace espectre::task_scheduling {

inline constexpr uint32_t kDirectHttpdPriority =
    CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY;
inline constexpr uint32_t kDirectWorkerPriority =
    CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY;
inline constexpr uint32_t kRawWorkerPriority =
    CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY;
inline constexpr uint32_t kTrafficPriority = CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY;
inline constexpr uint32_t kNativeLoopPriority =
    CONFIG_ESPECTRE_NATIVE_LOOP_TASK_PRIORITY;

}  // namespace espectre::task_scheduling
