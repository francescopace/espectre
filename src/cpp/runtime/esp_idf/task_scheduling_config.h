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
/** Default Direct HTTP server priority when the SDK Kconfig is not sourced. */
#define CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY 1
#endif

#ifndef CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY
/** Default Direct response and event worker priority when the SDK Kconfig is not sourced. */
#define CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY 2
#endif

#ifndef CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY
/** Default raw CSI delivery worker priority when the SDK Kconfig is not sourced. */
#define CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY 3
#endif

#ifndef CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY
/** Default managed traffic generator priority when the SDK Kconfig is not sourced. */
#define CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY 1
#endif

/**
 * FreeRTOS priorities of the tasks ESPectre services create, resolved from the
 * `Advanced task scheduling` menu. See the integration guide for the policy.
 */
namespace espectre::task_scheduling {

/** Direct HTTP server task. */
inline constexpr uint32_t kDirectHttpdPriority =
    CONFIG_ESPECTRE_DIRECT_HTTPD_TASK_PRIORITY;
/** Direct response and event delivery task. */
inline constexpr uint32_t kDirectWorkerPriority =
    CONFIG_ESPECTRE_DIRECT_WORKER_TASK_PRIORITY;
/** Raw CSI delivery task. */
inline constexpr uint32_t kRawWorkerPriority =
    CONFIG_ESPECTRE_RAW_WORKER_TASK_PRIORITY;
/** Managed traffic generator task. */
inline constexpr uint32_t kTrafficPriority = CONFIG_ESPECTRE_TRAFFIC_TASK_PRIORITY;

}  // namespace espectre::task_scheduling
