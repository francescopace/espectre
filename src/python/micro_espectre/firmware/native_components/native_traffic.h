// SPDX-License-Identifier: GPL-3.0-only
// Commercial licensing available under separate agreement; see LICENSING.md.

#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  ESPECTRE_NATIVE_TRAFFIC_PING = 0,
  ESPECTRE_NATIVE_TRAFFIC_DNS = 1,
  ESPECTRE_NATIVE_TRAFFIC_DNS_TCP = 2,
} espectre_native_traffic_mode_t;

void *espectre_native_traffic_create(void);
void espectre_native_traffic_destroy(void *handle);
bool espectre_native_traffic_start(
    void *handle,
    uint32_t gateway_addr,
    uint32_t rate_pps,
    espectre_native_traffic_mode_t mode);
void espectre_native_traffic_stop(void *handle);
bool espectre_native_traffic_pause(void *handle);
bool espectre_native_traffic_resume(void *handle);
bool espectre_native_traffic_is_running(void *handle);
uint32_t espectre_native_traffic_packet_count(void *handle);
uint32_t espectre_native_traffic_error_count(void *handle);

#ifdef __cplusplus
}
#endif
