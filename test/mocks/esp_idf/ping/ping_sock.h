/*
 * Mock for ESP-IDF ping/ping_sock.h
 * 
 * Provides stubs for esp_ping API used in traffic_generator_manager.cpp
 */

#pragma once

#include "esp_err.h"
#include "lwip/ip_addr.h"

#ifdef __cplusplus
extern "C" {
#endif

// Ping handle type
typedef void* esp_ping_handle_t;

// Infinite ping count
#define ESP_PING_COUNT_INFINITE 0

// Ping configuration
typedef struct {
    ip_addr_t target_addr;
    uint32_t count;
    uint32_t interval_ms;
    uint32_t timeout_ms;
    uint32_t data_size;
    uint32_t task_stack_size;
    uint32_t task_prio;
} esp_ping_config_t;

// Default config macro
#define ESP_PING_DEFAULT_CONFIG() { \
    .target_addr = {0}, \
    .count = 5, \
    .interval_ms = 1000, \
    .timeout_ms = 1000, \
    .data_size = 64, \
    .task_stack_size = 2048, \
    .task_prio = 2, \
}

// Ping callbacks
typedef struct {
    void *cb_args;
    void (*on_ping_success)(esp_ping_handle_t hdl, void *args);
    void (*on_ping_timeout)(esp_ping_handle_t hdl, void *args);
    void (*on_ping_end)(esp_ping_handle_t hdl, void *args);
} esp_ping_callbacks_t;

// Mock function declarations (stubs)
static inline esp_err_t esp_ping_new_session(const esp_ping_config_t *config, 
                                              const esp_ping_callbacks_t *cbs,
                                              esp_ping_handle_t *hdl_out) {
    (void)config;
    (void)cbs;
    *hdl_out = (void*)0x12345678;  // Fake handle
    return ESP_OK;
}

static inline esp_err_t esp_ping_start(esp_ping_handle_t hdl) {
    (void)hdl;
    return ESP_OK;
}

static inline esp_err_t esp_ping_stop(esp_ping_handle_t hdl) {
    (void)hdl;
    return ESP_OK;
}

static inline esp_err_t esp_ping_delete_session(esp_ping_handle_t hdl) {
    (void)hdl;
    return ESP_OK;
}

#ifdef __cplusplus
}
#endif

