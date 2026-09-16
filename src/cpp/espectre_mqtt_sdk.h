/*
 * ESPectre - Optional ESP-IDF MQTT SDK
 *
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file espectre_mqtt_sdk.h
 * @brief Opt-in ESP-IDF implementation of the public MQTT transport contract.
 *
 * Requires ESP-IDF's mqtt component and ESPECTRE_RUNTIME_ESP_IDF_MQTT_SOURCES
 * (or CONFIG_ESPECTRE_SDK_ENABLE_MQTT). The integrating firmware owns the
 * transport object and drives its lifecycle. Other services do not require
 * this facade or the MQTT stack.
 */
#include "espectre_sdk.h"
#include "runtime/esp_idf/mqtt_transport_esp_idf.h"
