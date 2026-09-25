/*
 * ESPectre - Protocol SDK
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

/**
 * @file espectre_protocol_sdk.h
 * @brief The ESPectre Protocol: messages, JSON, diagnostics, and transports.
 *
 * Includes the sensing SDK plus the canonical protocol messages, their JSON
 * form, the diagnostic field catalog, and the Direct HTTP and MQTT transport
 * contracts you implement to reach your own transports. Include it only when
 * your firmware speaks the ESPectre Protocol; the services and MQTT facades
 * include it for you.
 */
#include "espectre_sdk.h"
#include "runtime/diagnostic_fields.h"
#include "runtime/direct_http_protocol.h"
#include "runtime/direct_http_service.h"
#include "runtime/espectre_protocol.h"
#include "runtime/mqtt_transport.h"
#include "runtime/protocol_json.h"
#include "runtime/runtime_diagnostics_protocol.h"
