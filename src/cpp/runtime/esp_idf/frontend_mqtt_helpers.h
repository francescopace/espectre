/*
 * ESPectre - Frontend MQTT Helpers
 *
 * Sets up frontend MQTT transport and maps MQTT payloads to the shared
 * frontend command dispatcher.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string>

#include "runtime/frontend_command_engine.h"
#include "runtime/mqtt_transport.h"

namespace espectre {

/** Receives the broker connection state after each change. */
using FrontendMqttConnectedCallback = std::function<void(bool)>;
/**
 * Configure `transport` for the device and start connecting.
 *
 * Shuts the transport down and returns false when `config` has no complete
 * MQTT endpoint. Otherwise installs the callbacks and returns the result of
 * IMqttTransport::setup().
 */
bool setup_frontend_mqtt_transport(IMqttTransport *transport,
                                   const EspectreDeviceConfig &config,
                                   IMqttTransport::CommandCallback command_callback,
                                   FrontendMqttConnectedCallback connected_callback,
                                   const char *log_tag);

/**
 * Publish under the device topic prefix; false while disconnected.
 *
 * The prefix comes from the configuration passed to IMqttTransport::setup().
 */
bool publish_frontend_mqtt_message(IMqttTransport *transport,
                                   const char *suffix,
                                   const std::string &payload,
                                   bool retain);

/** Publish the retained `health` availability message. */
bool publish_frontend_mqtt_status(IMqttTransport *transport,
                                  const EspectreDeviceConfig &config,
                                  bool online,
                                  uint32_t timestamp_ms);

/** Publish a command result on `commands/result`. */
bool publish_frontend_mqtt_command_result(IMqttTransport *transport,
                                          const EspectreDeviceConfig &config,
                                          const FrontendCommandResult &result);

}  // namespace espectre
