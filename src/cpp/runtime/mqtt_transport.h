/*
 * ESPectre - MQTT Transport Boundary
 *
 * Abstract MQTT transport used by native frontends and shared helpers.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <functional>
#include <string>

#include "espectre_protocol.h"

namespace espectre {

/**
 * The MQTT client seam.
 *
 * Implement it to carry ESPectre Protocol messages over an MQTT stack you
 * already own, then hand the instance to a frontend. `EspIdfMqttTransport`
 * (`mqtt_transport_esp_idf.h`) is the shipped implementation over `esp-mqtt`,
 * and `test/cpp/support/mqtt_transport_mock.h` is the host double.
 *
 * Topic layout and payload schemas live in `docs/ESPECTRE_PROTOCOL.md`, and
 * `espectre_protocol.h` builds the payloads, so an implementation only has to
 * move bytes.
 *
 * @par Contract for implementers
 * - The transport is driven from the frontend's task: `loop()` is where you
 *   pump your client and deliver queued callbacks.
 * - Publishing while disconnected must fail cleanly rather than block.
 * - Registered subscriptions must survive a reconnect. Callers subscribe once
 *   and expect the broker subscription to be reissued on the next connect.
 */
class IMqttTransport {
 public:
  /** Payload of a message on the device command topic. */
  using CommandCallback = std::function<void(const std::string &)>;
  /** Broker connection state changed; the argument is the new state. */
  using ConnectionCallback = std::function<void(bool connected)>;
  /** Message on a topic registered through `subscribe()`: `(topic, payload)`. */
  using MessageCallback = std::function<void(const std::string &, const std::string &)>;

  virtual ~IMqttTransport() = default;

  /**
   * Configure and start connecting.
   *
   * Asynchronous: true means the client started, not that it reached the
   * broker. Wait for the connection callback before expecting publishes to
   * land. Calling it again reconfigures and tears down the previous client.
   *
   * @return false when the configuration cannot produce a client, such as an
   *         empty `EspectreDeviceConfig::mqtt_host`.
   */
  virtual bool setup(const EspectreDeviceConfig &config) = 0;
  /** Pump the client and dispatch callbacks. Called from the frontend loop. */
  virtual void loop() = 0;
  /** Disconnect and release resources. Safe to repeat. */
  virtual void shutdown() = 0;
  /** True while the broker connection is established. */
  virtual bool connected() const = 0;
  /**
   * Publish to an absolute topic.
   *
   * @param topic Full topic name, not a suffix.
   * @param payload Message body, copied before returning.
   * @param retain Ask the broker to retain the message, for state a late
   *        subscriber must still see, such as availability.
   * @return false when disconnected or the publish is rejected. Published at
   *         QoS 0, so true means handed to the client, not acknowledged.
   */
  virtual bool publish(const std::string &topic, const std::string &payload, bool retain) = 0;
  /**
   * Publish under this device's protocol topic prefix.
   *
   * The prefix comes from the `EspectreDeviceConfig` passed to `setup()`, so
   * callers pass only the trailing segment, for example `"/telemetry"`.
   */
  virtual bool publish_suffix(const char *suffix, const std::string &payload, bool retain) = 0;
  /**
   * Register a topic and its handler.
   *
   * Idempotent per topic: subscribing again replaces the handler. May be
   * called before the connection is up; the subscription is issued on connect.
   *
   * @return false for an empty topic or an empty callback.
   */
  virtual bool subscribe(const std::string &topic, MessageCallback callback) = 0;
  /** Handler for the device command topic, which the transport subscribes itself. */
  virtual void set_command_callback(CommandCallback callback) = 0;
  /** Handler for connection state changes, including reconnects. */
  virtual void set_connection_callback(ConnectionCallback callback) = 0;
};

}  // namespace espectre
