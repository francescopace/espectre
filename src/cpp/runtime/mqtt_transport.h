/*
 * ESPectre - MQTT Transport Boundary
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <functional>
#include <string>

#include "espectre_protocol.h"

namespace esphome {
namespace espectre {

class IMqttTransport {
 public:
  using CommandCallback = std::function<void(const std::string &)>;
  using ConnectionCallback = std::function<void(bool)>;

  virtual ~IMqttTransport() = default;

  virtual bool setup(const EspectreDeviceConfig &config) = 0;
  virtual void loop() = 0;
  virtual void shutdown() = 0;
  virtual bool connected() const = 0;
  virtual bool publish(const std::string &topic, const std::string &payload, bool retain) = 0;
  virtual void set_command_callback(CommandCallback callback) = 0;
  virtual void set_connection_callback(ConnectionCallback callback) = 0;
};

}  // namespace espectre
}  // namespace esphome
