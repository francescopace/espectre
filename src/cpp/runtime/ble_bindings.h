/*
 * ESPectre - BLE Bindings Interface
 *
 * Thin boundary between frontend adapters and the BLE transport stack.
 * Host-side tests provide a mock implementation.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace esphome {
namespace espectre {

class IBleBindings {
 public:
  using ConnectionStateCallback = std::function<void(bool connected)>;
  using ControlWriteCallback = std::function<void(const std::string &)>;
  using TelemetrySubscriptionCallback = std::function<void(bool subscribed)>;

  virtual ~IBleBindings() = default;

  virtual bool setup() = 0;
  virtual void loop() = 0;
  virtual void shutdown() = 0;

  virtual void set_connection_state_callback(ConnectionStateCallback callback) = 0;
  virtual void set_control_write_callback(ControlWriteCallback callback) = 0;
  virtual void set_telemetry_subscription_callback(TelemetrySubscriptionCallback callback) = 0;
  virtual void set_device_name(const char *name) = 0;

  virtual void publish_telemetry(const uint8_t *payload, size_t payload_len) = 0;
  virtual void replace_sysinfo_lines(std::vector<std::string> lines) = 0;
  virtual void publish_sysinfo_line(const char *line) = 0;
  virtual void report_fault(const char *message) = 0;
};

}  // namespace espectre
}  // namespace esphome
