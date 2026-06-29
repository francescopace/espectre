/*
 * ESPectre - BLE Frontend Adapter
 *
 * Thin frontend that maps IEspectreRuntime events to the custom BLE protocol.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <memory>
#include <string>

#include "ble_bindings.h"
#include "runtime_events.h"
#include "runtime_frontend_controller.h"

namespace esphome {
namespace espectre {

class BleFrontend : public IRuntimeListener {
 public:
  explicit BleFrontend(IBleBindings *bindings);

  void set_runtime_config(const RuntimeConfig &config);
  const RuntimeConfig &runtime_config() const { return runtime_.config(); }

  bool setup();
  void loop();
  void shutdown();
  ~BleFrontend();

  const RuntimeSnapshot &snapshot() const { return runtime_.snapshot(); }
  const RuntimeCapabilities &capabilities() const { return runtime_.capabilities(); }
  bool is_setup_complete() const { return runtime_.is_setup_complete(); }
  bool client_connected() const { return client_connected_; }

 protected:
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) override;
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override;
  void on_live_telemetry(float movement, float threshold) override;
  void on_runtime_fault(const char *message) override;

 private:
  bool handle_control_command_(const std::string &command);
  bool handle_threshold_write_(float threshold);
  void handle_connection_state_(bool connected);
  void send_system_info_();
  uint32_t now_ms_() const;

  IBleBindings *bindings_;
  RuntimeFrontendController runtime_;
  bool client_connected_{false};
  uint32_t telemetry_interval_ms_{40};
  uint32_t last_telemetry_ms_{0};
};

}  // namespace espectre
}  // namespace esphome
