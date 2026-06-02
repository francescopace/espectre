/*
 * ESPectre - Matter Frontend Adapter
 *
 * Thin frontend that maps IEspectreRuntime events to Matter bindings.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <memory>

#include "matter_bindings.h"
#include "runtime_capabilities.h"
#include "runtime_events.h"
#include "runtime_interface.h"
#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

class MatterFrontend : public IRuntimeListener {
 public:
  MatterFrontend(IMatterBindings *bindings, uint16_t endpoint_id);

  void set_runtime_config(const RuntimeConfig &config);
  const RuntimeConfig &runtime_config() const { return runtime_config_; }

  bool setup();
  void loop();
  void shutdown();
  ~MatterFrontend();

  bool handle_threshold_write(float threshold);
  bool handle_recalibrate_request();

  const RuntimeSnapshot &snapshot() const { return runtime_snapshot_; }
  const RuntimeCapabilities &capabilities() const { return runtime_capabilities_; }
  bool is_setup_complete() const { return setup_complete_; }

 protected:
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) override;
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override;
  void on_live_telemetry(float movement, float threshold) override;
  void on_runtime_fault(const char *message) override;

 private:
  IMatterBindings *bindings_;
  uint16_t endpoint_id_;
  RuntimeConfig runtime_config_{};
  RuntimeSnapshot runtime_snapshot_{};
  RuntimeCapabilities runtime_capabilities_{};
  std::unique_ptr<IEspectreRuntime> runtime_;
  bool setup_complete_{false};
  bool threshold_republished_{false};
};

}  // namespace espectre
}  // namespace esphome
