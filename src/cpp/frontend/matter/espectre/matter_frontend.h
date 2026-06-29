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
#include "runtime_events.h"
#include "runtime_frontend_controller.h"

namespace esphome {
namespace espectre {

class MatterFrontend : public IRuntimeListener {
 public:
  MatterFrontend(IMatterBindings *bindings, uint16_t endpoint_id);

  void set_runtime_config(const RuntimeConfig &config);
  void set_runtime_services_armed(bool armed);
  const RuntimeConfig &runtime_config() const { return runtime_.config(); }
  bool runtime_services_armed() const { return runtime_.services_armed(); }

  bool setup();
  void loop();
  void shutdown();
  ~MatterFrontend();

  bool handle_threshold_write(float threshold);
  bool handle_recalibrate_request();

  const RuntimeSnapshot &snapshot() const { return runtime_.snapshot(); }
  const RuntimeCapabilities &capabilities() const { return runtime_.capabilities(); }
  bool is_setup_complete() const { return runtime_.is_setup_complete(); }

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
  RuntimeFrontendController runtime_;
  bool threshold_republished_{false};
};

}  // namespace espectre
}  // namespace esphome
