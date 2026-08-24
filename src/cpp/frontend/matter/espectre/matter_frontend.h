/*
 * ESPectre - Matter Frontend Adapter
 *
 * Bridges runtime events to the standard Matter occupancy surface.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <memory>

#include "direct_websocket_service.h"
#include "matter_bindings.h"
#include "runtime_events.h"
#include "runtime_direct_websocket_bridge.h"
#include "runtime_frontend_controller.h"

namespace espectre {

class MatterFrontend : public IRuntimeListener {
 public:
  MatterFrontend(IMatterBindings *bindings,
                 uint16_t endpoint_id,
                 IDirectWebSocketService *direct_service = nullptr);

  void set_runtime_config(const RuntimeConfig &config);
  void set_runtime_services_armed(bool armed);
  const RuntimeConfig &runtime_config() const { return runtime_.config(); }
  bool runtime_services_armed() const { return runtime_.services_armed(); }

  bool setup();
  void loop();
  void shutdown();
  ~MatterFrontend();

  const RuntimeSnapshot &snapshot() const { return runtime_.snapshot(); }
  const RuntimeCapabilities &capabilities() const { return runtime_.capabilities(); }
  bool is_setup_complete() const { return runtime_.is_setup_complete(); }

 protected:
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) override;
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override;
  void on_detector_changed(const RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override;
  void on_live_telemetry(float movement, float threshold) override;
  void on_runtime_fault(const char *message) override;

 private:
  IMatterBindings *bindings_;
  uint16_t endpoint_id_;
  RuntimeFrontendController runtime_;
  IDirectWebSocketService *direct_service_{nullptr};
  RuntimeDirectWebSocketBridge direct_bridge_;
};

}  // namespace espectre
