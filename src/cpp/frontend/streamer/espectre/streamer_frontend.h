/*
 * ESPectre - Streamer Frontend
 *
 * Standalone frontend for raw CSI UDP streaming.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "direct_http_service.h"
#include "runtime_events.h"
#include "runtime_direct_http_bridge.h"
#include "runtime_frontend_controller.h"

namespace espectre {

class StreamerFrontend : public IRuntimeListener {
 public:
  explicit StreamerFrontend(IDirectHttpService *direct_service);
  bool setup();
  void loop();
  void shutdown();
  ~StreamerFrontend();

 private:
  RuntimeConfig build_runtime_config_() const;
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) override;
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override;
  void on_live_telemetry(float movement, float threshold) override;
  void on_runtime_fault(const char *message) override;

  RuntimeFrontendController runtime_;
  IDirectHttpService *direct_service_{nullptr};
  RuntimeDirectHttpBridge direct_bridge_;
  bool setup_complete_{false};
};

}  // namespace espectre
