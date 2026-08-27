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

#include "direct_http_service.h"
#include "matter_bindings.h"
#include "peer_discovery_service_esp_idf.h"
#include "runtime_events.h"
#include "runtime_direct_http_bridge.h"
#include "runtime_diagnostics.h"
#include "runtime_event_mailbox.h"
#include "runtime_frontend_controller.h"

namespace espectre {

class MatterFrontend : public IRuntimeListener {
 public:
  MatterFrontend(IMatterBindings *bindings,
                 uint16_t endpoint_id,
                 IDirectHttpService *direct_service = nullptr);

  void set_runtime_config(const RuntimeConfig &config);
  bool set_runtime_services_armed(bool armed);
  const RuntimeConfig &runtime_config() const { return runtime_.config(); }
  bool runtime_services_armed() const { return runtime_.services_armed(); }

  bool setup();
  void loop();
  void shutdown();
  ~MatterFrontend();

  const RuntimeSnapshot &snapshot() const { return runtime_.snapshot(); }
  const RuntimeCapabilities &capabilities() const { return runtime_.capabilities(); }
  bool is_setup_complete() const { return runtime_.is_setup_complete(); }
  void sync_device_label();

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
  bool start_direct_service_();
  void stop_direct_service_();
  void drain_pending_runtime_events_();
  void update_live_telemetry_enabled_();

  IMatterBindings *bindings_;
  uint16_t endpoint_id_;
  RuntimeFrontendController runtime_;
  IDirectHttpService *direct_service_{nullptr};
  RuntimeDirectHttpBridge direct_bridge_;
  EspIdfPeerDiscoveryService peer_discovery_;
  RuntimeDiagnosticsSampler diagnostics_sampler_;
  RuntimeDiagnosticsSample latest_diagnostics_{};
  RuntimeEventMailbox runtime_events_{};
  bool live_telemetry_enabled_{true};
  std::string fallback_device_label_{};
};

}  // namespace espectre
