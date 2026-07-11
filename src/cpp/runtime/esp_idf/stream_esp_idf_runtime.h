#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#include "csi_capture_service.h"
#include "csi_stream_transport.h"
#include "csi_traffic_service.h"
#include "runtime_interface.h"
#include "standalone_wifi_service.h"

namespace esphome {
namespace espectre {

class StreamEspIdfRuntime : public IEspectreRuntime {
 public:
  explicit StreamEspIdfRuntime(const RuntimeConfig &config);

  bool setup() override;
  void shutdown() override;
  void loop() override;
  void set_services_armed(bool armed) override;
  void set_live_telemetry_enabled(bool enabled) override;

  bool set_threshold_runtime(float threshold) override;
  bool trigger_recalibration() override;
  bool is_calibrating() const override;

  RuntimeSnapshot get_snapshot() const override;
  RuntimeCapabilities get_capabilities() const override;

  void set_listener(IRuntimeListener *listener) override;

 private:
  enum class WorkflowState : uint8_t {
    WAIT_WIFI = 0,
    WIFI_READY,
    CSI_READY,
    STREAMING,
  };

  bool init_nvs_();
  bool init_wifi_station_();
  bool start_capture_();
  void stop_capture_();
  const char *workflow_state_name_(WorkflowState state) const;
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  void transition_to_(WorkflowState next, const char *reason);
  void notify_fault_(const char *message);
  void handle_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized);

  RuntimeConfig config_{};
  RuntimeSnapshot snapshot_{};
  RuntimeCapabilities capabilities_{};
  IRuntimeListener *listener_{nullptr};

  CsiCaptureService capture_service_;
  CsiTrafficService csi_traffic_service_;
  CsiStreamTransport stream_transport_;
  StandaloneWifiService wifi_manager_;

  bool setup_complete_{false};
  bool services_armed_{true};
  bool live_telemetry_enabled_{true};
  std::atomic<bool> wifi_connected_{false};
  std::atomic<WorkflowState> state_{WorkflowState::WAIT_WIFI};
  std::string last_fault_;
  std::array<uint8_t, 6> ap_bssid_{};
};

}  // namespace espectre
}  // namespace esphome
