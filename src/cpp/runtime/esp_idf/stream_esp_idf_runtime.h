/*
 * ESPectre - Stream ESP-IDF Runtime
 *
 * ESP-IDF runtime variant for raw CSI collection and UDP streaming.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <string>

#include "csi_capture_service.h"
#include "csi_stream_transport.h"
#include "csi_traffic_service.h"
#include "esp_idf_runtime_base.h"
#include "runtime_interface.h"
#include "runtime_debug_telemetry.h"
#include "standalone_wifi_service.h"

namespace espectre {

class StreamEspIdfRuntime : public EspIdfRuntimeBase {
 public:
  explicit StreamEspIdfRuntime(const RuntimeConfig &config);

  bool setup() override;
  void shutdown() override;
  void loop() override;
  void set_services_armed(bool armed) override;
  void set_live_telemetry_enabled(bool enabled) override;

  bool set_threshold_runtime(float threshold) override;
  bool set_detection_algorithm_runtime(DetectionAlgorithm algorithm) override;
  bool trigger_recalibration() override;
  bool is_calibrating() const override;

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
  void handle_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized);
  void handle_pacing_packet_(const sockaddr_in &sender_addr, uint64_t pacing_total);
  static void capture_packet_callback_(void *context,
                                       const wifi_csi_info_t *info,
                                       const NormalizedCSIPayload &normalized);
  static void pacing_packet_callback_(void *context, const sockaddr_in &sender_addr, uint64_t pacing_total);

  CsiCaptureService capture_service_;
  CsiTrafficService csi_traffic_service_;
  CsiStreamTransport stream_transport_;
  StandaloneWifiService wifi_manager_;

  std::atomic<bool> wifi_connected_{false};
  std::atomic<WorkflowState> state_{WorkflowState::WAIT_WIFI};
  std::array<uint8_t, 6> ap_bssid_{};
};

}  // namespace espectre
