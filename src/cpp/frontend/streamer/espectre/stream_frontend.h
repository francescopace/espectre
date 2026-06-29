/*
 * ESPectre - Streamer Frontend
 *
 * Standalone frontend for raw CSI UDP streaming.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <atomic>
#include <cstdint>

#include "csi_capture_service.h"
#include "csi_udp_sender.h"
#include "standalone_wifi_manager.h"
#include "stimulus_service.h"

namespace esphome {
namespace espectre {

class StreamFrontend {
 public:
  enum class WorkflowState : uint8_t {
    WAIT_WIFI = 0,
    WIFI_READY,
    CSI_READY,
    GAIN_LOCK,
    STREAMING,
  };

  bool setup();
  void loop();
  void shutdown();
  ~StreamFrontend();

 private:
  bool init_nvs_();
  bool init_wifi_station_();
  bool start_capture_();
  void stop_capture_();
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  void handle_gain_lock_packet_(const wifi_csi_info_t *info);
  void handle_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized);
  void transition_to_(WorkflowState next, const char *reason);
  void log_runtime_telemetry_();
  void reset_runtime_telemetry_baseline_();

  CsiCaptureService capture_service_;
  StimulusService stimulus_service_;
  CsiUdpSender udp_sender_;
  StandaloneWifiManager wifi_manager_;
  bool setup_complete_{false};
  std::atomic<bool> wifi_connected_{false};
  std::atomic<bool> gain_lock_complete_{false};
  std::atomic<WorkflowState> state_{WorkflowState::WAIT_WIFI};
  uint64_t device_id_{0U};
  uint32_t stream_seq_{0U};
  uint32_t last_csi_ms_{0U};
  uint8_t last_csi_channel_{0U};
  uint64_t csi_rx_total_{0U};
  uint64_t csi_callback_total_{0U};
  uint64_t csi_nonempty_total_{0U};
  uint64_t csi_payload_present_total_{0U};
  uint64_t stimulus_valid_total_{0U};
  uint64_t reference_frame_total_{0U};
  uint64_t stimulus_parse_fail_total_{0U};
  uint64_t filtered_total_{0U};
  uint64_t last_log_ms_{0U};
  uint32_t collector_ip_addr_{0U};
  uint16_t last_csi_len_{0U};
  uint16_t last_csi_payload_len_{0U};
  uint64_t prev_csi_callback_total_{0U};
  uint64_t prev_stimulus_valid_total_{0U};
  uint64_t prev_traffic_rx_total_{0U};
  uint64_t prev_tx_total_{0U};
  uint64_t prev_drop_total_{0U};
  uint64_t prev_fail_total_{0U};
  uint64_t prev_parse_fail_total_{0U};
  uint64_t prev_log_sample_ms_{0U};
  bool stream_active_last_tick_{true};
};

}  // namespace espectre
}  // namespace esphome
