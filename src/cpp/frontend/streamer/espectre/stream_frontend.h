/*
 * ESPectre - Streamer Frontend
 *
 * Standalone frontend for raw CSI UDP streaming with optional FTM telemetry.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <atomic>
#include <cstdint>

#include "csi_udp_sender.h"
#include "ftm_manager.h"
#include "gain_controller.h"
#include "traffic_generator_manager.h"
#include "udp_listener.h"
#include "wifi_csi_interface.h"
#include "wifi_lifecycle.h"

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
  static void csi_rx_callback_wrapper_(void *ctx, wifi_csi_info_t *info);
  static void wifi_event_handler_(void *arg, esp_event_base_t event_base, int32_t event_id, void *event_data);

  bool init_nvs_();
  bool init_wifi_station_();
  bool start_csi_();
  void stop_csi_();
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  void handle_csi_packet_(wifi_csi_info_t *info);
  void transition_to_(WorkflowState next, const char *reason);
  void maybe_run_ftm_manager_();
  void log_runtime_telemetry_();

  WiFiLifecycleManager wifi_lifecycle_;
  WiFiCSIReal wifi_csi_;
  GainController gain_controller_;
  TrafficGeneratorManager traffic_generator_;
  UDPListener udp_listener_;
  CsiUdpSender udp_sender_;
  FtmManager ftm_manager_;
  esp_event_handler_instance_t wifi_event_instance_{nullptr};
  bool setup_complete_{false};
  std::atomic<bool> wifi_connected_{false};
  std::atomic<bool> csi_enabled_{false};
  std::atomic<bool> gain_lock_complete_{false};
  std::atomic<WorkflowState> state_{WorkflowState::WAIT_WIFI};
  uint64_t device_id_{0U};
  uint32_t boot_id_{0U};
  uint32_t stream_seq_{0U};
  uint32_t last_csi_ms_{0U};
  uint8_t last_csi_channel_{0U};
  bool collapse_logged_{false};
  bool remap_logged_{false};
  uint64_t csi_rx_total_{0U};
  uint64_t stimulus_valid_total_{0U};
  uint64_t reference_frame_total_{0U};
  uint64_t filtered_total_{0U};
  uint64_t traffic_rx_total_last_{0U};
  uint64_t last_log_ms_{0U};
  int wifi_retry_count_{0};
};

}  // namespace espectre
}  // namespace esphome
