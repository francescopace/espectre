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
#include <array>
#include <cstdint>
#include <string>

#include "ble_bindings_nimble.h"
#include "csi_capture_service.h"
#include "csi_udp_sender.h"
#include "deferred_loop_action.h"
#include "espectre_protocol.h"
#include "mqtt_transport.h"
#include "ota_service.h"
#include "standalone_wifi_manager.h"
#include "stimulus_service.h"
#include "wifi_provisioning_service.h"

namespace esphome {
namespace espectre {

class StreamFrontend {
 public:
  static constexpr size_t CSI_PAYLOAD_PREFIX_BYTES = 96U;
  static constexpr uint8_t CSI_DEFERRED_QUEUE_SLOTS = 16U;
  static constexpr uint8_t STIMULUS_DEDUP_WINDOW = 8U;

  enum class WorkflowState : uint8_t {
    WAIT_WIFI = 0,
    WIFI_READY,
    CSI_READY,
    STREAMING,
    OTA_IN_PROGRESS,
  };

  StreamFrontend() = default;
  StreamFrontend(IMqttTransport *mqtt_transport, IOtaService *ota_service);
  bool setup();
  void loop();
  void shutdown();
  ~StreamFrontend();

 private:
  struct DeferredCsiPacket final {
    wifi_pkt_rx_ctrl_t rx_ctrl{};
    uint32_t enqueued_at_ms{0U};
    std::array<uint8_t, 6> mac{};
    std::array<uint8_t, 6> dmac{};
    std::array<int8_t, HT20_CSI_LEN> normalized_csi{};
    std::array<uint8_t, CSI_PAYLOAD_PREFIX_BYTES> payload_prefix{};
    uint16_t normalized_len{0U};
    uint16_t payload_len{0U};
    uint16_t captured_payload_len{0U};
    uint16_t rx_seq{0U};
    bool first_word_invalid{false};
    bool payload_present{false};
  };

  struct RecentWifiRxFrame final {
    std::array<uint8_t, 6> src_mac{};
    uint16_t rx_seq{0U};
    bool valid{false};
  };

  bool init_nvs_();
  bool init_wifi_station_();
  bool setup_deferred_csi_queue_();
  void shutdown_deferred_csi_queue_();
  bool setup_ble_provisioning_();
  void setup_mqtt_();
  bool start_capture_();
  void stop_capture_();
  bool enqueue_deferred_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized);
  void process_deferred_csi_packets_();
  void process_deferred_csi_packet_(const DeferredCsiPacket &packet);
  void reset_collector_endpoint_();
  void prepare_for_ota_();
  void on_wifi_connected_();
  void on_wifi_disconnected_();
  void handle_ble_control_(const std::string &command);
  void handle_mqtt_command_(const std::string &payload);
  void publish_ble_sysinfo_();
  void publish_ble_line_(const char *line);
  void publish_mqtt_info_();
  void publish_mqtt_status_(bool online);
  void publish_mqtt_stats_();
  void publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message);
  void publish_mqtt_ota_status_(const EspectreOtaStatus &status);
  void handle_csi_packet_(const wifi_csi_info_t *info, const NormalizedCSIPayload &normalized);
  bool wifi_frame_recently_seen_(const std::array<uint8_t, 6> &src_mac, uint16_t rx_seq);
  bool stimulus_recently_seen_(uint32_t stimulus_id);
  void update_streaming_ble_policy_(float stimulus_pps, uint64_t dt_ms);
  void suspend_ble_for_streaming_();
  void resume_ble_after_streaming_();
  void transition_to_(WorkflowState next, const char *reason);
  void log_runtime_telemetry_();
  void reset_runtime_telemetry_baseline_();

  CsiCaptureService capture_service_;
  StimulusService stimulus_service_;
  CsiUdpSender udp_sender_;
  StandaloneWifiManager wifi_manager_;
  WifiProvisioningService wifi_provisioning_{&wifi_manager_};
  NimbleBleBindings ble_bindings_;
  IMqttTransport *mqtt_transport_{nullptr};
  IOtaService *ota_service_{nullptr};
  EspectreDeviceConfig device_config_{};
  EspectreDeviceInfo device_info_{};
  bool setup_complete_{false};
  bool ble_ready_{false};
  bool ble_client_connected_{false};
  DeferredLoopAction ble_sysinfo_refresh_;
  std::atomic<bool> wifi_connected_{false};
  std::atomic<WorkflowState> state_{WorkflowState::WAIT_WIFI};
  uint32_t stream_seq_{0U};
  uint32_t last_csi_ms_{0U};
  uint8_t last_csi_channel_{0U};
  DeferredCsiPacket *deferred_csi_slots_{nullptr};
  QueueHandle_t deferred_csi_free_slots_{nullptr};
  QueueHandle_t deferred_csi_ready_slots_{nullptr};
  uint64_t csi_rx_total_{0U};
  uint64_t csi_callback_total_{0U};
  uint64_t csi_nonempty_total_{0U};
  uint64_t csi_payload_present_total_{0U};
  uint64_t csi_deferred_drop_total_{0U};
  uint64_t wifi_retry_marked_total_{0U};
  uint64_t wifi_seq_dup_drop_total_{0U};
  uint64_t stimulus_dup_drop_total_{0U};
  std::array<RecentWifiRxFrame, STIMULUS_DEDUP_WINDOW> recent_wifi_frames_{};
  uint8_t recent_wifi_frame_idx_{0U};
  std::array<uint32_t, STIMULUS_DEDUP_WINDOW> recent_stimulus_ids_{};
  uint8_t recent_stimulus_idx_{0U};
  uint64_t stimulus_valid_total_{0U};
  uint64_t reference_frame_total_{0U};
  uint64_t stimulus_parse_fail_total_{0U};
  uint64_t filtered_total_{0U};
  uint64_t last_log_ms_{0U};
  uint32_t collector_ip_addr_{0U};
  uint32_t local_ip_addr_{0U};
  std::array<uint8_t, 6> local_mac_addr_{};
  uint16_t last_csi_len_{0U};
  uint16_t last_csi_payload_len_{0U};
  uint64_t prev_csi_callback_total_{0U};
  uint64_t prev_stimulus_valid_total_{0U};
  uint64_t prev_traffic_raw_total_{0U};
  uint64_t prev_traffic_rx_total_{0U};
  uint64_t prev_csi_deferred_drop_total_{0U};
  uint64_t prev_wifi_retry_marked_total_{0U};
  uint64_t prev_wifi_seq_dup_drop_total_{0U};
  uint64_t prev_stimulus_dup_drop_total_{0U};
  uint64_t prev_queued_total_{0U};
  uint64_t prev_tx_total_{0U};
  uint64_t prev_drop_total_{0U};
  uint64_t prev_fail_total_{0U};
  uint64_t prev_parse_fail_total_{0U};
  uint64_t prev_log_sample_ms_{0U};
  uint32_t deferred_max_age_ms_since_log_{0U};
  uint64_t ble_high_stimulus_ms_{0U};
  uint64_t ble_idle_stimulus_ms_{0U};
  bool ble_suspended_for_streaming_{false};
  bool stream_active_last_tick_{true};
  float last_loop_time_ms_{0.0F};
};

}  // namespace espectre
}  // namespace esphome
