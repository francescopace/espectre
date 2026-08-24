/*
 * ESPectre - Native Frontend Adapter
 *
 * Bridges runtime events and control flows to Direct WebSocket, MQTT, and OTA
 * services.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>
#include "direct_websocket_service.h"
#include "frontend_control_helpers.h"
#include "frontend_ha_mqtt_helpers.h"
#include "mqtt_transport.h"
#include "ota_service.h"
#include "peer_discovery.h"
#include "runtime_diagnostics.h"
#include "runtime_events.h"
#include "runtime_frontend_controller.h"

namespace espectre {

class NativeFrontend : public IRuntimeListener {
 public:
  struct WifiProvisioningInfo {
    std::string ssid;
    std::string bssid;
    uint8_t channel{0U};
    bool has_saved_config{false};
    WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
    std::string apply_state{"idle"};
    std::string apply_message;
  };

  using ProvisioningCommandCallback = std::function<bool(const std::string &command, std::string *message)>;
  using DeviceConfigChangeCallback = std::function<bool(const EspectreDeviceConfig &config, bool clear, std::string *message)>;

  explicit NativeFrontend(IMqttTransport *mqtt_transport = nullptr,
                          IOtaService *ota_service = nullptr,
                          IDirectWebSocketService *direct_service = nullptr);

  void set_runtime_config(const RuntimeConfig &config);
  void set_device_config(const EspectreDeviceConfig &config);
  void set_device_info(const EspectreDeviceInfo &info);
  void set_peer_discovery_service(IPeerDiscoveryService *service);
  void set_wifi_provisioning_info(const WifiProvisioningInfo &info);
  void set_provisioning_command_callback(ProvisioningCommandCallback callback);
  void set_device_config_change_callback(DeviceConfigChangeCallback callback);
  void prepare_for_wifi_reconfigure();
  void resume_after_wifi_reconfigure();
  const EspectreDeviceConfig &device_config() const { return device_config_; }
  const RuntimeConfig &runtime_config() const { return runtime_.config(); }

  bool setup();
  void loop();
  void shutdown();
  ~NativeFrontend();

  const RuntimeSnapshot &snapshot() const { return runtime_.snapshot(); }
  const RuntimeCapabilities &capabilities() const { return runtime_.capabilities(); }
  bool is_setup_complete() const { return runtime_.is_setup_complete(); }
  size_t direct_client_count() const { return direct_client_count_; }

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
  void handle_mqtt_command_(const std::string &payload);
  FrontendCommandResult dispatch_command_(const EspectreCommand &command, bool allow_local_config);
  std::string handle_direct_request_(const DirectWebSocketRequest &request);
  IDirectWebSocketService::DeferredRequestResult handle_deferred_direct_request_(
      uint64_t connection_token,
      const DirectWebSocketRequest &request);
  std::string direct_capabilities_payload_() const;
  std::string direct_status_payload_() const;
  std::string direct_config_payload_() const;
  std::string direct_diagnostics_payload_() const;
  bool handle_threshold_write_(float threshold);
  bool handle_motion_hits_write_(uint8_t motion_on_hits, uint8_t motion_off_hits);
  bool handle_csi_traffic_mode_write_(CsiTrafficMode mode);
  bool handle_traffic_generator_mode_write_(RuntimeTrafficMode mode);
  bool handle_detector_write_(DetectionAlgorithm algorithm);
  bool handle_recalibration_write_();
  bool wifi_configured_() const;
  void handle_ha_birth_message_(const std::string &topic, const std::string &payload);
  void handle_ha_threshold_command_(const std::string &payload);
  void handle_ha_motion_hits_command_(bool motion_on, const std::string &payload);
  void handle_ha_calibrate_command_(const std::string &payload);
  void handle_ha_csi_traffic_mode_command_(const std::string &payload);
  void handle_ha_traffic_generator_mode_command_(const std::string &payload);
  void handle_ha_diagnostics_command_(const std::string &payload);
  void drain_pending_runtime_events_();
  void update_live_telemetry_enabled_();
  void refresh_direct_service_();
  void stop_direct_service_();
  void publish_direct_event_(const char *event_name,
                             const std::string &data_json,
                             bool replaceable_telemetry = false);
  void setup_mqtt_();
  void setup_ha_mqtt_();
  void publish_ha_discovery_();
  void drain_pending_ha_snapshot_();
  bool ha_mqtt_ready_();
  void publish_ha_motion_(MotionState state);
  void publish_ha_movement_(float movement);
  void publish_ha_threshold_(float threshold);
  void publish_ha_motion_hits_(uint8_t motion_on_hits, uint8_t motion_off_hits);
  void publish_ha_calibrate_(bool calibrating);
  void publish_ha_detector_(const char *detector_name);
  void publish_ha_traffic_control_(CsiTrafficMode csi_traffic_mode, RuntimeTrafficMode traffic_generator_mode);
  void publish_ha_diagnostics_();
  void publish_ha_state_(const RuntimeSnapshot &snapshot);
  void publish_current_ha_state_();
  void publish_mqtt_info_();
  void publish_mqtt_commands_();
  EspectreDeviceInfo mqtt_protocol_device_info_() const;
  void publish_mqtt_status_(bool online);
  void publish_mqtt_telemetry_(const RuntimeSnapshot &snapshot, uint32_t now_ms);
  void publish_mqtt_stats_();
  EspectreOtaStatus current_ota_status_() const;
  void publish_ota_status_(const EspectreOtaStatus &status);
  void publish_mqtt_ota_status_(const EspectreOtaStatus &status);
  void publish_current_mqtt_ota_status_();
  void publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message);
  void prepare_for_ota_();
  void resume_after_ota_error_();
  void sample_diagnostics_(uint32_t now_ms);
  uint32_t now_ms_() const;

  IMqttTransport *mqtt_transport_{nullptr};
  IOtaService *ota_service_{nullptr};
  IDirectWebSocketService *direct_service_{nullptr};
  IPeerDiscoveryService *peer_discovery_{nullptr};
  ProvisioningCommandCallback provisioning_command_callback_{};
  DeviceConfigChangeCallback device_config_change_callback_{};
  RuntimeFrontendController runtime_;
  EspectreDeviceConfig device_config_{};
  EspectreDeviceInfo device_info_{};
  FrontendHaMqttSettings ha_settings_{};
  std::vector<FrontendHaDiscoveryMessage> pending_ha_discovery_{};
  size_t pending_ha_discovery_index_{0U};
  WifiProvisioningInfo wifi_info_{};
  RuntimeDiagnosticsSampler diagnostics_sampler_;
  RuntimeDiagnosticsSample latest_diagnostics_{};
  RuntimeSnapshot pending_live_telemetry_{};
  RuntimeSnapshot pending_motion_state_{};
  bool live_telemetry_pending_{false};
  bool motion_state_pending_{false};
  bool mqtt_connected_{false};
  bool mqtt_ha_online_{false};
  bool pending_ha_state_{false};
  size_t direct_client_count_{0U};
  bool ota_frontend_quiesced_{false};
  bool wifi_reconfigure_quiesced_{false};
  bool peer_discovery_enabled_{false};
  float last_loop_time_ms_{0.0f};
};

}  // namespace espectre
