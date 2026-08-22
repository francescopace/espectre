/*
 * ESPectre - Native Frontend Adapter
 *
 * Bridges runtime events and control flows to BLE, MQTT, and OTA services.
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
#include "ble_bindings.h"
#include "deferred_loop_action.h"
#include "frontend_ha_mqtt_helpers.h"
#include "mqtt_transport.h"
#include "ota_service.h"
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
  };

  using ProvisioningCommandCallback = std::function<bool(const std::string &command, std::string *message)>;
  using DeviceConfigChangeCallback = std::function<bool(const EspectreDeviceConfig &config, bool clear, std::string *message)>;

  explicit NativeFrontend(IBleBindings *bindings);
  NativeFrontend(IBleBindings *bindings, IMqttTransport *mqtt_transport, IOtaService *ota_service = nullptr);

  void set_runtime_config(const RuntimeConfig &config);
  void set_device_config(const EspectreDeviceConfig &config);
  void set_device_info(const EspectreDeviceInfo &info);
  void set_wifi_provisioning_info(const WifiProvisioningInfo &info);
  void set_provisioning_command_callback(ProvisioningCommandCallback callback);
  void set_device_config_change_callback(DeviceConfigChangeCallback callback);
  const EspectreDeviceConfig &device_config() const { return device_config_; }
  const RuntimeConfig &runtime_config() const { return runtime_.config(); }

  bool setup();
  void loop();
  void shutdown();
  ~NativeFrontend();

  const RuntimeSnapshot &snapshot() const { return runtime_.snapshot(); }
  const RuntimeCapabilities &capabilities() const { return runtime_.capabilities(); }
  bool is_setup_complete() const { return runtime_.is_setup_complete(); }
  bool client_connected() const { return client_connected_; }
  bool ble_active() const { return ble_active_; }
  void request_ble_recovery();

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
  bool handle_control_command_(const std::string &command);
  void handle_mqtt_command_(const std::string &payload);
  bool handle_threshold_write_(float threshold);
  bool handle_motion_hits_write_(uint8_t motion_on_hits, uint8_t motion_off_hits);
  bool handle_csi_traffic_mode_write_(CsiTrafficMode mode);
  bool handle_traffic_generator_mode_write_(RuntimeTrafficMode mode);
  bool handle_detector_write_(DetectionAlgorithm algorithm);
  bool handle_recalibration_write_();
  bool handle_ble_mode_write_(bool enable, std::string *message);
  bool ble_should_run_() const;
  bool wifi_configured_() const;
  bool provisioning_complete_() const;
  bool start_ble_();
  void stop_ble_();
  void refresh_ble_policy_();
  void apply_pending_ble_intent_();
  void handle_ha_birth_message_(const std::string &topic, const std::string &payload);
  void handle_ha_threshold_command_(const std::string &payload);
  void handle_ha_motion_hits_command_(bool motion_on, const std::string &payload);
  void handle_ha_calibrate_command_(const std::string &payload);
  void handle_ha_csi_traffic_mode_command_(const std::string &payload);
  void handle_ha_traffic_generator_mode_command_(const std::string &payload);
  void handle_ha_diagnostics_command_(const std::string &payload);
  void handle_connection_state_(bool connected);
  void handle_live_telemetry_subscription_(bool subscribed);
  void update_live_telemetry_enabled_();
  void setup_mqtt_();
  void setup_ha_mqtt_();
  void publish_ha_discovery_();
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
  void publish_mqtt_ota_status_(const EspectreOtaStatus &status);
  void publish_current_mqtt_ota_status_();
  void publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message);
  void prepare_for_ota_();
  void resume_after_ota_error_();
  void sample_diagnostics_(uint32_t now_ms);
  void send_system_info_();
  uint32_t now_ms_() const;

  IBleBindings *bindings_;
  IMqttTransport *mqtt_transport_{nullptr};
  IOtaService *ota_service_{nullptr};
  ProvisioningCommandCallback provisioning_command_callback_{};
  DeviceConfigChangeCallback device_config_change_callback_{};
  RuntimeFrontendController runtime_;
  EspectreDeviceConfig device_config_{};
  EspectreDeviceInfo device_info_{};
  FrontendHaMqttSettings ha_settings_{};
  WifiProvisioningInfo wifi_info_{};
  DeferredLoopAction system_info_refresh_;
  RuntimeDiagnosticsSampler diagnostics_sampler_;
  RuntimeDiagnosticsSample latest_diagnostics_{};
  bool client_connected_{false};
  bool mqtt_connected_{false};
  bool mqtt_ha_online_{false};
  bool ble_active_{false};
  bool ble_forced_{false};
  bool ota_frontend_quiesced_{false};
  enum class BleIntent : uint8_t { Unchanged = 0, Start, Stop };
  BleIntent pending_ble_intent_{BleIntent::Unchanged};
  float last_loop_time_ms_{0.0f};
};

}  // namespace espectre
