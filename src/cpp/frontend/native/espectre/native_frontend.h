/*
 * ESPectre - Native Frontend Adapter
 *
 * Thin frontend that maps IEspectreRuntime events to the custom BLE protocol.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstddef>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "ble_bindings.h"
#include "mqtt_transport.h"
#include "ota_service.h"
#include "periodic_sensing_status_logger.h"
#include "runtime_events.h"
#include "runtime_frontend_controller.h"

namespace esphome {
namespace espectre {

class NativeFrontend : public IRuntimeListener {
 public:
  struct WifiProvisioningInfo {
    std::string ssid;
    std::string bssid;
    uint8_t channel{0U};
    bool has_saved_config{false};
    bool password_set{false};
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
  bool mqtt_enabled() const { return device_config_.mqtt_enabled; }

 protected:
  void on_motion_state_changed(const RuntimeSnapshot &snapshot) override;
  void on_periodic_update(const RuntimeSnapshot &snapshot, uint32_t packets_received) override;
  void on_threshold_changed(const RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const RuntimeSnapshot &snapshot, bool success) override;
  void on_live_telemetry(float movement, float threshold) override;
  void on_runtime_fault(const char *message) override;

 private:
  bool handle_control_command_(const std::string &command);
  void handle_mqtt_command_(const std::string &payload);
  bool handle_threshold_write_(float threshold);
  void handle_connection_state_(bool connected);
  void handle_live_telemetry_subscription_(bool subscribed);
  void setup_mqtt_();
  bool handle_ota_command_(const EspectreCommand &command);
  void publish_mqtt_info_();
  void publish_mqtt_status_(bool online);
  void publish_mqtt_telemetry_(const RuntimeSnapshot &snapshot, uint32_t now_ms);
  void publish_mqtt_stats_();
  void publish_mqtt_ota_status_(const EspectreOtaStatus &status);
  void publish_mqtt_command_result_(const EspectreCommand &command, bool accepted, const char *message);
  void send_system_info_();
  void queue_system_info_line_(const char *line);
  void flush_pending_system_info_(bool force = false);
  uint32_t now_ms_() const;

  IBleBindings *bindings_;
  IMqttTransport *mqtt_transport_{nullptr};
  IOtaService *ota_service_{nullptr};
  ProvisioningCommandCallback provisioning_command_callback_{};
  DeviceConfigChangeCallback device_config_change_callback_{};
  RuntimeFrontendController runtime_;
  PeriodicSensingStatusLogger status_logger_{};
  EspectreDeviceConfig device_config_{};
  EspectreDeviceInfo device_info_{};
  WifiProvisioningInfo wifi_info_{};
  bool client_connected_{false};
  bool telemetry_subscribed_{false};
  float last_loop_time_ms_{0.0f};
  uint32_t sysinfo_line_interval_ms_{20};
  uint32_t last_sysinfo_line_ms_{0};
  std::vector<std::string> pending_sysinfo_lines_;
  size_t next_sysinfo_line_index_{0};
};

}  // namespace espectre
}  // namespace esphome
