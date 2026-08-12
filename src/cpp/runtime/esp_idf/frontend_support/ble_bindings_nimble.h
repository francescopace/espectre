/*
 * ESPectre - NimBLE Bindings
 *
 * NimBLE-backed BLE bindings for telemetry, sysinfo, and runtime control.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <string>
#include <vector>

#include "host/ble_gap.h"
#include "host/ble_gatt.h"

#include "ble_bindings.h"

namespace espectre {

class NimbleBleBindings : public IBleBindings {
 public:
  bool setup() override;
  void loop() override;
  void shutdown() override;

  void set_connection_state_callback(ConnectionStateCallback callback) override;
  void set_control_write_callback(ControlWriteCallback callback) override;
  void set_telemetry_subscription_callback(TelemetrySubscriptionCallback callback) override;
  void set_device_name(const char *name) override;

  void publish_telemetry(const uint8_t *payload, size_t payload_len) override;
  void replace_sysinfo_lines(std::vector<std::string> lines) override;
  void publish_sysinfo_line(const char *line) override;
  void report_fault(const char *message) override;

 private:
  void flush_pending_sysinfo_(bool force = false);
  bool notify_sysinfo_line_(const std::string &line);
  bool start_advertising_();
  void on_sync_();
  void on_reset_(int reason);
  int on_gap_event_(ble_gap_event *event);
  int on_gatt_access_(uint16_t conn_handle, uint16_t attr_handle, ble_gatt_access_ctxt *ctxt);

  static void host_task_(void *param);
  static void on_sync_static_();
  static void on_reset_static_(int reason);
  static int gap_event_static_(ble_gap_event *event, void *arg);
  static int gatt_access_static_(uint16_t conn_handle, uint16_t attr_handle, ble_gatt_access_ctxt *ctxt, void *arg);

  static NimbleBleBindings *instance_;

  ConnectionStateCallback connection_state_callback_;
  ControlWriteCallback control_write_callback_;
  TelemetrySubscriptionCallback telemetry_subscription_callback_;
  std::vector<uint8_t> telemetry_value_;
  std::string sysinfo_value_;
  std::string device_name_;
  uint8_t addr_type_{0};
  uint16_t conn_handle_{0xFFFF};
  bool setup_complete_{false};
  bool shutting_down_{false};
  bool telemetry_subscribed_{false};
  bool advertising_active_{false};
  uint32_t sysinfo_line_interval_ms_{20U};
  uint32_t last_sysinfo_line_ms_{0U};
  std::vector<std::string> pending_sysinfo_lines_;
  size_t next_sysinfo_line_index_{0U};
};

}  // namespace espectre
