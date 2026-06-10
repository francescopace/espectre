/*
 * ESPectre - NimBLE Bindings
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <string>
#include <vector>

#include "host/ble_gap.h"
#include "host/ble_gatt.h"

#include "ble_bindings.h"

namespace esphome {
namespace espectre {

class NimbleBleBindings : public IBleBindings {
 public:
  bool setup() override;
  void shutdown() override;

  void set_connection_state_callback(ConnectionStateCallback callback) override;
  void set_control_write_callback(ControlWriteCallback callback) override;

  void publish_telemetry(const uint8_t *payload, size_t payload_len) override;
  void publish_sysinfo_line(const char *line) override;
  void report_fault(const char *message) override;

 private:
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
  std::vector<uint8_t> telemetry_value_;
  std::string sysinfo_value_;
  uint8_t addr_type_{0};
  uint16_t conn_handle_{0xFFFF};
  bool setup_complete_{false};
};

}  // namespace espectre
}  // namespace esphome
