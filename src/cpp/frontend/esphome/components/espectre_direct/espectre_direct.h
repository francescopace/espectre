/*
 * ESPectre - Direct API add-on for the espectre component built into ESPHome
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include "esphome/components/espectre/espectre.h"
#include "esphome/components/wifi/wifi_component.h"
#include "esphome/core/component.h"

#include "espectre_sdk.h"
#include "espectre_services_sdk.h"

#include <cstdint>
#include <string>

namespace esphome::espectre_direct {

/**
 * Serves the ESPectre Direct API on a runtime that the built-in component owns.
 *
 * It reads the runtime through the SDK's guest access and never writes to it.
 * Every sensing change goes to the built-in component's own request methods,
 * which apply it in that component's loop, republish its entities, and save
 * it. Raw CSI collection is the one runtime operation the SDK bridge drives.
 */
class ESPectreDirect final : public Component, public ::espectre::IRuntimeListener,
                             public wifi::WiFiConnectStateListener, public wifi::WiFiScanResultsListener {
 public:
  void setup() override;
  void loop() override;
  void dump_config() override;
  void on_shutdown() override;
  float get_setup_priority() const override { return setup_priority::AFTER_WIFI; }

  void set_parent(espectre::ESPectreComponent *parent) { this->parent_ = parent; }
  void set_port(uint16_t port) { this->port_ = port; }
  void set_raw_csi(bool enabled) { this->raw_csi_ = enabled; }

 protected:
  void on_wifi_connect_state(StringRef, std::span<const uint8_t, 6>) override {
    this->wifi_pin_.notify_station_changed();
  }
  void on_wifi_scan_results(const wifi::wifi_scan_vector_t<wifi::WiFiScanResult> &) override {
    this->scan_state_ = ScanState::COMPLETE;
  }
  void on_sensing_readiness_changed(const ::espectre::RuntimeSnapshot &snapshot) override;
  void on_motion_state_changed(const ::espectre::RuntimeSnapshot &snapshot) override;
  void on_threshold_changed(const ::espectre::RuntimeSnapshot &snapshot) override;
  void on_detector_changed(const ::espectre::RuntimeSnapshot &snapshot) override;
  void on_calibration_started(const ::espectre::RuntimeSnapshot &snapshot) override;
  void on_calibration_finished(const ::espectre::RuntimeSnapshot &snapshot, bool success) override;
  void on_live_telemetry(const ::espectre::RuntimeSnapshot &snapshot) override;
  void on_runtime_fault(const char *message) override;

  bool start_();
  bool connect_to_access_point_(const std::string &bssid, std::string *message, bool *transition_started);
  ::espectre::RuntimeControlSink control_sink_();
  void publish_sensing_();
  void update_discovery_();
  ::espectre::MdnsTxtRecords mdns_txt_records_() const;

  static constexpr uint32_t RETRY_INTERVAL_MS = 5000;
  static constexpr uint32_t ADDRESS_POLL_INTERVAL_MS = 1000;

  espectre::ESPectreComponent *parent_{nullptr};
  ::espectre::WifiBssidPinService wifi_pin_;
  ::espectre::EspIdfDirectHttpService service_;
  ::espectre::MdnsDiscoveryService mdns_discovery_;
  ::espectre::MdnsBootstrapResponder mdns_bootstrap_responder_;
  ::espectre::EspIdfPeerDiscoveryService peer_discovery_;
  // Declared after the service and discovery it borrows; its destructor shuts them down.
  ::espectre::RuntimeDirectHttpBridge bridge_;
  uint64_t device_id_{0};
  uint32_t next_start_ms_{0};
  uint32_t next_address_poll_ms_{0};
  uint32_t next_mdns_setup_ms_{0};
  uint32_t ipv4_{0};
  uint16_t port_{::espectre::ESPECTRE_DIRECT_HTTP_PORT};
  bool raw_csi_{true};
  bool wifi_pin_ready_{false};
  enum class ScanState { EMPTY, SCANNING, COMPLETE };
  ScanState scan_state_{ScanState::EMPTY};
  bool started_{false};
  bool observing_{false};
};

}  // namespace esphome::espectre_direct
