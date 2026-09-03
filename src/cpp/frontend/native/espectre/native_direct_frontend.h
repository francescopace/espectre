/*
 * ESPectre - Native Direct Frontend Adapter
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "direct_http_service.h"
#include "frontend_command_engine.h"
#include "peer_discovery.h"
#include "raw_csi_session_controller.h"
#include "runtime_interface.h"

namespace espectre {

class NativeFrontend;

struct NativeWifiProvisioningInfo {
  struct AccessPoint {
    std::string bssid;
    int8_t rssi_dbm{0};
    uint8_t channel{0U};
  };

  std::string ssid;
  std::string bssid;
  uint8_t channel{0U};
  bool has_saved_config{false};
  WifiBandPolicy band_policy{WifiBandPolicy::BAND_2G};
  std::string apply_state{"idle"};
  std::string apply_message;
  bool scan_pending{false};
  std::string scan_message;
  std::vector<AccessPoint> access_points;
};

class NativeDirectFrontend {
 public:
  NativeDirectFrontend(NativeFrontend &owner, IDirectHttpService *service);

  void set_peer_discovery_service(IPeerDiscoveryService *service);
  void set_wifi_provisioning_info(const NativeWifiProvisioningInfo &info);
  void refresh_identity();
  void refresh();
  void loop();
  void shutdown();
  void shutdown_peer_discovery();

  size_t client_count() const { return client_count_; }
  bool wifi_configured() const { return !wifi_info_.ssid.empty(); }
  bool peer_discovery_available() const { return peer_discovery_enabled_; }
  bool raw_stream_available() const { return session_tokens_enabled_; }
  void publish_event(const char *event_name, const std::string &data_json, bool replaceable_telemetry = false);
  std::string capabilities_payload() const;
  std::string device_payload() const;
  std::string health_payload(bool online) const;
  std::string sensing_payload() const;
  std::string wifi_payload(bool mqtt_safe = false) const;
  std::string mqtt_payload() const;
  std::string ota_payload() const;
  std::string wifi_access_points_payload() const;
  std::string diagnostics_payload() const;
  bool handle_raw_stream_command(const EspectreCommand &command, const FrontendCommandContext &context,
                                 std::string *code, std::string *message, std::string *data_json);

 private:
  std::string handle_request_(const DirectRequest &request, uint64_t connection_token);
  IDirectHttpService::DeferredRequestResult handle_deferred_request_(uint64_t connection_token,
                                                                     const DirectRequest &request);
  void refresh_peer_candidate_();

  NativeFrontend &owner_;
  IDirectHttpService *service_{nullptr};
  IPeerDiscoveryService *peer_discovery_{nullptr};
  NativeWifiProvisioningInfo wifi_info_{};
  size_t client_count_{0U};
  bool peer_discovery_enabled_{false};
  bool session_tokens_enabled_{false};
  bool wifi_response_pending_{false};
  RawCsiSessionController raw_session_controller_{};
};

}  // namespace espectre
