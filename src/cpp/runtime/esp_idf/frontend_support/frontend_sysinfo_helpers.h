#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "espectre_protocol.h"

namespace esphome {
namespace espectre {

struct SysinfoCapabilities {
  bool supports_wifi_provisioning{true};
  bool supports_mqtt_config{true};
  bool supports_device_config{true};
  bool supports_runtime_threshold{false};
  bool supports_live_telemetry{false};
  bool supports_extended_diagnostics{false};
  bool supports_ota{false};
};

struct SysinfoWifiState {
  std::string ssid;
  std::string bssid;
  uint8_t channel{0U};
  bool password_set{false};
  bool connected{false};
};

struct FrontendSysinfoBase {
  const char *frontend{nullptr};
  SysinfoCapabilities capabilities{};
  EspectreDeviceConfig device_config{};
  EspectreDeviceInfo device_info{};
  bool include_proto_version{false};
  bool include_firmware_version{false};
  bool mqtt_connected{false};
  SysinfoWifiState wifi{};
};

void append_sysinfo_protocol_lines(std::vector<std::string> *lines,
                                   const char *frontend,
                                   const SysinfoCapabilities &capabilities,
                                   bool include_proto_version = false,
                                   const char *protocol_version = ESPECTRE_PROTOCOL_VERSION);

void append_sysinfo_identity_lines(std::vector<std::string> *lines,
                                   const EspectreDeviceConfig &device_config,
                                   const EspectreDeviceInfo &device_info,
                                   bool include_firmware_version);

void append_sysinfo_mqtt_lines(std::vector<std::string> *lines,
                               const EspectreDeviceConfig &device_config,
                               bool mqtt_connected);

void append_sysinfo_wifi_lines(std::vector<std::string> *lines, const SysinfoWifiState &wifi);

void append_sysinfo_network_lines(std::vector<std::string> *lines,
                                  const char *ip_address,
                                  const char *mac_address);

std::vector<std::string> build_frontend_sysinfo_lines(const FrontendSysinfoBase &base);
void append_sysinfo_end_line(std::vector<std::string> *lines);

}  // namespace espectre
}  // namespace esphome
