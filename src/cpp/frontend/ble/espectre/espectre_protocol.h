/*
 * ESPectre - ESPectre Protocol
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "runtime_snapshot.h"

namespace esphome {
namespace espectre {

inline constexpr const char *ESPECTRE_PROTOCOL_VERSION = "1.0";
inline constexpr const char *ESPECTRE_TOPIC_PREFIX = "espectre/v1/devices";
inline constexpr const char *ESPECTRE_DEFAULT_DEVICE_ID = "espectre-node";
inline constexpr const char *ESPECTRE_DEFAULT_DEVICE_NAME = "ESPectre Node";

struct EspectreDeviceConfig {
  std::string device_id{ESPECTRE_DEFAULT_DEVICE_ID};
  std::string device_name{ESPECTRE_DEFAULT_DEVICE_NAME};
  std::string mqtt_host;
  uint16_t mqtt_port{1883};
  std::string mqtt_username;
  std::string mqtt_password;
  std::string topic_prefix{ESPECTRE_TOPIC_PREFIX};
  bool mqtt_enabled{false};
};

struct EspectreNetworkInfo {
  std::string ip_address;
  std::string mac_address;
  uint8_t channel{0U};
};

struct EspectreDeviceInfo {
  std::string frontend{"ble"};
  std::string firmware_version{"unknown"};
  std::string chip{"unknown"};
  std::string detector;
  EspectreNetworkInfo network{};
};

struct EspectreCommand {
  std::string command_id;
  std::string command;
  float threshold{0.0f};
  bool has_threshold{false};
};

std::string espectre_effective_device_id(const EspectreDeviceConfig &config);
std::string espectre_effective_device_name(const EspectreDeviceConfig &config);
std::string espectre_ble_device_name(const EspectreDeviceConfig &config);
void clear_espectre_mqtt_config(EspectreDeviceConfig *config);

std::string espectre_topic(const EspectreDeviceConfig &config, const char *suffix);
std::string espectre_status_payload(const EspectreDeviceConfig &config, bool online, uint32_t timestamp_ms);
std::string espectre_info_payload(const EspectreDeviceConfig &config, const EspectreDeviceInfo &info);
std::string espectre_telemetry_payload(const EspectreDeviceConfig &config,
                                    const RuntimeSnapshot &snapshot,
                                    uint32_t timestamp_ms,
                                    uint32_t uptime_s);
std::string espectre_stats_payload(const EspectreDeviceConfig &config,
                                const RuntimeSnapshot &snapshot,
                                uint32_t timestamp_ms,
                                uint32_t uptime_s,
                                float free_memory_kb,
                                float loop_time_ms);
std::string espectre_command_result_payload(const EspectreDeviceConfig &config,
                                         const EspectreCommand &command,
                                         bool accepted,
                                         const char *message);

bool parse_espectre_command(const std::string &payload, EspectreCommand *command, std::string *error);
bool parse_espectre_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error);

}  // namespace espectre
}  // namespace esphome
