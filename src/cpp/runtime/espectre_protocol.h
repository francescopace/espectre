/*
 * ESPectre - ESPectre Protocol
 *
 * Shared device, command, and OTA protocol types used by frontend
 * transports.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include "runtime_snapshot.h"

namespace espectre {

inline constexpr const char *ESPECTRE_PROTOCOL_VERSION = "1.0";
inline constexpr const char *ESPECTRE_TOPIC_PREFIX = "espectre/v1/devices";
inline constexpr uint64_t ESPECTRE_DEFAULT_DEVICE_ID = 0U;
inline constexpr const char *ESPECTRE_DEFAULT_DEVICE_LABEL = "";

struct EspectreDeviceConfig {
  uint64_t device_id{ESPECTRE_DEFAULT_DEVICE_ID};
  std::string device_label{ESPECTRE_DEFAULT_DEVICE_LABEL};
  std::string mqtt_host;
  uint16_t mqtt_port{1883};
  std::string mqtt_username;
  std::string mqtt_password;
  std::string topic_prefix{ESPECTRE_TOPIC_PREFIX};
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
  bool supports_info{true};
  bool supports_stats{false};
  bool supports_runtime_threshold{false};
  bool supports_runtime_detector{false};
  bool supports_ota{false};
  EspectreNetworkInfo network{};
};

struct EspectreCommand {
  std::string command_id;
  std::string command;
  float threshold{0.0f};
  bool has_threshold{false};
  std::string detector;
  bool has_detector{false};
};

enum class EspectreOtaState : uint8_t {
  IDLE = 0,
  CHECKING,
  UPDATE_AVAILABLE,
  UP_TO_DATE,
  DOWNLOADING,
  APPLYING,
  REBOOT_SCHEDULED,
  ERROR,
};

struct EspectreOtaStatus {
  EspectreOtaState state{EspectreOtaState::IDLE};
  std::string current_version{"unknown"};
  std::string target_version;
  std::string manifest_url;
  std::string image_url;
  std::string message;
  bool busy{false};
  bool update_available{false};
};

std::string format_espectre_device_id(uint64_t device_id);
bool parse_espectre_device_id(const std::string &value, uint64_t *device_id);
uint64_t espectre_device_id_from_mac(const uint8_t *mac, size_t mac_len);
std::string espectre_device_name(uint64_t device_id, const char *chip = nullptr);
uint64_t espectre_effective_device_id_u64(const EspectreDeviceConfig &config);
std::string espectre_effective_device_id(const EspectreDeviceConfig &config);
std::string espectre_effective_device_label(const EspectreDeviceConfig &config);
EspectreDeviceInfo normalize_protocol_device_info(const EspectreDeviceInfo &info,
                                                  const RuntimeSnapshot *snapshot,
                                                  bool supports_ota,
                                                  const char *default_frontend,
                                                  const char *default_chip = nullptr);
void clear_espectre_mqtt_config(EspectreDeviceConfig *config);

std::string espectre_topic(const EspectreDeviceConfig &config, const char *suffix);
std::string espectre_status_payload(const EspectreDeviceConfig &config, bool online, uint32_t timestamp_ms);
std::string espectre_info_payload(const EspectreDeviceConfig &config, const EspectreDeviceInfo &info);
std::string espectre_telemetry_payload(const EspectreDeviceConfig &config,
                                    const RuntimeSnapshot &snapshot,
                                    uint32_t timestamp_ms,
                                    uint32_t uptime_s,
                                    const char *frontend);
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
std::string espectre_ota_status_payload(const EspectreDeviceConfig &config,
                                    const EspectreOtaStatus &status,
                                    uint32_t timestamp_ms);

bool parse_espectre_command(const std::string &payload, EspectreCommand *command, std::string *error);
bool parse_espectre_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error);
bool parse_espectre_mqtt_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error);

}  // namespace espectre
