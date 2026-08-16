/*
 * ESPectre - ESPectre Protocol
 *
 * Shared device, command, and OTA protocol types used by frontend
 * transports.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "espectre_protocol.h"

#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdlib>

#include "base_detector.h"
#include "protocol_json.h"
#include "runtime_diagnostics.h"
#include "runtime_sensing_schema.h"

namespace espectre {

namespace {

const char *motion_state_name(MotionState state) {
  return state == MotionState::MOTION ? "motion" : "idle";
}

bool parse_float_value(const std::string &value, float *out) {
  if (out == nullptr || value.empty()) {
    return false;
  }
  char *end_ptr = nullptr;
  errno = 0;
  const float parsed = std::strtof(value.c_str(), &end_ptr);
  if (end_ptr == value.c_str() || end_ptr == nullptr || *end_ptr != '\0' || errno == ERANGE ||
      !std::isfinite(parsed)) {
    return false;
  }
  *out = parsed;
  return true;
}

bool parse_uint16_value(const std::string &value, uint16_t *out) {
  if (out == nullptr || value.empty()) {
    return false;
  }
  char *end_ptr = nullptr;
  errno = 0;
  const unsigned long parsed = std::strtoul(value.c_str(), &end_ptr, 10);
  if (end_ptr == value.c_str() || end_ptr == nullptr || *end_ptr != '\0' || errno == ERANGE || parsed > 65535UL) {
    return false;
  }
  *out = static_cast<uint16_t>(parsed);
  return true;
}

bool parse_uint8_value(const std::string &value, uint8_t *out) {
  uint16_t parsed = 0U;
  if (!parse_uint16_value(value, &parsed) || parsed > UINT8_MAX) {
    return false;
  }
  *out = static_cast<uint8_t>(parsed);
  return true;
}

std::string normalize_ble_chip_label(const char *chip) {
  if (chip == nullptr || chip[0] == '\0') {
    return "UNK";
  }
  std::string normalized;
  normalized.reserve(8);
  for (const char *p = chip; *p != '\0'; ++p) {
    const unsigned char ch = static_cast<unsigned char>(*p);
    if (std::isalnum(ch)) {
      normalized.push_back(static_cast<char>(std::toupper(ch)));
    }
  }
  if (normalized == "ESP32C3") return "C3";
  if (normalized == "ESP32C5") return "C5";
  if (normalized == "ESP32C6") return "C6";
  if (normalized == "ESP32S3") return "S3";
  if (normalized == "ESP32") return "ESP32";
  return normalized.empty() ? "UNK" : normalized;
}

const char *ota_state_name(EspectreOtaState state) {
  switch (state) {
    case EspectreOtaState::IDLE:
      return "idle";
    case EspectreOtaState::CHECKING:
      return "checking";
    case EspectreOtaState::UPDATE_AVAILABLE:
      return "update_available";
    case EspectreOtaState::UP_TO_DATE:
      return "up_to_date";
    case EspectreOtaState::DOWNLOADING:
      return "downloading";
    case EspectreOtaState::APPLYING:
      return "applying";
    case EspectreOtaState::REBOOT_SCHEDULED:
      return "reboot_scheduled";
    case EspectreOtaState::ERROR:
      return "error";
    default:
      return "unknown";
  }
}

bool assign_config_field(const std::string &field, const std::string &value, EspectreDeviceConfig *config) {
  if (field == "device_label") {
    config->device_label = value;
    return true;
  }
  return false;
}

bool parse_mqtt_port_value(const std::string &value, uint16_t *port) {
  return parse_uint16_value(value, port) && port != nullptr && *port > 0U;
}

}  // namespace

std::string format_espectre_device_id(uint64_t device_id) {
  char text[sizeof("0x0123456789abcdef")] = {0};
  std::snprintf(text, sizeof(text), "0x%016" PRIx64, device_id);
  return text;
}

bool parse_espectre_device_id(const std::string &value, uint64_t *device_id) {
  if (device_id == nullptr || value.empty()) {
    return false;
  }
  char *end_ptr = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end_ptr, 0);
  if (end_ptr == value.c_str() || end_ptr == nullptr || *end_ptr != '\0' || errno == ERANGE) {
    return false;
  }
  *device_id = static_cast<uint64_t>(parsed);
  return true;
}

uint64_t espectre_device_id_from_mac(const uint8_t *mac, size_t mac_len) {
  if (mac == nullptr || mac_len < 6U) {
    return ESPECTRE_DEFAULT_DEVICE_ID;
  }
  uint64_t device_id = 0U;
  for (size_t i = 0; i < 6U; ++i) {
    device_id = (device_id << 8U) | static_cast<uint64_t>(mac[i]);
  }
  return device_id;
}

std::string espectre_device_name(uint64_t device_id, const char *chip) {
  const std::string chip_label = normalize_ble_chip_label(chip);
  const std::string formatted_id = format_espectre_device_id(device_id);
  const std::string suffix = formatted_id.size() >= 6 ? formatted_id.substr(formatted_id.size() - 6) : formatted_id;
  return std::string("ESPectre ") + chip_label + " " + suffix;
}

uint64_t espectre_effective_device_id_u64(const EspectreDeviceConfig &config) {
  return config.device_id;
}

std::string espectre_effective_device_id(const EspectreDeviceConfig &config) {
  return format_espectre_device_id(espectre_effective_device_id_u64(config));
}

std::string espectre_effective_device_label(const EspectreDeviceConfig &config) {
  return config.device_label;
}

EspectreDeviceInfo normalize_protocol_device_info(const EspectreDeviceInfo &info,
                                                  const RuntimeSnapshot *snapshot,
                                                  bool supports_ota,
                                                  const char *default_frontend,
                                                  const char *default_chip) {
  EspectreDeviceInfo normalized = info;
  normalized.frontend =
      normalized.frontend.empty() ? (default_frontend != nullptr ? default_frontend : "native") : normalized.frontend;
  normalized.firmware_version = normalized.firmware_version.empty() ? "unknown" : normalized.firmware_version;
  normalized.chip = normalized.chip.empty() ? (default_chip != nullptr ? default_chip : "unknown") : normalized.chip;
  normalized.supports_ota = supports_ota;
  if (normalized.detector.empty() && snapshot != nullptr && snapshot->detector_name != nullptr) {
    normalized.detector = snapshot->detector_name;
  }
  return normalized;
}

void clear_espectre_mqtt_config(EspectreDeviceConfig *config) {
  if (config == nullptr) {
    return;
  }
  config->mqtt_host.clear();
  config->mqtt_port = 1883;
  config->mqtt_username.clear();
  config->mqtt_password.clear();
  config->topic_prefix = ESPECTRE_TOPIC_PREFIX;
}

std::string espectre_topic(const EspectreDeviceConfig &config, const char *suffix) {
  std::string topic = config.topic_prefix.empty() ? ESPECTRE_TOPIC_PREFIX : config.topic_prefix;
  if (!topic.empty() && topic.back() == '/') {
    topic.pop_back();
  }
  topic.append("/");
  topic.append(espectre_effective_device_id(config));
  topic.append("/");
  topic.append(suffix != nullptr ? suffix : "");
  return topic;
}

std::string espectre_status_payload(const EspectreDeviceConfig &config, bool online, uint32_t timestamp_ms) {
  char line[160];
  const std::string device_id = espectre_effective_device_id(config);
  std::snprintf(line,
                sizeof(line),
                "{\"protocol_version\":\"%s\",\"device_id\":\"%s\",\"online\":%s,\"timestamp_ms\":%u}",
                ESPECTRE_PROTOCOL_VERSION,
                device_id.c_str(),
                online ? "true" : "false",
                static_cast<unsigned>(timestamp_ms));
  return line;
}

std::string espectre_info_payload(const EspectreDeviceConfig &config, const EspectreDeviceInfo &info) {
  const std::string device_id = espectre_effective_device_id(config);
  const std::string device_name = espectre_device_name(espectre_effective_device_id_u64(config),
                                                       info.chip.empty() ? nullptr : info.chip.c_str());
  const std::string device_label = espectre_effective_device_label(config);
  std::string out;
  out.reserve(192U + device_id.size() + device_name.size() + device_label.size() +
              info.frontend.size() + info.firmware_version.size() + info.chip.size() +
              info.network.ip_address.size() + info.network.mac_address.size() + info.detector.size());
  out = "{";
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", device_id.c_str());
  append_json_pair(&out, "device_name", device_name.c_str());
  append_json_pair(&out, "device_label", device_label.c_str());
  append_json_pair(&out, "frontend", info.frontend.empty() ? "native" : info.frontend.c_str());
  append_json_pair(&out, "firmware_version", info.firmware_version.empty() ? "unknown" : info.firmware_version.c_str());
  append_json_pair(&out, "chip", info.chip.empty() ? "unknown" : info.chip.c_str());
  out += ",\"supports_info\":";
  out += info.supports_info ? "true" : "false";
  out += ",\"supports_stats\":";
  out += info.supports_stats ? "true" : "false";
  out += ",\"supports_runtime_threshold\":";
  out += info.supports_runtime_threshold ? "true" : "false";
  out += ",\"supports_runtime_motion_hits\":";
  out += info.supports_runtime_motion_hits ? "true" : "false";
  out += ",\"supports_runtime_detector\":";
  out += info.supports_runtime_detector ? "true" : "false";
  out += ",\"supports_manual_recalibration\":";
  out += info.supports_manual_recalibration ? "true" : "false";
  out += ",\"supports_traffic_control\":";
  out += info.supports_traffic_control ? "true" : "false";
  out += ",\"supports_ota\":";
  out += info.supports_ota ? "true" : "false";

  if (!info.network.ip_address.empty() || !info.network.mac_address.empty() || info.network.channel > 0U) {
    out += ",\"network\":{";
    bool first = true;
    if (!info.network.ip_address.empty()) {
      append_json_pair(&out, "ip_address", info.network.ip_address.c_str(), first);
      first = false;
    }
    if (!info.network.mac_address.empty()) {
      append_json_pair(&out, "mac_address", info.network.mac_address.c_str(), first);
      first = false;
    }
    if (info.network.channel > 0U) {
      if (!first) {
        out += ",";
      }
      out += "\"channel\":{\"primary\":";
      out += std::to_string(static_cast<unsigned>(info.network.channel));
      out += "}";
    }
    out += "}";
  }

  if (!info.detector.empty()) {
    out += ",\"detection\":{";
    append_json_pair(&out, "algorithm", info.detector.c_str(), true);
    out += "}";
  }
  out += "}";
  return out;
}

std::string espectre_telemetry_payload(const EspectreDeviceConfig &config,
                                    const RuntimeSnapshot &snapshot,
                                    uint32_t timestamp_ms,
                                    uint32_t uptime_s,
                                    const char *frontend) {
  char line[320];
  const std::string device_id = espectre_effective_device_id(config);
  std::snprintf(line,
                sizeof(line),
                "{\"protocol_version\":\"%s\",\"device_id\":\"%s\",\"frontend\":\"%s\","
                "\"timestamp_ms\":%u,\"motion_state\":\"%s\",\"movement_score\":%.6g,"
                "\"threshold\":%.6g,\"detector\":\"%s\",\"health\":{\"uptime_s\":%u}}",
                ESPECTRE_PROTOCOL_VERSION,
                device_id.c_str(),
                frontend != nullptr && frontend[0] != '\0' ? frontend : "unknown",
                static_cast<unsigned>(timestamp_ms),
                motion_state_name(snapshot.motion_state),
                static_cast<double>(snapshot.movement_metric),
                static_cast<double>(snapshot.threshold),
                snapshot.detector_name != nullptr ? snapshot.detector_name : "unknown",
                static_cast<unsigned>(uptime_s));
  return line;
}

std::string espectre_stats_payload(const EspectreDeviceConfig &config,
                                const RuntimeSnapshot &snapshot,
                                uint32_t timestamp_ms,
                                uint32_t uptime_s,
                                float free_memory_kb,
                                float loop_time_ms,
                                const RuntimeDiagnosticsSample *diagnostics) {
  (void) snapshot;
  char line[640];
  const std::string device_id = espectre_effective_device_id(config);
  if (diagnostics == nullptr) {
    std::snprintf(line,
                  sizeof(line),
                  "{\"protocol_version\":\"%s\",\"device_id\":\"%s\",\"timestamp_ms\":%u,"
                  "\"uptime\":%u,\"free_memory_kb\":%.6g,\"loop_time_ms\":%.6g}",
                  ESPECTRE_PROTOCOL_VERSION,
                  device_id.c_str(),
                  static_cast<unsigned>(timestamp_ms),
                  static_cast<unsigned>(uptime_s),
                  static_cast<double>(free_memory_kb),
                  static_cast<double>(loop_time_ms));
    return line;
  }
  const char *wifi_rssi = diagnostics->wifi_rssi_dbm == INT8_MIN ? "null" : nullptr;
  char wifi_rssi_value[16];
  if (wifi_rssi == nullptr) {
    std::snprintf(wifi_rssi_value, sizeof(wifi_rssi_value), "%d", diagnostics->wifi_rssi_dbm);
    wifi_rssi = wifi_rssi_value;
  }
  std::snprintf(line,
                sizeof(line),
                "{\"protocol_version\":\"%s\",\"device_id\":\"%s\",\"timestamp_ms\":%u,"
                "\"uptime\":%u,\"free_memory_kb\":%.6g,\"loop_time_ms\":%.6g,"
                "\"traffic_tx_pps\":%.6g,\"csi_callback_pps\":%.6g,"
                "\"csi_accepted_pps\":%.6g,\"csi_admitted_pps\":%.6g,"
                "\"csi_filtered_pps\":%.6g,\"csi_missing_slots_pps\":%.6g,"
                "\"csi_excess_pps\":%.6g,\"csi_stale_pps\":%.6g,"
                "\"csi_out_of_order_pps\":%.6g,\"csi_occupancy\":%.6g,"
                "\"wifi_channel\":%u,\"wifi_rssi_dbm\":%s}",
                ESPECTRE_PROTOCOL_VERSION,
                device_id.c_str(),
                static_cast<unsigned>(timestamp_ms),
                static_cast<unsigned>(uptime_s),
                static_cast<double>(free_memory_kb),
                static_cast<double>(loop_time_ms),
                static_cast<double>(diagnostics->traffic_tx_pps),
                static_cast<double>(diagnostics->csi_callback_pps),
                static_cast<double>(diagnostics->csi_accepted_pps),
                static_cast<double>(diagnostics->csi_admitted_pps),
                static_cast<double>(diagnostics->csi_filtered_pps),
                static_cast<double>(diagnostics->csi_missing_slots_pps),
                static_cast<double>(diagnostics->csi_excess_pps),
                static_cast<double>(diagnostics->csi_stale_pps),
                static_cast<double>(diagnostics->csi_out_of_order_pps),
                static_cast<double>(diagnostics->csi_occupancy_ratio),
                static_cast<unsigned>(diagnostics->wifi_channel),
                wifi_rssi);
  return line;
}

std::string espectre_command_result_payload(const EspectreDeviceConfig &config,
                                         const EspectreCommand &command,
                                         bool accepted,
                                         const char *message) {
  const std::string device_id = espectre_effective_device_id(config);
  std::string out;
  out.reserve(128U + device_id.size() + command.command_id.size() + command.command.size() +
              (message != nullptr ? std::strlen(message) : 0U));
  out = "{";
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", device_id.c_str());
  append_json_pair(&out, "command_id", command.command_id.c_str());
  append_json_pair(&out, "command", command.command.c_str());
  out += ",\"accepted\":";
  out += accepted ? "true" : "false";
  append_json_pair(&out, "message", message != nullptr ? message : "");
  out += "}";
  return out;
}

std::string espectre_ota_status_payload(const EspectreDeviceConfig &config,
                                    const EspectreOtaStatus &status,
                                    uint32_t timestamp_ms) {
  const std::string device_id = espectre_effective_device_id(config);
  std::string out;
  out.reserve(192U + device_id.size() + status.current_version.size() + status.target_version.size() +
              status.manifest_url.size() + status.image_url.size() + status.message.size());
  out = "{";
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", device_id.c_str());
  append_json_pair(&out, "state", ota_state_name(status.state));
  out += ",\"timestamp_ms\":";
  out += std::to_string(static_cast<unsigned>(timestamp_ms));
  out += ",\"busy\":";
  out += status.busy ? "true" : "false";
  out += ",\"update_available\":";
  out += status.update_available ? "true" : "false";
  append_json_pair(&out, "current_version", status.current_version.empty() ? "unknown" : status.current_version.c_str());
  append_json_pair(&out, "target_version", status.target_version.c_str());
  append_json_pair(&out, "manifest_url", status.manifest_url.c_str());
  append_json_pair(&out, "image_url", status.image_url.c_str());
  append_json_pair(&out, "message", status.message.c_str());
  out += "}";
  return out;
}

bool parse_espectre_command(const std::string &payload, EspectreCommand *command, std::string *error) {
  if (command == nullptr) {
    return false;
  }
  EspectreCommand parsed;
  parsed.command_id = extract_json_string(payload, "command_id");
  parsed.command = extract_json_string(payload, "command");
  if (parsed.command.empty()) {
    if (error != nullptr) {
      *error = "missing command";
    }
    return false;
  }
  if (parsed.command == "set_threshold") {
    const std::string threshold_token = extract_json_number_token(payload, "threshold");
    if (!parse_float_value(threshold_token, &parsed.threshold)) {
      if (error != nullptr) {
        *error = "invalid threshold";
      }
      return false;
    }
    parsed.has_threshold = true;
  } else if (parsed.command == "set_motion_hits") {
    const std::string motion_on_hits_token = extract_json_number_token(payload, "motion_on_hits");
    const std::string motion_off_hits_token = extract_json_number_token(payload, "motion_off_hits");
    if (!parse_uint8_value(motion_on_hits_token, &parsed.motion_on_hits) ||
        !parse_uint8_value(motion_off_hits_token, &parsed.motion_off_hits)) {
      if (error != nullptr) {
        *error = "invalid motion hits";
      }
      return false;
    }
    parsed.has_motion_hits = true;
  } else if (parsed.command == "set_csi_traffic_mode") {
    parsed.csi_traffic_mode = extract_json_string(payload, "csi_traffic_mode");
    if (parsed.csi_traffic_mode != RUNTIME_CSI_TRAFFIC_MODE_INTERNAL_NAME &&
        parsed.csi_traffic_mode != RUNTIME_CSI_TRAFFIC_MODE_EXTERNAL_NAME &&
        parsed.csi_traffic_mode != RUNTIME_CSI_TRAFFIC_MODE_PACING_NAME &&
        parsed.csi_traffic_mode != RUNTIME_CSI_TRAFFIC_MODE_DISABLED_NAME) {
      if (error != nullptr) {
        *error = "invalid csi traffic mode";
      }
      return false;
    }
    parsed.has_csi_traffic_mode = true;
  } else if (parsed.command == "set_traffic_generator_mode") {
    parsed.traffic_generator_mode = extract_json_string(payload, "traffic_generator_mode");
    if (parsed.traffic_generator_mode != RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME &&
        parsed.traffic_generator_mode != RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME) {
      if (error != nullptr) {
        *error = "invalid traffic generator mode";
      }
      return false;
    }
    parsed.has_traffic_generator_mode = true;
  } else if (parsed.command == "set_detector") {
    parsed.detector = extract_json_string(payload, "detector");
    if (parsed.detector != RUNTIME_DETECTION_ALGORITHM_LIGHTWEIGHT_NAME &&
        parsed.detector != RUNTIME_DETECTION_ALGORITHM_HIGH_ACCURACY_NAME) {
      if (error != nullptr) {
        *error = "invalid detector";
      }
      return false;
    }
    parsed.has_detector = true;
  } else if (parsed.command == "recalibrate") {
    // No additional payload required.
  } else if (parsed.command == "ota_check" || parsed.command == "ota_start") {
    if (has_json_key(payload, "manifest_url") || has_json_key(payload, "image_url") ||
        has_json_key(payload, "version")) {
      if (error != nullptr) {
        *error = "ota overrides are not supported";
      }
      return false;
    }
  } else if (parsed.command == "ota_status") {
    // No additional payload required.
  }
  *command = parsed;
  return true;
}

bool parse_espectre_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error) {
  if (config == nullptr) {
    return false;
  }
  constexpr const char *prefix = "SET_DEVICE_CONFIG:";
  if (command.rfind(prefix, 0) != 0) {
    if (error != nullptr) {
      *error = "invalid prefix";
    }
    return false;
  }
  const std::string body = command.substr(std::string(prefix).size());
  const size_t equal = body.find('=');
  if (equal == std::string::npos) {
    if (error != nullptr) {
      *error = "expected key=value";
    }
    return false;
  }
  const std::string field = body.substr(0, equal);
  const std::string value = body.substr(equal + 1);
  if (!assign_config_field(field, value, config)) {
    if (error != nullptr) {
      *error = "invalid config field";
    }
    return false;
  }
  return true;
}

bool parse_espectre_mqtt_config_command(const std::string &command, EspectreDeviceConfig *config, std::string *error) {
  if (config == nullptr) {
    if (error != nullptr) {
      *error = "config output is required";
    }
    return false;
  }
  constexpr const char *prefix = "SET_MQTT_CONFIG:";
  if (command.rfind(prefix, 0) != 0) {
    if (error != nullptr) {
      *error = "invalid mqtt config command";
    }
    return false;
  }

  std::vector<std::pair<std::string, std::string>> pairs;
  if (!parse_urlencoded_key_value_pairs(command.substr(std::strlen(prefix)), &pairs, error)) {
    return false;
  }

  bool has_host = false;
  bool has_port = false;
  for (const auto &pair : pairs) {
    if (pair.first == "host") {
      config->mqtt_host = pair.second;
      has_host = true;
      continue;
    }
    if (pair.first == "port") {
      uint16_t port = 0U;
      if (!parse_mqtt_port_value(pair.second, &port)) {
        if (error != nullptr) {
          *error = "mqtt port must be 1..65535";
        }
        return false;
      }
      config->mqtt_port = port;
      has_port = true;
      continue;
    }
    if (pair.first == "username") {
      config->mqtt_username = pair.second;
      continue;
    }
    if (pair.first == "password") {
      config->mqtt_password = pair.second;
      continue;
    }
    if (pair.first == "topic_prefix") {
      config->topic_prefix = pair.second.empty() ? ESPECTRE_TOPIC_PREFIX : pair.second;
      continue;
    }
    if (error != nullptr) {
      *error = "unsupported mqtt config field";
    }
    return false;
  }

  if (!has_host || config->mqtt_host.empty()) {
    if (error != nullptr) {
      *error = "missing mqtt host";
    }
    return false;
  }
  if (!has_port) {
    if (error != nullptr) {
      *error = "missing mqtt port";
    }
    return false;
  }
  return true;
}

}  // namespace espectre
