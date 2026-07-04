/*
 * ESPectre - ESPectre Protocol
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */

#include "espectre_protocol.h"

#include <cctype>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>

#include "base_detector.h"

namespace esphome {
namespace espectre {

namespace {

const char *motion_state_name(MotionState state) {
  return state == MotionState::MOTION ? "motion" : "idle";
}

void append_json_string(std::string *out, const char *value) {
  out->push_back('"');
  if (value != nullptr) {
    for (const char *p = value; *p != '\0'; ++p) {
      switch (*p) {
        case '"':
          out->append("\\\"");
          break;
        case '\\':
          out->append("\\\\");
          break;
        case '\n':
          out->append("\\n");
          break;
        case '\r':
          out->append("\\r");
          break;
        case '\t':
          out->append("\\t");
          break;
        default:
          out->push_back(*p);
          break;
      }
    }
  }
  out->push_back('"');
}

std::string json_pair_string(const char *key, const char *value, bool first = false) {
  std::string out;
  if (!first) {
    out.append(",");
  }
  append_json_string(&out, key);
  out.append(":");
  append_json_string(&out, value);
  return out;
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
  if (normalized == "ESP32S2") return "S2";
  if (normalized == "ESP32S3") return "S3";
  if (normalized == "ESP32") return "ESP32";
  return normalized.empty() ? "UNK" : normalized;
}

std::string extract_json_string(const std::string &payload, const char *key) {
  const std::string needle = std::string("\"") + key + "\"";
  const size_t key_pos = payload.find(needle);
  if (key_pos == std::string::npos) {
    return {};
  }
  const size_t colon = payload.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return {};
  }
  const size_t first_quote = payload.find('"', colon + 1);
  if (first_quote == std::string::npos) {
    return {};
  }
  std::string value;
  bool escaped = false;
  for (size_t i = first_quote + 1; i < payload.size(); ++i) {
    const char ch = payload[i];
    if (escaped) {
      value.push_back(ch);
      escaped = false;
      continue;
    }
    if (ch == '\\') {
      escaped = true;
      continue;
    }
    if (ch == '"') {
      return value;
    }
    value.push_back(ch);
  }
  return {};
}

std::string extract_json_number_token(const std::string &payload, const char *key) {
  const std::string needle = std::string("\"") + key + "\"";
  const size_t key_pos = payload.find(needle);
  if (key_pos == std::string::npos) {
    return {};
  }
  const size_t colon = payload.find(':', key_pos + needle.size());
  if (colon == std::string::npos) {
    return {};
  }
  size_t begin = payload.find_first_not_of(" \t\r\n", colon + 1);
  if (begin == std::string::npos) {
    return {};
  }
  size_t end = begin;
  while (end < payload.size()) {
    const char ch = payload[end];
    if ((ch >= '0' && ch <= '9') || ch == '-' || ch == '+' || ch == '.' || ch == 'e' || ch == 'E') {
      ++end;
      continue;
    }
    break;
  }
  return payload.substr(begin, end - begin);
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
  if (field == "mqtt_host") {
    config->mqtt_host = value;
    config->mqtt_enabled = !config->mqtt_host.empty();
    return true;
  }
  if (field == "mqtt_username") {
    config->mqtt_username = value;
    return true;
  }
  if (field == "mqtt_password") {
    config->mqtt_password = value;
    return true;
  }
  if (field == "topic_prefix") {
    config->topic_prefix = value.empty() ? ESPECTRE_TOPIC_PREFIX : value;
    return true;
  }
  if (field == "mqtt_port") {
    uint16_t port = 0;
    if (!parse_uint16_value(value, &port) || port == 0) {
      return false;
    }
    config->mqtt_port = port;
    return true;
  }
  if (field == "mqtt_enabled") {
    if (value == "1" || value == "true") {
      config->mqtt_enabled = true;
      return true;
    }
    if (value == "0" || value == "false") {
      config->mqtt_enabled = false;
      return true;
    }
    return false;
  }
  return false;
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
  return config.device_id == ESPECTRE_DEFAULT_DEVICE_ID ? ESPECTRE_DEFAULT_DEVICE_ID : config.device_id;
}

std::string espectre_effective_device_id(const EspectreDeviceConfig &config) {
  return format_espectre_device_id(espectre_effective_device_id_u64(config));
}

std::string espectre_effective_device_label(const EspectreDeviceConfig &config) {
  return config.device_label;
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
  config->mqtt_enabled = false;
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
  std::string out = "{";
  out += json_pair_string("protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  out += json_pair_string("device_id", device_id.c_str());
  out += json_pair_string("device_name", device_name.c_str());
  out += json_pair_string("device_label", device_label.c_str());
  out += json_pair_string("frontend", info.frontend.empty() ? "native" : info.frontend.c_str());
  out += json_pair_string("firmware_version", info.firmware_version.empty() ? "unknown" : info.firmware_version.c_str());
  out += json_pair_string("chip", info.chip.empty() ? "unknown" : info.chip.c_str());
  out += ",\"supports_ota\":";
  out += info.supports_ota ? "true" : "false";

  if (!info.network.ip_address.empty() || !info.network.mac_address.empty() || info.network.channel > 0U) {
    out += ",\"network\":{";
    bool first = true;
    if (!info.network.ip_address.empty()) {
      out += json_pair_string("ip_address", info.network.ip_address.c_str(), first);
      first = false;
    }
    if (!info.network.mac_address.empty()) {
      out += json_pair_string("mac_address", info.network.mac_address.c_str(), first);
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
    out += json_pair_string("algorithm", info.detector.c_str(), true);
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
                                float loop_time_ms) {
  (void) snapshot;
  char line[384];
  const std::string device_id = espectre_effective_device_id(config);
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

std::string espectre_command_result_payload(const EspectreDeviceConfig &config,
                                         const EspectreCommand &command,
                                         bool accepted,
                                         const char *message) {
  const std::string device_id = espectre_effective_device_id(config);
  std::string out = "{";
  out += json_pair_string("protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  out += json_pair_string("device_id", device_id.c_str());
  out += json_pair_string("command_id", command.command_id.c_str());
  out += json_pair_string("command", command.command.c_str());
  out += ",\"accepted\":";
  out += accepted ? "true" : "false";
  out += json_pair_string("message", message != nullptr ? message : "");
  out += "}";
  return out;
}

std::string espectre_ota_status_payload(const EspectreDeviceConfig &config,
                                    const EspectreOtaStatus &status,
                                    uint32_t timestamp_ms) {
  const std::string device_id = espectre_effective_device_id(config);
  std::string out = "{";
  out += json_pair_string("protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  out += json_pair_string("device_id", device_id.c_str());
  out += json_pair_string("state", ota_state_name(status.state));
  out += ",\"timestamp_ms\":";
  out += std::to_string(static_cast<unsigned>(timestamp_ms));
  out += ",\"busy\":";
  out += status.busy ? "true" : "false";
  out += ",\"update_available\":";
  out += status.update_available ? "true" : "false";
  out += json_pair_string("current_version", status.current_version.empty() ? "unknown" : status.current_version.c_str());
  out += json_pair_string("target_version", status.target_version.c_str());
  out += json_pair_string("manifest_url", status.manifest_url.c_str());
  out += json_pair_string("image_url", status.image_url.c_str());
  out += json_pair_string("message", status.message.c_str());
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
  } else if (parsed.command == "ota_check") {
    parsed.manifest_url = extract_json_string(payload, "manifest_url");
    parsed.has_manifest_url = !parsed.manifest_url.empty();
    if (!parsed.has_manifest_url) {
      if (error != nullptr) {
        *error = "missing manifest_url";
      }
      return false;
    }
  } else if (parsed.command == "ota_start") {
    parsed.manifest_url = extract_json_string(payload, "manifest_url");
    parsed.has_manifest_url = !parsed.manifest_url.empty();
    parsed.image_url = extract_json_string(payload, "image_url");
    parsed.has_image_url = !parsed.image_url.empty();
    parsed.version = extract_json_string(payload, "version");
    parsed.has_version = !parsed.version.empty();
    if (!parsed.has_manifest_url && !parsed.has_image_url) {
      if (error != nullptr) {
        *error = "missing manifest_url or image_url";
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

}  // namespace espectre
}  // namespace esphome
