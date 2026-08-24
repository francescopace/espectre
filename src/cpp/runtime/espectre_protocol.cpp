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
#include <utility>

#include "base_detector.h"
#include "protocol_json.h"
#include "runtime_diagnostics.h"
#include "runtime_sensing_schema.h"

namespace espectre {

namespace {

const char *motion_state_name(MotionState state) {
  return state == MotionState::MOTION ? "motion" : "idle";
}

double json_finite(float value) {
  return std::isfinite(value) ? static_cast<double>(value) : 0.0;
}

void append_command_descriptor(std::string *out,
                               bool *first,
                               bool enabled,
                               const char *name,
                               const char *kind,
                               const char *access,
                               const char *properties,
                               const char *required,
                               const char *result_schema) {
  if (out == nullptr || first == nullptr || !enabled || name == nullptr) {
    return;
  }
  if (!*first) {
    out->append(",");
  }
  *first = false;
  out->append("{\"name\":\"");
  out->append(name);
  out->append("\",\"kind\":\"");
  out->append(kind != nullptr ? kind : "query");
  out->append("\",\"access\":\"");
  out->append(access != nullptr ? access : "read");
  out->append("\",\"params\":{");
  if (properties != nullptr && properties[0] != '\0') {
    out->append("\"type\":\"object\",\"properties\":{");
    out->append(properties);
    out->append("}");
  }
  if (required != nullptr && required[0] != '\0') {
    out->append(",\"required\":[");
    out->append(required);
    out->append("]");
  }
  if (properties != nullptr && properties[0] != '\0') out->append(",");
  out->append("\"additionalProperties\":false}");
  if (result_schema != nullptr && result_schema[0] != '\0') {
    out->append(",\"result\":\"");
    out->append(result_schema);
    out->append("\"");
  }
  out->append("}");
}

void append_capability_commands(std::string *out,
                                const EspectreDeviceInfo &info,
                                bool supports_status,
                                bool supports_config,
                                bool supports_sensing_control,
                                bool supports_wifi_config,
                                bool supports_mqtt_config,
                                bool supports_peer_discovery) {
  if (out == nullptr) {
    return;
  }
  bool first = true;
  const auto add = [&](bool enabled,
                       const char *name,
                       const char *kind,
                       const char *access,
                       const char *properties = "",
                       const char *required = "",
                       const char *result = nullptr) {
    append_command_descriptor(out, &first, enabled, name, kind, access, properties, required, result);
  };
  add(true, "capabilities", "query", "read", "", "", "capabilities");
  add(info.supports_info, "info", "query", "read", "", "", "info");
  add(supports_status, "status", "query", "read", "", "", "status");
  add(supports_config, "config", "query", "read", "", "", "config");
  add(info.supports_diagnostics, "diagnostics", "query", "read", "", "", "diagnostics");
  add(supports_sensing_control,
      "set_sensing",
      "mutation",
      "control",
      "\"enabled\":{\"type\":\"boolean\"}",
      "\"enabled\"");
  add(info.supports_device_config,
      "set_device_label",
      "mutation",
      "device_admin",
      "\"device_label\":{\"type\":\"string\"}",
      "\"device_label\"");
  add(info.supports_runtime_threshold,
      "set_threshold",
      "mutation",
      "control",
      "\"threshold\":{\"type\":\"number\",\"minimum\":0,\"maximum\":1}",
      "\"threshold\"");
  add(info.supports_runtime_motion_hits,
      "set_motion_hits",
      "mutation",
      "control",
      "\"motion_on_hits\":{\"type\":\"integer\",\"minimum\":1,\"maximum\":20},"
      "\"motion_off_hits\":{\"type\":\"integer\",\"minimum\":1,\"maximum\":20}",
      "\"motion_on_hits\",\"motion_off_hits\"");
  add(info.supports_runtime_detector,
      "set_detector",
      "mutation",
      "control",
      "\"detector\":{\"type\":\"string\",\"enum\":[\"lightweight\",\"high_accuracy\"]}",
      "\"detector\"");
  add(info.supports_manual_recalibration, "recalibrate", "action", "control");
  add(info.supports_traffic_control,
      "set_csi_traffic_mode",
      "mutation",
      "control",
      "\"csi_traffic_mode\":{\"type\":\"string\",\"enum\":[\"internal\",\"external\",\"disabled\"]}",
      "\"csi_traffic_mode\"");
  add(info.supports_traffic_control,
      "set_traffic_generator_mode",
      "mutation",
      "control",
      "\"traffic_generator_mode\":{\"type\":\"string\",\"enum\":[\"ping\",\"dns\"]}",
      "\"traffic_generator_mode\"");
  add(supports_wifi_config,
      "set_wifi_config",
      "mutation",
      "network_admin",
      "\"ssid\":{\"type\":\"string\"},\"password\":{\"type\":\"string\"},"
      "\"bssid\":{\"type\":\"string\"},\"channel\":{\"type\":\"integer\",\"minimum\":0,\"maximum\":255},"
      "\"band_policy\":{\"type\":\"string\",\"enum\":[\"2g\",\"5g\",\"auto\"]}");
  add(supports_wifi_config, "clear_wifi_config", "mutation", "network_admin");
  add(supports_mqtt_config,
      "set_mqtt_config",
      "mutation",
      "network_admin",
      "\"host\":{\"type\":\"string\"},\"port\":{\"type\":\"integer\",\"minimum\":1,\"maximum\":65535},"
      "\"username\":{\"type\":\"string\"},\"password\":{\"type\":\"string\"},"
      "\"topic_prefix\":{\"type\":\"string\"}",
      "\"host\"");
  add(supports_mqtt_config, "clear_mqtt_config", "mutation", "network_admin");
  add(info.supports_ota, "ota_status", "query", "firmware_update", "", "", "ota_status");
  add(info.supports_ota,
      "ota_check",
      "action",
      "firmware_update",
      "\"channel\":{\"type\":\"string\",\"enum\":[\"release\",\"preview\",\"develop\"]}");
  add(info.supports_ota,
      "ota_start",
      "action",
      "firmware_update",
      "\"channel\":{\"type\":\"string\",\"enum\":[\"release\",\"preview\",\"develop\"]}");
  add(supports_peer_discovery, "discover_peers", "query", "discovery", "", "", "peers");
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

std::string normalize_chip_label(const char *chip) {
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

bool single_line_string(const std::string &value, size_t max_size) {
  return value.size() <= max_size && value.find_first_of("\r\n\0", 0U, 3U) == std::string::npos;
}

bool bssid_string_accepted(const std::string &value) {
  if (value.empty()) {
    return true;
  }
  if (value.size() != 17U) {
    return false;
  }
  for (size_t index = 0U; index < value.size(); ++index) {
    if (index % 3U == 2U) {
      if (value[index] != ':') {
        return false;
      }
    } else if (!std::isxdigit(static_cast<unsigned char>(value[index]))) {
      return false;
    }
  }
  return true;
}

bool parse_command_fields(const std::string &command_id,
                          const std::string &command_name,
                          const std::vector<JsonObjectField> &fields,
                          EspectreCommand *command,
                          std::string *error) {
  if (command == nullptr) {
    if (error != nullptr) {
      *error = "command output is required";
    }
    return false;
  }
  EspectreCommand parsed;
  parsed.command_id = command_id;
  parsed.command = command_name;
  const auto reject = [&](const char *message) {
    if (error != nullptr) {
      *error = message;
    }
    *command = parsed;
    return false;
  };
  const auto string_field = [&](const char *name, std::string *value) {
    const JsonObjectField *field = find_json_object_field(fields, name);
    if (field == nullptr || field->type != JsonValueType::STRING || value == nullptr) {
      return false;
    }
    *value = field->value;
    return true;
  };
  const auto number_field = [&](const char *name, std::string *value) {
    const JsonObjectField *field = find_json_object_field(fields, name);
    if (field == nullptr || field->type != JsonValueType::NUMBER || value == nullptr) {
      return false;
    }
    *value = field->value;
    return true;
  };
  const auto bool_field = [&](const char *name, bool *value) {
    const JsonObjectField *field = find_json_object_field(fields, name);
    if (field == nullptr || field->type != JsonValueType::BOOLEAN || value == nullptr) {
      return false;
    }
    *value = field->value == "true";
    return true;
  };

  if (parsed.command.empty()) {
    return reject("missing command");
  }
  const auto field_allowed = [&parsed](const std::string &name) {
    if (name == "protocol_version" || name == "command_id" || name == "command") return true;
    if (parsed.command == "set_sensing") return name == "enabled";
    if (parsed.command == "set_device_label") return name == "device_label";
    if (parsed.command == "set_threshold") return name == "threshold";
    if (parsed.command == "set_motion_hits") return name == "motion_on_hits" || name == "motion_off_hits";
    if (parsed.command == "set_detector") return name == "detector";
    if (parsed.command == "set_csi_traffic_mode") return name == "csi_traffic_mode";
    if (parsed.command == "set_traffic_generator_mode") return name == "traffic_generator_mode";
    if (parsed.command == "set_wifi_config") {
      return name == "ssid" || name == "password" || name == "bssid" || name == "channel" ||
             name == "band_policy";
    }
    if (parsed.command == "set_mqtt_config") {
      return name == "host" || name == "port" || name == "username" || name == "password" ||
             name == "topic_prefix";
    }
    if (parsed.command == "ota_check" || parsed.command == "ota_start") {
      return name == "channel" || name == "manifest_url" || name == "image_url" || name == "version";
    }
    return false;
  };
  for (const JsonObjectField &field : fields) {
    if (!field_allowed(field.name)) return reject("unknown command parameter");
  }
  if (parsed.command == "set_sensing") {
    if (!bool_field("enabled", &parsed.sensing_enabled)) {
      return reject("invalid sensing state (accepted: boolean enabled)");
    }
    parsed.has_sensing_enabled = true;
  } else if (parsed.command == "set_device_label") {
    if (!string_field("device_label", &parsed.device_label) ||
        parsed.device_label.find_first_of("\r\n\0", 0U, 3U) != std::string::npos) {
      return reject("invalid device label (accepted: a single-line string)");
    }
    parsed.has_device_label = true;
  } else if (parsed.command == "set_threshold") {
    std::string threshold_token;
    if (!number_field("threshold", &threshold_token) || !parse_float_value(threshold_token, &parsed.threshold)) {
      return reject("invalid threshold (accepted: 0.0-1.0)");
    }
    parsed.has_threshold = true;
  } else if (parsed.command == "set_motion_hits") {
    std::string motion_on_hits_token;
    std::string motion_off_hits_token;
    if (!number_field("motion_on_hits", &motion_on_hits_token) ||
        !number_field("motion_off_hits", &motion_off_hits_token) ||
        !parse_uint8_value(motion_on_hits_token, &parsed.motion_on_hits) ||
        !parse_uint8_value(motion_off_hits_token, &parsed.motion_off_hits)) {
      return reject("invalid motion hits (accepted: motion_on_hits and motion_off_hits in 1-20)");
    }
    parsed.has_motion_hits = true;
  } else if (parsed.command == "set_csi_traffic_mode") {
    if (!string_field("csi_traffic_mode", &parsed.csi_traffic_mode) ||
        (parsed.csi_traffic_mode != RUNTIME_CSI_TRAFFIC_MODE_INTERNAL_NAME &&
         parsed.csi_traffic_mode != RUNTIME_CSI_TRAFFIC_MODE_EXTERNAL_NAME &&
         parsed.csi_traffic_mode != RUNTIME_CSI_TRAFFIC_MODE_DISABLED_NAME)) {
      return reject("invalid csi traffic mode (accepted: internal, external, and disabled)");
    }
    parsed.has_csi_traffic_mode = true;
  } else if (parsed.command == "set_traffic_generator_mode") {
    if (!string_field("traffic_generator_mode", &parsed.traffic_generator_mode) ||
        (parsed.traffic_generator_mode != RUNTIME_TRAFFIC_GENERATOR_MODE_PING_NAME &&
         parsed.traffic_generator_mode != RUNTIME_TRAFFIC_GENERATOR_MODE_DNS_NAME)) {
      return reject("invalid traffic generator mode (accepted: ping and dns)");
    }
    parsed.has_traffic_generator_mode = true;
  } else if (parsed.command == "set_detector") {
    if (!string_field("detector", &parsed.detector) ||
        (parsed.detector != RUNTIME_DETECTION_ALGORITHM_LIGHTWEIGHT_NAME &&
         parsed.detector != RUNTIME_DETECTION_ALGORITHM_HIGH_ACCURACY_NAME)) {
      return reject("invalid detector (accepted: lightweight and high_accuracy)");
    }
    parsed.has_detector = true;
  } else if (parsed.command == "set_wifi_config") {
    if (find_json_object_field(fields, "ssid") != nullptr) {
      if (!string_field("ssid", &parsed.wifi_ssid) || parsed.wifi_ssid.empty() ||
          !single_line_string(parsed.wifi_ssid, 32U)) {
        return reject("invalid SSID (accepted: 1..32 bytes)");
      }
      parsed.has_wifi_ssid = true;
    }
    if (find_json_object_field(fields, "password") != nullptr) {
      if (!string_field("password", &parsed.wifi_password) || !single_line_string(parsed.wifi_password, 63U)) {
        return reject("invalid Wi-Fi password (accepted: 0..63 bytes)");
      }
      parsed.has_wifi_password = true;
    }
    if (find_json_object_field(fields, "bssid") != nullptr) {
      if (!string_field("bssid", &parsed.wifi_bssid) || !bssid_string_accepted(parsed.wifi_bssid)) {
        return reject("invalid BSSID (accepted: empty or six hexadecimal octets)");
      }
      parsed.has_wifi_bssid = true;
    }
    if (find_json_object_field(fields, "channel") != nullptr) {
      std::string channel_token;
      if (!number_field("channel", &channel_token) || !parse_uint8_value(channel_token, &parsed.wifi_channel)) {
        return reject("invalid Wi-Fi channel");
      }
      parsed.has_wifi_channel = true;
    }
    if (find_json_object_field(fields, "band_policy") != nullptr) {
      if (!string_field("band_policy", &parsed.wifi_band_policy) ||
          (parsed.wifi_band_policy != "2g" && parsed.wifi_band_policy != "5g" &&
           parsed.wifi_band_policy != "auto")) {
        return reject("invalid Wi-Fi band policy (accepted: 2g, 5g, and auto)");
      }
      parsed.has_wifi_band_policy = true;
    }
    if (!parsed.has_wifi_ssid && !parsed.has_wifi_password && !parsed.has_wifi_bssid && !parsed.has_wifi_channel &&
        !parsed.has_wifi_band_policy) {
      return reject("set_wifi_config requires at least one field");
    }
  } else if (parsed.command == "clear_wifi_config") {
    // No additional payload required.
  } else if (parsed.command == "set_mqtt_config") {
    if (!string_field("host", &parsed.mqtt_host) || parsed.mqtt_host.empty() ||
        !single_line_string(parsed.mqtt_host, 253U)) {
      return reject("invalid MQTT host");
    }
    parsed.has_mqtt_host = true;
    if (find_json_object_field(fields, "port") != nullptr) {
      std::string port_token;
      if (!number_field("port", &port_token) || !parse_mqtt_port_value(port_token, &parsed.mqtt_port)) {
        return reject("invalid MQTT port (accepted: 1..65535)");
      }
      parsed.has_mqtt_port = true;
    }
    if (find_json_object_field(fields, "username") != nullptr) {
      if (!string_field("username", &parsed.mqtt_username) || !single_line_string(parsed.mqtt_username, 128U)) {
        return reject("invalid MQTT username");
      }
      parsed.has_mqtt_username = true;
    }
    if (find_json_object_field(fields, "password") != nullptr) {
      if (!string_field("password", &parsed.mqtt_password) || !single_line_string(parsed.mqtt_password, 256U)) {
        return reject("invalid MQTT password");
      }
      parsed.has_mqtt_password = true;
    }
    if (find_json_object_field(fields, "topic_prefix") != nullptr) {
      if (!string_field("topic_prefix", &parsed.mqtt_topic_prefix) ||
          !single_line_string(parsed.mqtt_topic_prefix, 128U)) {
        return reject("invalid MQTT topic prefix");
      }
      parsed.has_mqtt_topic_prefix = true;
    }
  } else if (parsed.command == "clear_mqtt_config") {
    // No additional payload required.
  } else if (parsed.command == "recalibrate") {
    // No additional payload required.
  } else if (parsed.command == "ota_check" || parsed.command == "ota_start") {
    if (find_json_object_field(fields, "manifest_url") != nullptr ||
        find_json_object_field(fields, "image_url") != nullptr ||
        find_json_object_field(fields, "version") != nullptr) {
      return reject("ota overrides are not supported (manifest_url, image_url, and version are not accepted)");
    }
    if (find_json_object_field(fields, "channel") != nullptr) {
      if (!string_field("channel", &parsed.ota_channel) || !espectre_ota_channel_accepted(parsed.ota_channel)) {
        return reject("invalid ota channel (accepted: release, preview, and develop)");
      }
      parsed.has_ota_channel = true;
    }
  } else if (parsed.command == "ota_status" || parsed.command == "info" ||
             parsed.command == "capabilities" || parsed.command == "status" ||
             parsed.command == "config" || parsed.command == "diagnostics" ||
             parsed.command == "discover_peers") {
    // No additional payload required.
  } else {
    // The command engine owns registry filtering and returns the stable
    // `unsupported` code. The transport parser only validates the envelope and
    // parameters it knows how to decode.
  }
  *command = std::move(parsed);
  return true;
}

}  // namespace

std::string format_espectre_device_id(uint64_t device_id) {
  char text[sizeof("0123456789abcdef")] = {0};
  std::snprintf(text, sizeof(text), "%016" PRIx64, device_id);
  return text;
}

bool parse_espectre_device_id(const std::string &value, uint64_t *device_id) {
  if (device_id == nullptr || value.empty()) {
    return false;
  }
  char *end_ptr = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(value.c_str(), &end_ptr, 16);
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
  for (size_t i = 0U; i < 6U; ++i) {
    device_id = (device_id << 8U) | static_cast<uint64_t>(mac[i]);
  }
  return device_id;
}

std::string espectre_device_name(uint64_t device_id, const char *chip) {
  const std::string chip_label = normalize_chip_label(chip);
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
  out.reserve(256U + device_id.size() + device_name.size() + device_label.size() +
              info.frontend.size() + info.firmware_version.size() + info.chip.size() +
              info.detector.size() + info.csi_traffic_mode.size() + info.traffic_mode.size());
  out = "{";
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", device_id.c_str());
  append_json_pair(&out, "device_name", device_name.c_str());
  append_json_pair(&out, "device_label", device_label.c_str());
  append_json_pair(&out, "frontend", info.frontend.empty() ? "native" : info.frontend.c_str());
  append_json_pair(&out, "firmware_version", info.firmware_version.empty() ? "unknown" : info.firmware_version.c_str());
  append_json_pair(&out, "chip", info.chip.empty() ? "unknown" : info.chip.c_str());
  if (info.network.channel > 0U) {
    out += ",\"network\":{\"channel\":{\"primary\":";
    out += std::to_string(static_cast<unsigned>(info.network.channel));
    out += "}";
    out += "}";
  }

  if (!info.detector.empty()) {
    out += ",\"detection\":{";
    append_json_pair(&out, "algorithm", info.detector.c_str(), true);
    out += "}";
  }
  if (!info.csi_traffic_mode.empty()) {
    append_json_pair(&out, "csi_traffic_mode", info.csi_traffic_mode.c_str());
  }
  if (!info.traffic_mode.empty()) {
    append_json_pair(&out, "traffic_mode", info.traffic_mode.c_str());
  }
  if (info.csi_target_pps > 0U) {
    out += ",\"csi_target_pps\":";
    out += std::to_string(static_cast<unsigned>(info.csi_target_pps));
  }
  if (info.evaluation_interval_ms > 0U) {
    out += ",\"evaluation_interval_ms\":";
    out += std::to_string(static_cast<unsigned>(info.evaluation_interval_ms));
  }
  if (info.publish_interval_ms > 0U) {
    out += ",\"publish_interval_ms\":";
    out += std::to_string(static_cast<unsigned>(info.publish_interval_ms));
  }
  out += "}";
  return out;
}

std::string espectre_capabilities_payload(const EspectreDeviceConfig &config,
                                          const EspectreDeviceInfo &info,
                                          bool supports_status,
                                          bool supports_config,
                                          bool supports_sensing_control,
                                          bool supports_wifi_config,
                                          bool supports_mqtt_config,
                                          bool supports_peer_discovery) {
  const std::string device_id = espectre_effective_device_id(config);
  std::string out;
  out.reserve(3072U + device_id.size());
  out = "{";
  append_json_pair(&out, "protocol_version", ESPECTRE_PROTOCOL_VERSION, true);
  append_json_pair(&out, "device_id", device_id.c_str());
  out += ",\"commands\":[";
  append_capability_commands(&out,
                             info,
                             supports_status,
                             supports_config,
                             supports_sensing_control,
                             supports_wifi_config,
                             supports_mqtt_config,
                             supports_peer_discovery);
  out += "],\"events\":[\"telemetry\",\"status\",\"info\",\"config\"";
  if (info.supports_ota) out += ",\"ota_status\"";
  out += ",\"fault\"],"
         "\"config_sections\":[\"runtime\"";
  if (info.supports_device_config) out += ",\"device\"";
  if (supports_wifi_config) out += ",\"wifi\"";
  if (supports_mqtt_config) out += ",\"mqtt\"";
  out += "],\"features\":{\"raw_csi\":false}}";
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
                json_finite(snapshot.movement_metric),
                json_finite(snapshot.threshold),
                snapshot.detector_name != nullptr ? snapshot.detector_name : "unknown",
                static_cast<unsigned>(uptime_s));
  return line;
}

std::string espectre_diagnostics_payload(const EspectreDeviceConfig &config,
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
                                            const char *code,
                                            const char *message,
                                            const std::string &data_json) {
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
  append_json_pair(&out, "code", code != nullptr ? code : (accepted ? "ok" : "internal_error"));
  append_json_pair(&out, "message", message != nullptr ? message : "");
  if (!data_json.empty()) {
    std::vector<JsonObjectField> fields;
    if (parse_json_object_fields(data_json, &fields, nullptr)) {
      out += ",\"data\":";
      out += data_json;
    }
  }
  out += "}";
  return out;
}

std::string espectre_ota_status_payload(const EspectreDeviceConfig &config,
                                    const EspectreOtaStatus &status,
                                    uint32_t timestamp_ms) {
  const std::string device_id = espectre_effective_device_id(config);
  std::string out;
  out.reserve(192U + device_id.size() + status.current_version.size() + status.target_version.size() +
              status.manifest_url.size() + status.image_url.size() + status.message.size() +
              status.default_channel.size() + status.channel.size());
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
  append_json_pair(&out, "default_channel", status.default_channel.c_str());
  append_json_pair(&out, "channel", status.channel.c_str());
  append_json_pair(&out, "message", status.message.c_str());
  out += "}";
  return out;
}

bool parse_espectre_command(const std::string &payload, EspectreCommand *command, std::string *error) {
  if (command == nullptr) {
    if (error != nullptr) {
      *error = "command output is required";
    }
    return false;
  }
  std::vector<JsonObjectField> fields;
  std::string json_error;
  if (!parse_json_object_fields(payload, &fields, &json_error)) {
    if (error != nullptr) {
      *error = json_error.empty() ? "invalid command JSON" : json_error;
    }
    *command = EspectreCommand{};
    return false;
  }
  std::string command_id;
  const JsonObjectField *id_field = find_json_object_field(fields, "command_id");
  if (id_field != nullptr) {
    if (id_field->type != JsonValueType::STRING) {
      if (error != nullptr) {
        *error = "invalid command_id (accepted: string)";
      }
      *command = EspectreCommand{};
      return false;
    }
    command_id = id_field->value;
  }
  const JsonObjectField *command_field = find_json_object_field(fields, "command");
  if (command_field == nullptr || command_field->type != JsonValueType::STRING) {
    if (error != nullptr) {
      *error = "missing command";
    }
    *command = EspectreCommand{};
    command->command_id = std::move(command_id);
    return false;
  }
  return parse_command_fields(command_id, command_field->value, fields, command, error);
}

bool parse_espectre_command_request(const std::string &command_id,
                                    const std::string &command_name,
                                    const std::string &params_json,
                                    EspectreCommand *command,
                                    std::string *error) {
  std::vector<JsonObjectField> fields;
  std::string json_error;
  if (!parse_json_object_fields(params_json, &fields, &json_error)) {
    if (error != nullptr) {
      *error = json_error.empty() ? "invalid command params" : json_error;
    }
    if (command != nullptr) {
      *command = EspectreCommand{};
      command->command_id = command_id;
      command->command = command_name;
    }
    return false;
  }
  return parse_command_fields(command_id, command_name, fields, command, error);
}

bool espectre_ota_channel_accepted(const std::string &channel) {
  return channel == ESPECTRE_OTA_CHANNEL_RELEASE || channel == ESPECTRE_OTA_CHANNEL_PREVIEW ||
         channel == ESPECTRE_OTA_CHANNEL_DEVELOP;
}

std::string espectre_ota_manifest_url(const char *frontend, const char *chip, const std::string &channel) {
  if (frontend == nullptr || frontend[0] == '\0' || chip == nullptr || chip[0] == '\0' ||
      !espectre_ota_channel_accepted(channel)) {
    return {};
  }
  std::string url = "https://github.com/francescopace/espectre/releases/";
  if (channel == ESPECTRE_OTA_CHANNEL_RELEASE) {
    url += "latest/download/";
  } else {
    url += "download/";
    url += (channel == ESPECTRE_OTA_CHANNEL_PREVIEW) ? ESPECTRE_OTA_RELEASE_TAG_PREVIEW
                                                     : ESPECTRE_OTA_RELEASE_TAG_DEVELOP;
    url += "/";
  }
  url += "espectre-";
  url += frontend;
  url += "-ota-";
  url += chip;
  url += ".json";
  return url;
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
