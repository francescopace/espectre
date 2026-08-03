/*
 * ESPectre - Frontend Home Assistant MQTT Helpers
 *
 * Builds Home Assistant MQTT discovery and simple state topics for
 * standalone frontends while preserving the canonical ESPectre protocol.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * License: GPLv3
 */
#include "frontend_ha_mqtt_helpers.h"

#include <cctype>
#include <utility>

#include "protocol_json.h"
#include "sdkconfig.h"

#ifndef CONFIG_ESPECTRE_HA_DISCOVERY_PREFIX
#define CONFIG_ESPECTRE_HA_DISCOVERY_PREFIX "homeassistant"
#endif

namespace espectre {

namespace {

constexpr const char *kHomeAssistantBirthTopic = "homeassistant/status";
constexpr const char *kStatusAvailabilityTemplate = "{{ 'online' if value_json.online else 'offline' }}";

std::string ha_entity_base_topic(const EspectreDeviceConfig &config, const char *entity) {
  return espectre_topic(config, (std::string("ha/") + entity).c_str());
}

std::string sanitize_identifier(const std::string &value) {
  std::string sanitized;
  sanitized.reserve(value.size());
  for (const char ch : value) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      sanitized.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    } else {
      sanitized.push_back('_');
    }
  }
  return sanitized;
}

std::string build_discovery_topic(const char *component, const std::string &prefix, const std::string &object_id) {
  return prefix + "/" + component + "/" + object_id + "/config";
}

void append_discovery_device(std::string *out, const FrontendHaMqttSettings &settings, const EspectreDeviceInfo &info) {
  if (out == nullptr) {
    return;
  }
  out->append(",\"device\":{");
  append_json_string(out, "identifiers");
  out->append(":[");
  append_json_string(out, settings.device_id.c_str());
  out->append("],");
  append_json_pair(out, "name", settings.device_name.c_str(), true);
  append_json_pair(out, "manufacturer", "ESPectre");
  append_json_pair(out, "model", settings.model.c_str());
  append_json_pair(out, "sw_version", info.firmware_version.empty() ? "unknown" : info.firmware_version.c_str());
  out->append("}");
}

void append_discovery_availability(std::string *out, const FrontendHaMqttSettings &settings) {
  append_json_pair(out, "availability_topic", settings.availability_topic.c_str());
  if (!settings.availability_template.empty()) {
    append_json_pair(out, "availability_template", settings.availability_template.c_str());
  }
  append_json_pair(out, "payload_available", "online");
  append_json_pair(out, "payload_not_available", "offline");
}

std::string build_motion_discovery_payload(const FrontendHaMqttSettings &settings, const EspectreDeviceInfo &info) {
  std::string out = "{";
  append_json_pair(&out, "name", "Motion Detected", true);
  append_json_pair(&out, "unique_id", settings.motion_object_id.c_str());
  append_json_pair(&out, "object_id", settings.motion_object_id.c_str());
  append_json_pair(&out, "state_topic", settings.motion_state_topic.c_str());
  append_discovery_availability(&out, settings);
  append_json_pair(&out, "payload_on", "ON");
  append_json_pair(&out, "payload_off", "OFF");
  append_json_pair(&out, "device_class", "motion");
  append_discovery_device(&out, settings, info);
  out.push_back('}');
  return out;
}

std::string build_movement_discovery_payload(const FrontendHaMqttSettings &settings, const EspectreDeviceInfo &info) {
  std::string out = "{";
  append_json_pair(&out, "name", "Movement Score", true);
  append_json_pair(&out, "unique_id", settings.movement_object_id.c_str());
  append_json_pair(&out, "object_id", settings.movement_object_id.c_str());
  append_json_pair(&out, "state_topic", settings.movement_state_topic.c_str());
  append_discovery_availability(&out, settings);
  append_json_pair(&out, "state_class", "measurement");
  append_json_pair(&out, "icon", "mdi:sine-wave");
  append_discovery_device(&out, settings, info);
  out.push_back('}');
  return out;
}

std::string build_detector_discovery_payload(const FrontendHaMqttSettings &settings, const EspectreDeviceInfo &info) {
  std::string out = "{";
  append_json_pair(&out, "name", "Detector", true);
  append_json_pair(&out, "unique_id", settings.detector_object_id.c_str());
  append_json_pair(&out, "object_id", settings.detector_object_id.c_str());
  append_json_pair(&out, "state_topic", settings.detector_state_topic.c_str());
  append_json_pair(&out, "command_topic", settings.detector_command_topic.c_str());
  append_discovery_availability(&out, settings);
  out.append(",\"options\":[");
  append_json_string(&out, "classic");
  out.push_back(',');
  append_json_string(&out, "ml");
  out.push_back(']');
  append_discovery_device(&out, settings, info);
  out.push_back('}');
  return out;
}

}  // namespace

bool frontend_ha_mqtt_enabled() {
#ifdef CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED
  return CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED;
#else
  return false;
#endif
}

FrontendHaMqttSettings build_frontend_ha_mqtt_settings(const EspectreDeviceConfig &config,
                                                       const EspectreDeviceInfo &info,
                                                       const char *frontend_name) {
  const std::string device_id = espectre_effective_device_id(config);
  const std::string effective_name = espectre_effective_device_label(config).empty()
                                         ? espectre_device_name(espectre_effective_device_id_u64(config),
                                                                info.chip.empty() ? nullptr : info.chip.c_str())
                                         : espectre_effective_device_label(config);
  const std::string frontend = frontend_name == nullptr || frontend_name[0] == '\0' ? "device" : frontend_name;
  const std::string device_key = sanitize_identifier(frontend + "_" + device_id);
  FrontendHaMqttSettings settings{};
  settings.discovery_prefix = CONFIG_ESPECTRE_HA_DISCOVERY_PREFIX;
  settings.birth_topic = kHomeAssistantBirthTopic;
  settings.availability_topic = espectre_topic(config, "status");
  settings.availability_template = kStatusAvailabilityTemplate;
  settings.motion_state_topic = ha_entity_base_topic(config, "motion/state");
  settings.movement_state_topic = ha_entity_base_topic(config, "movement/state");
  settings.detector_state_topic = ha_entity_base_topic(config, "detector/state");
  settings.detector_command_topic = ha_entity_base_topic(config, "detector/set");
  settings.motion_object_id = device_key + "_motion";
  settings.movement_object_id = device_key + "_movement";
  settings.detector_object_id = device_key + "_detector";
  settings.device_id = device_id;
  settings.device_name = effective_name;
  settings.model = std::string("ESPectre ") + frontend;
  return settings;
}

std::vector<FrontendHaDiscoveryMessage> build_frontend_ha_discovery_messages(
    const FrontendHaMqttSettings &settings, const EspectreDeviceInfo &info, bool supports_detector) {
  std::vector<FrontendHaDiscoveryMessage> messages;
  messages.push_back(FrontendHaDiscoveryMessage{
      build_discovery_topic("binary_sensor", settings.discovery_prefix, settings.motion_object_id),
      build_motion_discovery_payload(settings, info),
  });
  messages.push_back(FrontendHaDiscoveryMessage{
      build_discovery_topic("sensor", settings.discovery_prefix, settings.movement_object_id),
      build_movement_discovery_payload(settings, info),
  });
  if (supports_detector) {
    messages.push_back(FrontendHaDiscoveryMessage{
        build_discovery_topic("select", settings.discovery_prefix, settings.detector_object_id),
        build_detector_discovery_payload(settings, info),
    });
  }
  return messages;
}

}  // namespace espectre
