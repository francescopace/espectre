/*
 * ESPectre - Frontend Home Assistant MQTT Helpers
 *
 * Builds Home Assistant MQTT discovery and simple state topics for
 * standalone frontends while preserving the canonical ESPectre protocol.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <string>
#include <vector>

#include "espectre_protocol.h"

namespace espectre {

struct FrontendHaMqttSettings {
  std::string discovery_prefix;
  std::string birth_topic;
  std::string availability_topic;
  std::string availability_template;
  std::string motion_state_topic;
  std::string movement_state_topic;
  std::string detector_state_topic;
  std::string detector_command_topic;
  std::string motion_object_id;
  std::string movement_object_id;
  std::string detector_object_id;
  std::string device_id;
  std::string device_name;
  std::string model;
};

struct FrontendHaDiscoveryMessage {
  std::string topic;
  std::string payload;
};

bool frontend_ha_mqtt_enabled();
FrontendHaMqttSettings build_frontend_ha_mqtt_settings(const EspectreDeviceConfig &config,
                                                       const EspectreDeviceInfo &info,
                                                       const char *frontend_name);
std::vector<FrontendHaDiscoveryMessage> build_frontend_ha_discovery_messages(
    const FrontendHaMqttSettings &settings, const EspectreDeviceInfo &info, bool supports_detector);

}  // namespace espectre
