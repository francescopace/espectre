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

#include "runtime/espectre_protocol.h"

namespace espectre {

/** One Home Assistant sensor fed by a diagnostic field. */
struct FrontendHaDiagnosticSensor {
  /** Entity name shown in Home Assistant. */
  std::string name;
  /** Diagnostic field the sensor reports. */
  std::string key;
  /** Home Assistant object id. */
  std::string object_id;
  /** Topic the frontend publishes this sensor's value to. */
  std::string state_topic;
  /** Optional unit; `nullptr` omits it. */
  const char *unit_of_measurement{nullptr};
  /** Optional `mdi:` icon; `nullptr` omits it. */
  const char *icon{nullptr};
  /** Optional Home Assistant device class; `nullptr` omits it. */
  const char *device_class{nullptr};
  /** Declare `state_class: measurement`, so Home Assistant keeps statistics. */
  bool state_class_measurement{true};
};

/**
 * Topics and entity ids for Home Assistant MQTT discovery.
 *
 * build_frontend_ha_mqtt_settings() fills every field from the device
 * configuration; adjust fields before building the discovery messages.
 */
struct FrontendHaMqttSettings {
  /** Discovery root, from `CONFIG_ESPECTRE_HA_DISCOVERY_PREFIX`. */
  std::string discovery_prefix;
  /** Home Assistant birth topic; republish discovery when it announces `online`. */
  std::string birth_topic;
  /** Topic whose payload reports availability; the `health` topic. */
  std::string availability_topic;
  /** Template that turns the availability payload into `online` or `offline`. */
  std::string availability_template;
  /** @name State and command topics */
  /** @{ */
  std::string motion_state_topic;
  std::string movement_state_topic;
  std::string threshold_state_topic;
  std::string threshold_command_topic;
  std::string motion_on_hits_state_topic;
  std::string motion_on_hits_command_topic;
  std::string motion_off_hits_state_topic;
  std::string motion_off_hits_command_topic;
  std::string calibrate_state_topic;
  std::string calibrate_command_topic;
  std::string detector_state_topic;
  std::string detector_command_topic;
  std::string traffic_generator_mode_state_topic;
  std::string traffic_generator_mode_command_topic;
  std::string diagnostics_command_topic;
  /** @} */
  /** @name Home Assistant object ids, each starting with ha_object_prefix */
  /** @{ */
  std::string motion_object_id;
  std::string movement_object_id;
  std::string threshold_object_id;
  std::string motion_on_hits_object_id;
  std::string motion_off_hits_object_id;
  std::string recalibrate_object_id;
  std::string calibration_active_object_id;
  std::string detector_object_id;
  std::string traffic_generator_mode_object_id;
  std::string diagnostics_object_id;
  /** @} */
  /** Prefix shared by every object id, derived from the frontend name and device id. */
  std::string ha_object_prefix;
  /** Diagnostic sensors to announce. */
  std::vector<FrontendHaDiagnosticSensor> diagnostic_sensors;
  /** Device identity in the Home Assistant device registry. */
  std::string device_id;
  /** Device name in the Home Assistant device registry. */
  std::string device_name;
  /** Device model in the Home Assistant device registry. */
  std::string model;
};

/** One retained discovery message. */
struct FrontendHaDiscoveryMessage {
  std::string topic;
  /** Discovery JSON. */
  std::string payload;
};

/** Whether Home Assistant discovery is enabled with `CONFIG_ESPECTRE_HA_DISCOVERY_ENABLED`. */
bool frontend_ha_mqtt_enabled();
/** Build the discovery topics and ids for this device; the model is `ESPectre <frontend_name>`. */
FrontendHaMqttSettings build_frontend_ha_mqtt_settings(const EspectreDeviceConfig &config,
                                                       const EspectreDeviceInfo &info,
                                                       const char *frontend_name);
/// Build one discovery message in publication order; false marks the end.
bool build_frontend_ha_discovery_message(
    const FrontendHaMqttSettings &settings,
    const EspectreDeviceInfo &info,
    bool supports_detector,
    bool supports_motion_hits,
    bool supports_traffic_control, size_t index, FrontendHaDiscoveryMessage *message);
/** Build every discovery message at once, in publication order. */
std::vector<FrontendHaDiscoveryMessage> build_frontend_ha_discovery_messages(
    const FrontendHaMqttSettings &settings,
    const EspectreDeviceInfo &info,
    bool supports_detector,
    bool supports_motion_hits,
    bool supports_traffic_control);

}  // namespace espectre
