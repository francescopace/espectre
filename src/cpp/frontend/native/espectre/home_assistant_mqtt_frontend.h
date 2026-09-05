/*
 * ESPectre - Native Home Assistant MQTT Frontend Adapter
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#pragma once

#include <cstdint>
#include <string>

#include "frontend_ha_mqtt_helpers.h"
#include "mqtt_transport.h"
#include "runtime_events.h"

namespace espectre {

class NativeFrontend;

class HomeAssistantMqttFrontend {
 public:
  HomeAssistantMqttFrontend(NativeFrontend &owner, IMqttTransport *transport);

  void setup();
  void set_online(bool online);
  void cancel_pending_state();
  void schedule_discovery();
  void drain_pending_snapshot();
  void publish_motion(MotionState state);
  void publish_movement(float movement);
  void publish_threshold(float threshold);
  void publish_motion_hits(uint8_t motion_on_hits, uint8_t motion_off_hits);
  void publish_calibrate(bool calibrating);
  void publish_detector(const char *detector_name);
  void publish_traffic_control(CsiTrafficMode csi_traffic_mode, RuntimeTrafficMode traffic_generator_mode);
  void publish_diagnostics();
  void publish_state(const RuntimeSnapshot &snapshot);
  void publish_current_state();

 private:
  bool ready_();
  void handle_birth_message_(const std::string &topic, const std::string &payload);
  void handle_threshold_command_(const std::string &payload);
  void handle_motion_hits_command_(bool motion_on, const std::string &payload);
  void handle_calibrate_command_(const std::string &payload);
  void handle_csi_traffic_mode_command_(const std::string &payload);
  void handle_traffic_generator_mode_command_(const std::string &payload);

  NativeFrontend &owner_;
  IMqttTransport *transport_{nullptr};
  FrontendHaMqttSettings settings_{};
  bool pending_discovery_{false};
  FrontendHaDiscoveryMessage pending_discovery_message_{};
  size_t pending_discovery_index_{0U};
  bool online_{false};
  bool pending_state_{false};
};

}  // namespace espectre
