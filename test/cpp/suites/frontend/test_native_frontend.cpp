/*
 * ESPectre - Native Frontend Unit Tests
 *
 * Unit tests for Native Frontend.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <algorithm>
#include <vector>

#define private public
#define protected public
#include "native_frontend.h"
#include "recovery_button_service.h"
#undef protected
#undef private

#include "direct_websocket_service_mock.h"
#include "frontend_runtime_shim.h"
#include "mqtt_transport_mock.h"
#include "ota_service_mock.h"

using namespace espectre;
using espectre::direct_websocket_service_mock::MockDirectWebSocketService;
using espectre::mqtt_transport_mock::MockMqttTransport;
using espectre::ota_service_mock::MockOtaService;

namespace {

RuntimeSnapshot make_ready_snapshot() {
  RuntimeSnapshot snapshot{};
  snapshot.ready_to_publish = true;
  snapshot.motion_state = MotionState::MOTION;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.startup_threshold = 0.42f;
  snapshot.detector_name = "lightweight";
  return snapshot;
}

bool has_mqtt_publish(const std::string &topic, const char *payload = nullptr) {
  return std::any_of(mqtt_transport_mock::state.publishes.begin(),
                     mqtt_transport_mock::state.publishes.end(),
                     [&](const mqtt_transport_mock::Publish &publish) {
                       if (publish.topic != topic) {
                         return false;
                       }
                       return payload == nullptr || publish.payload == payload;
                     });
}

int mqtt_publish_index(const std::string &topic) {
  const auto &publishes = mqtt_transport_mock::state.publishes;
  for (size_t i = 0; i < publishes.size(); ++i) {
    if (publishes[i].topic == topic) {
      return static_cast<int>(i);
    }
  }
  return -1;
}

}  // namespace

void setUp(void) {
  frontend_runtime_shim::reset();
  direct_websocket_service_mock::reset();
  mqtt_transport_mock::reset();
  ota_service_mock::reset();
}

void tearDown(void) {}

void test_native_frontend_setup_registers_runtime_listener(void) {
  frontend_runtime_shim::state.snapshot.threshold = 3.25f;

  NativeFrontend frontend;
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.is_setup_complete());
  TEST_ASSERT_NOT_NULL(frontend_runtime_shim::state.last_listener);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener != &frontend);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_live_telemetry_enabled_calls);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);
  TEST_ASSERT_EQUAL_FLOAT(3.25f, frontend.snapshot().threshold);
}

void test_native_frontend_setup_fails_when_runtime_setup_fails(void) {
  frontend_runtime_shim::state.setup_result = false;
  NativeFrontend frontend;
  TEST_ASSERT_FALSE(frontend.setup());
}

void test_native_frontend_loop_and_shutdown_forward_to_runtime(void) {
  {
    NativeFrontend frontend;
    TEST_ASSERT_TRUE(frontend.setup());
    frontend.loop();
    TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
  }

  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
}

void test_native_frontend_mqtt_connect_publishes_ha_discovery_and_subscribes_birth_topics(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  config.device_label = "Kitchen Sensor";
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_connection(true);

  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic == "homeassistant/status";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/threshold/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/motion_on_hits/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/motion_off_hits/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/calibrate/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/detector/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/csi_traffic_mode/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/traffic_generator_mode/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0000111122223333/ha/diagnostics/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0000111122223333_motion_detected/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Motion Detected\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/sensor/native_0000111122223333_movement_score/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Movement Score\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/sensor/native_0000111122223333_intensity/config" &&
                                        publish.retain && publish.payload.empty();
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0000111122223333_motion/config" &&
                                        publish.retain && publish.payload.empty();
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/switch/native_0000111122223333_calibrate/config" &&
                                        publish.retain && publish.payload.empty();
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/sensor/native_0000111122223333_traffic_tx_rate/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Traffic TX Rate\"") != std::string::npos &&
                                        publish.payload.find("\"entity_category\":\"diagnostic\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/button/native_0000111122223333_refresh_diagnostics/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Refresh Diagnostics\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/number/native_0000111122223333_threshold/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Threshold\"") != std::string::npos &&
                                        publish.payload.find("\"command_topic\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/number/native_0000111122223333_motion_on_hits/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Motion On Hits\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/number/native_0000111122223333_motion_off_hits/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Motion Off Hits\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/switch/native_0000111122223333_trigger_calibration/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Trigger Calibration\"") != std::string::npos &&
                                        publish.payload.find("\"command_topic\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/select/native_0000111122223333_detection_profile/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Detection Profile\"") !=
                                            std::string::npos &&
                                        publish.payload.find("\"entity_category\":\"config\"") !=
                                            std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/select/native_0000111122223333_csi_traffic_ownership/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"CSI Traffic Ownership\"") !=
                                            std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/select/native_0000111122223333_csi_traffic_source/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"CSI Traffic Source\"") !=
                                            std::string::npos;
                               }));
  const int csi_traffic_discovery = mqtt_publish_index(
      "homeassistant/select/native_0000111122223333_csi_traffic_ownership/config");
  const int traffic_generator_discovery = mqtt_publish_index(
      "homeassistant/select/native_0000111122223333_csi_traffic_source/config");
  const int calibrate_discovery =
      mqtt_publish_index("homeassistant/switch/native_0000111122223333_trigger_calibration/config");
  TEST_ASSERT_TRUE(csi_traffic_discovery >= 0);
  TEST_ASSERT_TRUE(csi_traffic_discovery < traffic_generator_discovery);
  TEST_ASSERT_TRUE(traffic_generator_discovery < calibrate_discovery);
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0000111122223333_motion_detected/config" &&
                                        publish.payload.find(
                                            "\"availability_topic\":\"espectre/v1/devices/0000111122223333/status\"") !=
                                            std::string::npos &&
                                        publish.payload.find("\"availability_template\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0000111122223333/ha/motion/state" &&
                                        publish.payload == "ON";
                               }));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/motion_on_hits/state", "4"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/motion_off_hits/state", "3"));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0000111122223333/ha/movement/state" &&
                                        publish.payload == "2.7500";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0000111122223333/ha/threshold/state" &&
                                        publish.payload == "1.5000";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0000111122223333/ha/calibrate/state" &&
                                        publish.payload == "OFF";
                               }));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/csi_traffic_mode/state", "internal"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/traffic_generator_mode/state", "ping"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/traffic_tx_rate/state"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.shutdown();
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0000111122223333/status" &&
                                        publish.payload.find("\"online\":false") != std::string::npos && publish.retain;
                               }));
}

void test_native_frontend_ha_birth_message_republishes_discovery_and_state(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("homeassistant/status", "online");

  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0000111122223333_motion_detected/config" &&
                                        publish.retain;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0000111122223333/ha/movement/state";
                               }));
  TEST_ASSERT_TRUE(std::none_of(mqtt_transport_mock::state.publishes.begin(),
                                mqtt_transport_mock::state.publishes.end(),
                                [](const mqtt_transport_mock::Publish &publish) {
                                  return publish.topic.find("/ha/intensity/state") != std::string::npos;
                                }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/sensor/native_0000111122223333_traffic_tx_rate/config" &&
                                        publish.retain;
                               }));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/traffic_tx_rate/state"));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0000111122223333/ha/threshold/state";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0000111122223333/ha/calibrate/state";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0000111122223333/status" &&
                                        publish.payload.find("\"online\":true") != std::string::npos && publish.retain;
                               }));
}

void test_native_frontend_retries_the_complete_ha_snapshot_after_queue_backpressure(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_runtime_config(RuntimeConfig{});
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();
  mqtt_transport_mock::state.diagnostics.queue_capacity = 16U;
  mqtt_transport_mock::state.diagnostics.queued_publishes = 0U;

  mqtt.emit_connection(true);
  TEST_ASSERT_EQUAL(16U, mqtt_transport_mock::state.publishes.size());
  TEST_ASSERT_TRUE(frontend.pending_ha_discovery_index_ < frontend.pending_ha_discovery_.size());
  TEST_ASSERT_TRUE(frontend.pending_ha_state_);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/motion/state"));

  frontend.loop();
  TEST_ASSERT_EQUAL(16U, mqtt_transport_mock::state.publishes.size());
  TEST_ASSERT_TRUE(frontend.pending_ha_state_);

  mqtt_transport_mock::state.diagnostics.queued_publishes = 0U;
  frontend.loop();
  TEST_ASSERT_FALSE(frontend.pending_ha_discovery_.empty());
  TEST_ASSERT_TRUE(frontend.pending_ha_state_);

  mqtt_transport_mock::state.diagnostics.queued_publishes = 0U;
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.pending_ha_discovery_.empty());
  TEST_ASSERT_TRUE(frontend.pending_ha_state_);

  mqtt_transport_mock::state.diagnostics.queued_publishes = 0U;
  frontend.loop();
  TEST_ASSERT_FALSE(frontend.pending_ha_state_);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000111122223333/ha/motion/state", "ON"));
  TEST_ASSERT_TRUE(has_mqtt_publish(
      "homeassistant/sensor/native_0000111122223333_csi_temporal_occupancy/config"));
  TEST_ASSERT_TRUE(has_mqtt_publish(
      "homeassistant/sensor/native_0000111122223333_csi_occupancy/config", ""));
}

void test_native_frontend_ha_entities_follow_esphome_cadences(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  mqtt_transport_mock::state.publishes.clear();
  RuntimeSnapshot snapshot = make_ready_snapshot();
  frontend.on_live_telemetry(snapshot.movement_metric, snapshot.threshold);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.empty());
  frontend.loop();
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/movement/state", "2.7500"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/telemetry"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/threshold/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/intensity/state"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.on_periodic_update(snapshot, 10);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/movement/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/telemetry"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/intensity/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/threshold/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.on_motion_state_changed(snapshot);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.empty());
  frontend.loop();
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion/state", "ON"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/telemetry"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/movement/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/intensity/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/threshold/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state"));

  mqtt_transport_mock::state.publishes.clear();
  snapshot.threshold = 0.45f;
  frontend_runtime_shim::state.last_listener->on_threshold_changed(snapshot);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/threshold/state", "0.4500"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/intensity/state"));
}

void test_native_frontend_mqtt_connect_enables_live_telemetry(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);

  mqtt.emit_connection(true);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  mqtt.emit_connection(false);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);
}

void test_native_frontend_direct_reports_mqtt_connection_changes(void) {
  MockMqttTransport mqtt;
  MockDirectWebSocketService direct;
  EspectreDeviceConfig config;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt, nullptr, &direct);
  frontend.set_device_config(config);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);

  mqtt.emit_connection(true);
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_websocket_service_mock::state.published_events.size()));
  TEST_ASSERT_EQUAL_STRING(
      "status", direct_websocket_service_mock::state.published_events[0].event_name.c_str());
  TEST_ASSERT_TRUE(
      direct_websocket_service_mock::state.published_events[0].data_json.find("\"mqtt_connected\":true") !=
      std::string::npos);

  direct_websocket_service_mock::state.published_events.clear();
  mqtt.emit_connection(false);
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_websocket_service_mock::state.published_events.size()));
  TEST_ASSERT_EQUAL_STRING(
      "status", direct_websocket_service_mock::state.published_events[0].event_name.c_str());
  TEST_ASSERT_TRUE(
      direct_websocket_service_mock::state.published_events[0].data_json.find("\"mqtt_connected\":false") !=
      std::string::npos);
}

void test_native_frontend_direct_clear_mqtt_disconnects_and_reports_status(void) {
  MockMqttTransport mqtt;
  MockDirectWebSocketService direct;
  EspectreDeviceConfig config;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt, nullptr, &direct);
  frontend.set_device_config(config);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);
  mqtt.emit_connection(true);
  direct_websocket_service_mock::state.published_events.clear();

  const std::string response = direct.emit_request(
      DirectWebSocketRequest{"clear-mqtt", "clear_mqtt_config", "{}"});

  TEST_ASSERT_TRUE(response.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.shutdown_called);
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_websocket_service_mock::state.published_events.size()));
  TEST_ASSERT_EQUAL_STRING(
      "status", direct_websocket_service_mock::state.published_events[0].event_name.c_str());
  TEST_ASSERT_TRUE(
      direct_websocket_service_mock::state.published_events[0].data_json.find("\"mqtt_connected\":false") !=
      std::string::npos);
  TEST_ASSERT_TRUE(
      direct_websocket_service_mock::state.published_events[0].data_json.find("\"mqtt_configured\":false") !=
      std::string::npos);

  const std::string status =
      direct.emit_request(DirectWebSocketRequest{"read-status", "status", "{}"});
  TEST_ASSERT_TRUE(status.find("\"mqtt_connected\":false") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"mqtt_configured\":false") != std::string::npos);
}

void test_native_frontend_ha_threshold_command_updates_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/threshold/set", "0.45");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.45f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/threshold/state", "0.4500"));
}

void test_native_frontend_ha_motion_hits_commands_update_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/motion_on_hits/set", "6");
  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/motion_off_hits/set", "4");

  TEST_ASSERT_EQUAL(2, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL_UINT8(6U, frontend_runtime_shim::state.last_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(4U, frontend_runtime_shim::state.last_motion_off_hits);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion_on_hits/state", "6"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion_off_hits/state", "4"));
}

void test_native_frontend_ha_calibrate_command_triggers_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/set", "ON");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state", "ON"));

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/set", "OFF");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state", "ON"));

  mqtt_transport_mock::state.publishes.clear();
  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.calibrating = true;
  frontend_runtime_shim::state.last_listener->on_calibration_started(snapshot);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state", "ON"));

  mqtt_transport_mock::state.publishes.clear();
  snapshot.calibrating = false;
  snapshot.threshold = 0.42f;
  frontend.on_calibration_finished(snapshot, true);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state", "OFF"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/threshold/state", "0.4200"));
}

void test_native_frontend_ha_calibrate_command_respects_manual_recalibration_capability(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  frontend_runtime_shim::state.capabilities.supports_manual_recalibration = false;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/set", "ON");

  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/calibrate/state", "OFF"));
}

void test_native_frontend_ha_traffic_control_commands_update_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/csi_traffic_mode/set", "external");
  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/traffic_generator_mode/set", "dns");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_csi_traffic_mode == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_traffic_generator_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_traffic_generator_mode == RuntimeTrafficMode::DNS);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_traffic_mode/state", "external"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/traffic_generator_mode/state", "dns"));

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/csi_traffic_mode/set", "pacing");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_csi_traffic_mode == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_traffic_mode/state", "pacing"));
}

void test_native_frontend_ha_diagnostics_button_publishes_cached_sample(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  frontend.latest_diagnostics_.traffic_tx_pps = 100.0f;
  frontend.latest_diagnostics_.csi_callback_pps = 96.0f;
  frontend.latest_diagnostics_.csi_accepted_pps = 90.0f;
  frontend.latest_diagnostics_.csi_admitted_pps = 84.0f;
  frontend.latest_diagnostics_.csi_filtered_pps = 6.0f;
  frontend.latest_diagnostics_.csi_missing_slots_pps = 10.0f;
  frontend.latest_diagnostics_.csi_excess_pps = 6.0f;
  frontend.latest_diagnostics_.csi_stale_pps = 0.0f;
  frontend.latest_diagnostics_.csi_out_of_order_pps = 0.0f;
  frontend.latest_diagnostics_.csi_occupancy_ratio = 0.84f;
  frontend.latest_diagnostics_.wifi_channel = 10U;
  frontend.latest_diagnostics_.wifi_rssi_dbm = -55;
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0000abcdeffedcba/ha/diagnostics/set", "PRESS");

  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/traffic_tx_rate/state", "100.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_callback_rate/state", "96.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_accepted_rate/state", "90.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_admitted_rate/state", "84.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_filtered_rate/state", "6.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_missing_rate/state", "10.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_excess_rate/state", "6.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_stale_rate/state", "0.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_out_of_order_rate/state", "0.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/csi_occupancy/state", "84.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/wifi_channel/state", "10"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/wifi_rssi/state", "-55"));
}

void test_native_frontend_live_telemetry_publishes_mqtt_telemetry(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  frontend.on_periodic_update(snapshot, 10);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/telemetry"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.on_live_telemetry(snapshot.movement_metric, snapshot.threshold);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.empty());
  frontend.loop();
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0000abcdeffedcba/telemetry" &&
                                        publish.payload.find("\"motion_state\":\"motion\"") != std::string::npos &&
                                        publish.payload.find("\"movement_score\":2.75") != std::string::npos;
                               }));
}

void test_native_frontend_motion_edge_publishes_ready_ha_motion(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.ready_to_publish = false;
  frontend.on_motion_state_changed(snapshot);
  frontend.loop();
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/telemetry"));

  snapshot.ready_to_publish = true;
  frontend.on_motion_state_changed(snapshot);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion/state"));
  frontend.loop();
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/ha/motion/state", "ON"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/telemetry"));
}

void test_native_frontend_mqtt_set_threshold_command_publishes_result(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command("{\"command_id\":\"cmd-1\",\"command\":\"set_threshold\",\"threshold\":0.45}");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.45f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/accepted", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_set_device_label_persists_and_republishes_info(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.device_label = "Living Room";
  config.mqtt_host = "localhost";
  std::vector<EspectreDeviceConfig> persisted_configs;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  frontend.set_device_config_change_callback(
      [&persisted_configs](const EspectreDeviceConfig &updated, bool clear, std::string *) {
        TEST_ASSERT_FALSE(clear);
        persisted_configs.push_back(updated);
        return true;
      });
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command(
      "{\"command_id\":\"label-1\",\"command\":\"set_device_label\",\"device_label\":\"Kitchen Sensor\"}");

  TEST_ASSERT_EQUAL_STRING("Kitchen Sensor", frontend.device_config().device_label.c_str());
  TEST_ASSERT_EQUAL(1, static_cast<int>(persisted_configs.size()));
  TEST_ASSERT_EQUAL_STRING("Kitchen Sensor", persisted_configs[0].device_label.c_str());
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0000abcdeffedcba/info" &&
                                        publish.retain &&
                                        publish.payload.find("\"device_label\":\"Kitchen Sensor\"") !=
                                            std::string::npos;
                               }));
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &result = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/accepted", result.topic.c_str());
  TEST_ASSERT_TRUE(result.payload.find("\"command\":\"set_device_label\"") != std::string::npos);
  TEST_ASSERT_TRUE(result.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_recalibrate_command_publishes_result(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command("{\"command_id\":\"cmd-recal\",\"command\":\"recalibrate\"}");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/accepted", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_detector_command_updates_runtime(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command(
      "{\"command_id\":\"det-1\",\"command\":\"set_detector\",\"detector\":\"lightweight\"}");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_detector_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_detector == DetectionAlgorithm::LIGHTWEIGHT);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":true") !=
                   std::string::npos);
}

void test_native_frontend_mqtt_motion_hits_command_updates_runtime(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command(
      "{\"command_id\":\"motion-1\",\"command\":\"set_motion_hits\",\"motion_on_hits\":5,\"motion_off_hits\":3}");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL_UINT8(5U, frontend_runtime_shim::state.last_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(3U, frontend_runtime_shim::state.last_motion_off_hits);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":true") !=
                   std::string::npos);
}

void test_native_frontend_mqtt_traffic_commands_update_runtime(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command(
      "{\"command_id\":\"traffic-1\",\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"pacing\"}");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":false") !=
                   std::string::npos);

  mqtt.emit_command(
      "{\"command_id\":\"traffic-1b\",\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"external\"}");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_csi_traffic_mode == CsiTrafficMode::EXTERNAL);

  mqtt.emit_command(
      "{\"command_id\":\"traffic-2\",\"command\":\"set_traffic_generator_mode\",\"traffic_generator_mode\":\"ping\"}");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_traffic_generator_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_traffic_generator_mode == RuntimeTrafficMode::PING);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":true") !=
                   std::string::npos);
}

void test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";
  frontend_runtime_shim::state.diagnostics.traffic_packets_total = 100U;
  frontend_runtime_shim::state.diagnostics.csi_callbacks_total = 100U;
  frontend_runtime_shim::state.diagnostics.csi_accepted_total = 90U;
  frontend_runtime_shim::state.diagnostics.csi_filtered_total = 10U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  const RuntimeDiagnosticsSnapshot diagnostics_baseline = frontend_runtime_shim::state.diagnostics;
  const uint32_t diagnostics_baseline_ms = frontend.now_ms_();
  frontend.diagnostics_sampler_.reset(diagnostics_baseline, diagnostics_baseline_ms);
  frontend_runtime_shim::state.diagnostics.traffic_packets_total = 600U;
  frontend_runtime_shim::state.diagnostics.csi_callbacks_total = 580U;
  frontend_runtime_shim::state.diagnostics.csi_accepted_total = 540U;
  frontend_runtime_shim::state.diagnostics.csi_filtered_total = 40U;
  frontend_runtime_shim::state.diagnostics.wifi_channel = 10U;
  frontend_runtime_shim::state.diagnostics.wifi_rssi_dbm = -55;
  mqtt_transport_mock::state.publishes.clear();
  frontend.sample_diagnostics_(diagnostics_baseline_ms + 5000U);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.empty());

  RuntimeSnapshot snapshot = make_ready_snapshot();
  frontend.on_motion_state_changed(snapshot);
  frontend.loop();
  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"command_id\":\"cmd-info\",\"command\":\"info\"}");
  mqtt.emit_command("{\"command_id\":\"cmd-stats\",\"command\":\"stats\"}");

  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.size() >= 4);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/info",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].retain);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_runtime_motion_hits\":true") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_device_config\":true") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_manual_recalibration\":true") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_traffic_control\":true") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"csi_traffic_mode\":\"internal\"") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"traffic_mode\":\"ping\"") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"csi_target_pps\":100") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"evaluation_interval_ms\":250") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"publish_interval_ms\":1000") !=
                   std::string::npos);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/accepted",
                           mqtt_transport_mock::state.publishes[1].topic.c_str());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/stats",
                           mqtt_transport_mock::state.publishes[2].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"uptime\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"free_memory_kb\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"loop_time_ms\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"traffic_tx_pps\":100") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"csi_callback_pps\":96") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"csi_accepted_pps\":90") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"csi_filtered_pps\":6") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"wifi_channel\":10") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"wifi_rssi_dbm\":-55") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"movement\":") == std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"threshold\":") == std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"state\":") == std::string::npos);

  mqtt.emit_command("{\"command_id\":\"cmd-commands\",\"command\":\"commands\"}");
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.size() >= 6);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/catalog",
                           mqtt_transport_mock::state.publishes[4].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[4].payload.find("\"commands\":[") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[4].payload.find("\"set_device_label\"") != std::string::npos);
}

void test_native_frontend_mqtt_connect_publishes_current_ota_state(void) {
  MockMqttTransport mqtt;
  MockOtaService ota;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt, &ota);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.firmware_version = "1.2.3";
  frontend.set_device_config(config);
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_connection(true);
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0000abcdeffedcba/ota/state" &&
                                        publish.payload.find("\"state\":\"idle\"") != std::string::npos &&
                                        publish.payload.find("\"current_version\":\"1.2.3\"") != std::string::npos &&
                                        !publish.retain;
                               }));
}

void test_native_frontend_mqtt_ota_commands_use_ota_service_and_publish_state(void) {
  MockMqttTransport mqtt;
  MockOtaService ota;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&mqtt, &ota);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.firmware_version = "1.0.0";
  frontend.set_device_config(config);
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command("{\"command_id\":\"cmd-ota-check\",\"command\":\"ota_check\"}");
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_check_calls);
  TEST_ASSERT_EQUAL_STRING("1.0.0", ota_service_mock::state.last_current_version.c_str());
  TEST_ASSERT_TRUE(ota_service_mock::state.last_channel.empty());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/accepted",
                           mqtt_transport_mock::state.publishes.back().topic.c_str());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"command_id\":\"cmd-ota-check-preview\",\"command\":\"ota_check\",\"channel\":\"preview\"}");
  TEST_ASSERT_EQUAL(2, ota_service_mock::state.start_check_calls);
  TEST_ASSERT_EQUAL_STRING("preview", ota_service_mock::state.last_channel.c_str());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"command_id\":\"cmd-ota-start\",\"command\":\"ota_start\"}");
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_update_calls);
  TEST_ASSERT_EQUAL_STRING("1.0.0", ota_service_mock::state.last_current_version.c_str());

  mqtt_transport_mock::state.publishes.clear();
  EspectreOtaStatus ota_status;
  ota_status.state = EspectreOtaState::UPDATE_AVAILABLE;
  ota_status.current_version = "1.0.0";
  ota_status.target_version = "1.1.0";
  ota_status.image_url = "https://fw.example/native.bin";
  ota.emit_status(ota_status);
  TEST_ASSERT_EQUAL(1, static_cast<int>(mqtt_transport_mock::state.publishes.size()));
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/ota/state",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"state\":\"update_available\"") !=
                   std::string::npos);
}

void test_native_frontend_ota_prepare_quiesces_transports_and_recovers_on_error(void) {
  MockMqttTransport mqtt;
  MockOtaService ota;
  EspectreDeviceConfig config;
  config.mqtt_host = "localhost";
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "HomeNet";

  NativeFrontend frontend(&mqtt, &ota);
  frontend.set_device_config(config);
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  ota.emit_prepare();

  TEST_ASSERT_TRUE(frontend.ota_frontend_quiesced_);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.shutdown_called);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);

  EspectreOtaStatus downloading;
  downloading.state = EspectreOtaState::DOWNLOADING;
  downloading.busy = true;
  ota.emit_status(downloading);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.empty());

  const int mqtt_setup_calls = mqtt_transport_mock::state.setup_calls;
  EspectreOtaStatus error;
  error.state = EspectreOtaState::ERROR;
  error.message = "download failed";
  ota.emit_status(error);

  TEST_ASSERT_FALSE(frontend.ota_frontend_quiesced_);
  TEST_ASSERT_EQUAL(mqtt_setup_calls + 1, mqtt_transport_mock::state.setup_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
}

void test_espectre_protocol_parses_config_and_rejects_bad_commands(void) {
  EspectreDeviceConfig config;
  std::string error;
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_label=Living Room", &config, &error));
  TEST_ASSERT_EQUAL_STRING("Living Room", config.device_label.c_str());
  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_id=manual", &config, &error));
  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_port=1884", &config, &error));

  EspectreCommand command;
  TEST_ASSERT_TRUE(parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":3.25}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("set_threshold", command.command.c_str());
  TEST_ASSERT_TRUE(command.has_threshold);
  TEST_ASSERT_EQUAL_FLOAT(3.25f, command.threshold);
  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command\":\"set_motion_hits\",\"motion_on_hits\":6,\"motion_off_hits\":4}", &command, &error));
  TEST_ASSERT_TRUE(command.has_motion_hits);
  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"external\"}", &command, &error));
  TEST_ASSERT_TRUE(command.has_csi_traffic_mode);
  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command\":\"set_traffic_generator_mode\",\"traffic_generator_mode\":\"dns\"}", &command, &error));
  TEST_ASSERT_TRUE(command.has_traffic_generator_mode);
  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":\"bad\"}", &command, &error));
}

void test_native_frontend_allows_sensing_when_mqtt_is_missing(void) {
  NativeFrontend frontend;
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);

  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
}

void test_native_recovery_button_requires_one_complete_long_press(void) {
  unsigned callbacks = 0U;
  RecoveryButtonService button(3000U, [&callbacks]() { ++callbacks; });

  button.update(true, 100U);
  button.update(true, 3099U);
  TEST_ASSERT_EQUAL(0U, callbacks);
  button.update(true, 3100U);
  button.update(true, 8000U);
  TEST_ASSERT_EQUAL(1U, callbacks);

  button.update(false, 8001U);
  button.update(true, UINT32_MAX - 1000U);
  button.update(true, 1999U);
  TEST_ASSERT_EQUAL(2U, callbacks);
}

void test_native_frontend_direct_service_follows_station_address_lifecycle(void) {
  MockDirectWebSocketService direct;
  NativeFrontend frontend(nullptr, nullptr, &direct);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_EQUAL(0, direct_websocket_service_mock::state.setup_calls);

  EspectreDeviceInfo info;
  info.frontend = "native";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_EQUAL(1, direct_websocket_service_mock::state.setup_calls);
  TEST_ASSERT_TRUE(direct_websocket_service_mock::state.running);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT, "/espectre/v1/ws");
  TEST_ASSERT_EQUAL(2, static_cast<int>(direct_websocket_service_mock::state.last_config.max_clients));
  TEST_ASSERT_EQUAL(8, static_cast<int>(direct_websocket_service_mock::state.last_config.outbound_queue_depth));
  TEST_ASSERT_FALSE(direct_websocket_service_mock::state.last_config.allow_missing_origin);
  const auto expected_origins = DirectWebSocketServiceConfig::for_first_party_portals().allowed_origins;
  TEST_ASSERT_EQUAL(expected_origins.size(),
                    direct_websocket_service_mock::state.last_config.allowed_origins.size());
  for (const auto &origin : expected_origins) {
    TEST_ASSERT_TRUE(std::find(direct_websocket_service_mock::state.last_config.allowed_origins.begin(),
                               direct_websocket_service_mock::state.last_config.allowed_origins.end(),
                               origin) != direct_websocket_service_mock::state.last_config.allowed_origins.end());
  }

  direct.emit_client_count(1U);
  TEST_ASSERT_EQUAL(1, static_cast<int>(frontend.direct_client_count()));
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  info.network.ip_address.clear();
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(direct_websocket_service_mock::state.shutdown_called);
  TEST_ASSERT_EQUAL(0, static_cast<int>(frontend.direct_client_count()));
}

void test_native_frontend_direct_requests_share_command_dispatch_and_return_correlated_results(void) {
  MockDirectWebSocketService direct;
  NativeFrontend frontend(nullptr, nullptr, &direct);
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  frontend.set_wifi_provisioning_info(wifi);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.firmware_version = "3.0.0-test";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  std::string saved_label;
  frontend.set_device_config_change_callback(
      [&saved_label](const EspectreDeviceConfig &updated, bool clear, std::string *message) {
        TEST_ASSERT_FALSE(clear);
        saved_label = updated.device_label;
        if (message != nullptr) {
          *message = "device config saved";
        }
        return true;
      });
  TEST_ASSERT_TRUE(frontend.setup());

  const std::string info_response = direct.emit_request(DirectWebSocketRequest{"req-info", "info", "{}"});
  TEST_ASSERT_TRUE(info_response.find("\"id\":\"req-info\"") != std::string::npos);
  TEST_ASSERT_TRUE(info_response.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_TRUE(info_response.find("3.0.0-test") != std::string::npos);

  const std::string update_response = direct.emit_request(
      DirectWebSocketRequest{"req-label", "set_device_label", "{\"device_label\":\"Kitchen\"}"});
  TEST_ASSERT_EQUAL_STRING("Kitchen", saved_label.c_str());
  TEST_ASSERT_TRUE(update_response.find("\"id\":\"req-label\"") != std::string::npos);
  TEST_ASSERT_TRUE(update_response.find("\"ok\":true") != std::string::npos);

  const std::string invalid_response =
      direct.emit_request(DirectWebSocketRequest{"req-bad", "not_supported", "{}"});
  TEST_ASSERT_TRUE(invalid_response.find("\"id\":\"req-bad\"") != std::string::npos);
  TEST_ASSERT_TRUE(invalid_response.find("\"ok\":false") != std::string::npos);
  TEST_ASSERT_TRUE(invalid_response.find("invalid_params") != std::string::npos);
}

void test_native_frontend_direct_telemetry_uses_replaceable_event_queue(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockDirectWebSocketService direct;
  NativeFrontend frontend(nullptr, nullptr, &direct);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  frontend.set_wifi_provisioning_info(wifi);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);

  frontend.on_live_telemetry(2.5f, 1.25f);
  TEST_ASSERT_EQUAL(0, static_cast<int>(direct_websocket_service_mock::state.published_events.size()));
  frontend.loop();
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_websocket_service_mock::state.published_events.size()));
  const auto &event = direct_websocket_service_mock::state.published_events.front();
  TEST_ASSERT_EQUAL_STRING("telemetry", event.event_name.c_str());
  TEST_ASSERT_TRUE(event.replaceable_telemetry);
  TEST_ASSERT_TRUE(event.data_json.find("\"movement_score\":2.5") != std::string::npos);
}

void test_native_frontend_direct_updates_bssid_and_mqtt_without_returning_secrets(void) {
  MockMqttTransport mqtt;
  MockDirectWebSocketService direct;
  NativeFrontend frontend(&mqtt, nullptr, &direct);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Ohana";
  frontend.set_wifi_provisioning_info(wifi);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  std::string provisioning_command;
  frontend.set_provisioning_command_callback(
      [&provisioning_command](const std::string &command, std::string *message) {
        provisioning_command = command;
        if (message != nullptr) {
          *message = "Wi-Fi candidate accepted";
        }
        return true;
      });
  EspectreDeviceConfig persisted;
  frontend.set_device_config_change_callback(
      [&persisted](const EspectreDeviceConfig &updated, bool clear, std::string *message) {
        TEST_ASSERT_FALSE(clear);
        persisted = updated;
        if (message != nullptr) {
          *message = "device config saved";
        }
        return true;
      });
  TEST_ASSERT_TRUE(frontend.setup());

  const std::string wifi_response = direct.emit_request(
      DirectWebSocketRequest{"wifi-1", "set_wifi_config", "{\"bssid\":\"E6:FA:C4:20:19:DE\"}"});
  TEST_ASSERT_EQUAL_STRING(
      "SET_WIFI_CONFIG:ssid=Ohana&bssid=E6%3AFA%3AC4%3A20%3A19%3ADE", provisioning_command.c_str());
  TEST_ASSERT_TRUE(wifi_response.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_TRUE(wifi_response.find("password") == std::string::npos);

  const std::string clear_wifi_response = direct.emit_request(
      DirectWebSocketRequest{"wifi-clear", "clear_wifi_config", "{}"});
  TEST_ASSERT_EQUAL_STRING("CLEAR_WIFI", provisioning_command.c_str());
  TEST_ASSERT_TRUE(clear_wifi_response.find("\"ok\":true") != std::string::npos);

  const std::string mqtt_response = direct.emit_request(DirectWebSocketRequest{
      "mqtt-1",
      "set_mqtt_config",
      "{\"host\":\"homeassistant.local\",\"port\":1883,\"username\":\"mqtt\",\"password\":\"secret\"}"});
  TEST_ASSERT_EQUAL_STRING("homeassistant.local", persisted.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(1883U, persisted.mqtt_port);
  TEST_ASSERT_EQUAL_STRING("mqtt", persisted.mqtt_username.c_str());
  TEST_ASSERT_EQUAL_STRING("secret", persisted.mqtt_password.c_str());
  TEST_ASSERT_EQUAL(1, mqtt_transport_mock::state.setup_calls);
  TEST_ASSERT_TRUE(mqtt_response.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_response.find("secret") == std::string::npos);
}

void test_native_frontend_direct_exposes_portal_reads_without_secrets(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  frontend_runtime_shim::state.diagnostics.minimum_free_memory_bytes = 2048U;
  frontend_runtime_shim::state.diagnostics.largest_free_memory_block_bytes = 1024U;
  frontend_runtime_shim::state.diagnostics.cpu_frequency_mhz = 160U;
  frontend_runtime_shim::state.diagnostics.performance_window_ready = true;
  frontend_runtime_shim::state.diagnostics.performance_window_duration_us = 10000000U;
  frontend_runtime_shim::state.diagnostics.runtime_load_percent = 7.5f;
  frontend_runtime_shim::state.diagnostics.loop_samples = 100U;
  frontend_runtime_shim::state.diagnostics.loop_average_us = 75U;
  frontend_runtime_shim::state.diagnostics.loop_maximum_us = 250U;
  frontend_runtime_shim::state.diagnostics.detection_timing_supported = true;
  frontend_runtime_shim::state.diagnostics.detection_samples = 40U;
  frontend_runtime_shim::state.diagnostics.detection_sum_us = 4000U;
  frontend_runtime_shim::state.diagnostics.detection_average_us = 100U;
  frontend_runtime_shim::state.diagnostics.detection_minimum_us = 80U;
  frontend_runtime_shim::state.diagnostics.detection_maximum_us = 140U;
  MockMqttTransport mqtt;
  MockDirectWebSocketService direct;
  direct_websocket_service_mock::state.diagnostics.accepted_connections = 4U;
  direct_websocket_service_mock::state.diagnostics.client_limit = 2U;
  direct_websocket_service_mock::state.diagnostics.queue_capacity = 8U;
  direct_websocket_service_mock::state.diagnostics.dropped_telemetry_events = 3U;
  direct_websocket_service_mock::state.diagnostics.slow_client_disconnects = 2U;
  mqtt_transport_mock::state.diagnostics.queued_publishes = 5U;
  mqtt_transport_mock::state.diagnostics.queue_capacity = 16U;
  mqtt_transport_mock::state.diagnostics.outbox_capacity_bytes = 8192U;
  mqtt_transport_mock::state.diagnostics.dropped_publishes = 6U;
  mqtt_transport_mock::state.diagnostics.publish_failures = 7U;
  mqtt_transport_mock::state.diagnostics.reconnects = 8U;
  NativeFrontend frontend(&mqtt, nullptr, &direct);

  EspectreDeviceConfig config;
  config.device_label = "Kitchen";
  config.mqtt_host = "broker.local";
  config.mqtt_port = 2883U;
  config.mqtt_username = "private-user";
  config.mqtt_password = "private-password";
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.bssid = "AA:BB:CC:DD:EE:FF";
  wifi.channel = 6U;
  wifi.has_saved_config = true;
  wifi.apply_state = "rolled_back";
  wifi.apply_message = "last-known-good configuration restored";
  frontend.set_wifi_provisioning_info(wifi);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);

  const std::string capabilities =
      direct.emit_request(DirectWebSocketRequest{"read-cap", "capabilities", "{}"});
  TEST_ASSERT_TRUE(capabilities.find("\"start_sensing\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("\"raw_csi\":false") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("mqtt_only") != std::string::npos);

  const std::string status = direct.emit_request(DirectWebSocketRequest{"read-status", "status", "{}"});
  TEST_ASSERT_TRUE(status.find("\"wifi_connected\":true") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"mqtt_configured\":true") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"sensing_enabled\":true") != std::string::npos);

  const std::string visible_config =
      direct.emit_request(DirectWebSocketRequest{"read-config", "config", "{}"});
  TEST_ASSERT_TRUE(visible_config.find("AA:BB:CC:DD:EE:FF") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("broker.local") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"apply_state\":\"rolled_back\"") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("last-known-good configuration restored") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"username_configured\":true") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("private-user") == std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("private-password") == std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"password\"") == std::string::npos);

  const std::string diagnostics =
      direct.emit_request(DirectWebSocketRequest{"read-diag", "diagnostics", "{}"});
  TEST_ASSERT_TRUE(diagnostics.find("\"clients\":1") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"minimum_free_memory_kb\":") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"largest_free_memory_kb\":1") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"cpu_frequency_mhz\":160") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"runtime_load_percent\":7.5") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"loop_avg_us\":75") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"detection_samples\":40") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"detection_max_us\":140") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"task_stack_high_water_bytes\":") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"client_limit\":2") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"queue_capacity\":8") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"accepted_connections\":4") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"dropped_telemetry_events\":3") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"slow_client_disconnects\":2") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"queued_publishes\":5") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"queue_capacity\":16") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"outbox_capacity_bytes\":8192") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"dropped_publishes\":6") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"publish_failures\":7") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"reconnects\":8") != std::string::npos);
}

void test_native_frontend_direct_start_and_stop_sensing_are_correlated(void) {
  MockDirectWebSocketService direct;
  NativeFrontend frontend(nullptr, nullptr, &direct);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());

  const std::string stopped =
      direct.emit_request(DirectWebSocketRequest{"sense-stop", "stop_sensing", "{}"});
  TEST_ASSERT_TRUE(stopped.find("\"id\":\"sense-stop\"") != std::string::npos);
  TEST_ASSERT_TRUE(stopped.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);

  const std::string started =
      direct.emit_request(DirectWebSocketRequest{"sense-start", "start_sensing", "{}"});
  TEST_ASSERT_TRUE(started.find("\"id\":\"sense-start\"") != std::string::npos);
  TEST_ASSERT_TRUE(started.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
}

void test_native_frontend_direct_ota_returns_status_and_streams_updates(void) {
  MockOtaService ota;
  MockDirectWebSocketService direct;
  ota_service_mock::state.status.default_channel = "develop";

  NativeFrontend frontend(nullptr, &ota, &direct);
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  frontend.set_device_config(config);
  EspectreDeviceInfo info;
  info.firmware_version = "1.2.3";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);

  const std::string status =
      direct.emit_request(DirectWebSocketRequest{"ota-status", "ota_status", "{}"});
  TEST_ASSERT_TRUE(status.find("\"id\":\"ota-status\"") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"current_version\":\"1.2.3\"") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"default_channel\":\"develop\"") != std::string::npos);

  const std::string check = direct.emit_request(
      DirectWebSocketRequest{"ota-check", "ota_check", "{\"channel\":\"preview\"}"});
  TEST_ASSERT_TRUE(check.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_check_calls);
  TEST_ASSERT_EQUAL_STRING("1.2.3", ota_service_mock::state.last_current_version.c_str());
  TEST_ASSERT_EQUAL_STRING("preview", ota_service_mock::state.last_channel.c_str());

  EspectreOtaStatus available;
  available.state = EspectreOtaState::UPDATE_AVAILABLE;
  available.current_version = "1.2.3";
  available.target_version = "1.3.0";
  available.update_available = true;
  ota.emit_status(available);
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_websocket_service_mock::state.published_events.size()));
  const direct_websocket_service_mock::PublishedEvent &event =
      direct_websocket_service_mock::state.published_events[0];
  TEST_ASSERT_EQUAL_STRING("ota_status", event.event_name.c_str());
  TEST_ASSERT_TRUE(event.data_json.find("\"state\":\"update_available\"") != std::string::npos);
  TEST_ASSERT_TRUE(event.data_json.find("\"target_version\":\"1.3.0\"") != std::string::npos);
  TEST_ASSERT_FALSE(event.replaceable_telemetry);

  const std::string update = direct.emit_request(
      DirectWebSocketRequest{"ota-start", "ota_start", "{\"channel\":\"preview\"}"});
  TEST_ASSERT_TRUE(update.find("\"ok\":true") != std::string::npos);
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_update_calls);
  TEST_ASSERT_EQUAL_STRING("preview", ota_service_mock::state.last_channel.c_str());
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_native_frontend_setup_registers_runtime_listener);
  RUN_TEST(test_native_frontend_setup_fails_when_runtime_setup_fails);
  RUN_TEST(test_native_frontend_loop_and_shutdown_forward_to_runtime);
  RUN_TEST(test_native_frontend_mqtt_connect_publishes_ha_discovery_and_subscribes_birth_topics);
  RUN_TEST(test_native_frontend_ha_birth_message_republishes_discovery_and_state);
  RUN_TEST(test_native_frontend_retries_the_complete_ha_snapshot_after_queue_backpressure);
  RUN_TEST(test_native_frontend_ha_entities_follow_esphome_cadences);
  RUN_TEST(test_native_frontend_ha_threshold_command_updates_runtime);
  RUN_TEST(test_native_frontend_ha_motion_hits_commands_update_runtime);
  RUN_TEST(test_native_frontend_ha_calibrate_command_triggers_runtime);
  RUN_TEST(test_native_frontend_ha_calibrate_command_respects_manual_recalibration_capability);
  RUN_TEST(test_native_frontend_ha_traffic_control_commands_update_runtime);
  RUN_TEST(test_native_frontend_ha_diagnostics_button_publishes_cached_sample);
  RUN_TEST(test_native_frontend_mqtt_connect_enables_live_telemetry);
  RUN_TEST(test_native_frontend_direct_reports_mqtt_connection_changes);
  RUN_TEST(test_native_frontend_direct_clear_mqtt_disconnects_and_reports_status);
  RUN_TEST(test_native_frontend_live_telemetry_publishes_mqtt_telemetry);
  RUN_TEST(test_native_frontend_motion_edge_publishes_ready_ha_motion);
  RUN_TEST(test_native_frontend_mqtt_set_threshold_command_publishes_result);
  RUN_TEST(test_native_frontend_mqtt_set_device_label_persists_and_republishes_info);
  RUN_TEST(test_native_frontend_mqtt_recalibrate_command_publishes_result);
  RUN_TEST(test_native_frontend_mqtt_detector_command_updates_runtime);
  RUN_TEST(test_native_frontend_mqtt_motion_hits_command_updates_runtime);
  RUN_TEST(test_native_frontend_mqtt_traffic_commands_update_runtime);
  RUN_TEST(test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads);
  RUN_TEST(test_native_frontend_mqtt_connect_publishes_current_ota_state);
  RUN_TEST(test_native_frontend_mqtt_ota_commands_use_ota_service_and_publish_state);
  RUN_TEST(test_native_frontend_ota_prepare_quiesces_transports_and_recovers_on_error);
  RUN_TEST(test_espectre_protocol_parses_config_and_rejects_bad_commands);
  RUN_TEST(test_native_frontend_allows_sensing_when_mqtt_is_missing);
  RUN_TEST(test_native_recovery_button_requires_one_complete_long_press);
  RUN_TEST(test_native_frontend_direct_service_follows_station_address_lifecycle);
  RUN_TEST(test_native_frontend_direct_requests_share_command_dispatch_and_return_correlated_results);
  RUN_TEST(test_native_frontend_direct_telemetry_uses_replaceable_event_queue);
  RUN_TEST(test_native_frontend_direct_updates_bssid_and_mqtt_without_returning_secrets);
  RUN_TEST(test_native_frontend_direct_exposes_portal_reads_without_secrets);
  RUN_TEST(test_native_frontend_direct_start_and_stop_sensing_are_correlated);
  RUN_TEST(test_native_frontend_direct_ota_returns_status_and_streams_updates);
  return UNITY_END();
}
