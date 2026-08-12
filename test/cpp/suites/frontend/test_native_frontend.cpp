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
#include <cstring>
#include <vector>

#define private public
#define protected public
#include "native_frontend.h"
#undef protected
#undef private

#include "ble_bindings_mock.h"
#include "frontend_runtime_shim.h"
#include "mqtt_transport_mock.h"
#include "ota_service_mock.h"

using namespace espectre;
using espectre::ble_bindings_mock::MockBleBindings;
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
  snapshot.detector_name = "classic";
  return snapshot;
}

float read_float_at(const std::vector<uint8_t> &payload, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, payload.data() + offset, sizeof(float));
  return value;
}

void drain_pending_sysinfo(NativeFrontend &frontend) {
  for (int i = 0; i < 4; ++i) {
    frontend.loop();
  }
}

}  // namespace

void setUp(void) {
  frontend_runtime_shim::reset();
  ble_bindings_mock::reset();
  mqtt_transport_mock::reset();
  ota_service_mock::reset();
}

void tearDown(void) {}

void test_native_frontend_setup_registers_runtime_listener_and_bindings_callbacks(void) {
  frontend_runtime_shim::state.snapshot.threshold = 3.25f;

  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.is_setup_complete());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_listener == &frontend);
  TEST_ASSERT_TRUE(static_cast<bool>(ble_bindings_mock::state.connection_callback));
  TEST_ASSERT_TRUE(static_cast<bool>(ble_bindings_mock::state.control_callback));
  TEST_ASSERT_TRUE(static_cast<bool>(ble_bindings_mock::state.telemetry_subscription_callback));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_live_telemetry_enabled_calls);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);
  TEST_ASSERT_EQUAL_FLOAT(3.25f, frontend.snapshot().threshold);
}

void test_native_frontend_setup_fails_without_bindings_or_when_transport_fails(void) {
  NativeFrontend without_bindings(nullptr);
  TEST_ASSERT_FALSE(without_bindings.setup());

  ble_bindings_mock::state.setup_result = false;
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_FALSE(frontend.setup());
}

void test_native_frontend_loop_and_shutdown_forward_to_runtime(void) {
  MockBleBindings bindings;
  {
    NativeFrontend frontend(&bindings);
    TEST_ASSERT_TRUE(frontend.setup());
    frontend.loop();
    TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.loop_calls);
  }

  TEST_ASSERT_TRUE(frontend_runtime_shim::state.shutdown_called);
  TEST_ASSERT_TRUE(ble_bindings_mock::state.shutdown_called);
}

void test_native_frontend_connection_and_sysinfo_paths(void) {
  frontend_runtime_shim::state.snapshot.motion_state = MotionState::MOTION;
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  EspectreDeviceInfo info;
  info.firmware_version = "3.0.0-test";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());

  bindings.emit_connection(true);
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(frontend.client_connected());
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());
  TEST_ASSERT_EQUAL_STRING("proto_version=1", ble_bindings_mock::state.sysinfo_lines.front().c_str());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "frontend=native") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::any_of(ble_bindings_mock::state.sysinfo_lines.begin(),
                               ble_bindings_mock::state.sysinfo_lines.end(),
                               [](const std::string &line) {
                                 return line.rfind("device_name=ESPectre ", 0) == 0;
                               }));
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "espectre_protocol_version=1.0") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "supports_runtime_motion_hits=true") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "supports_wifi_5ghz=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "wifi_band_policy=2g") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "firmware_version=3.0.0-test") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "wifi_connected=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::none_of(ble_bindings_mock::state.sysinfo_lines.begin(),
                                ble_bindings_mock::state.sysinfo_lines.end(),
                                [](const std::string &line) {
                                  return line.rfind("wifi_password_set=", 0U) == 0U;
                                }));
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "mqtt_connected=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_EQUAL_STRING("END", ble_bindings_mock::state.sysinfo_lines.back().c_str());

  info.frontend = "native";
  info.network.channel = 6;
  frontend.set_device_info(info);
  ble_bindings_mock::state.sysinfo_lines.clear();
  bindings.emit_control("REQ_SYSINFO");
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "wifi_connected=true") != ble_bindings_mock::state.sysinfo_lines.end());

  bindings.emit_connection(false);
  TEST_ASSERT_FALSE(frontend.client_connected());
}

void test_native_frontend_device_config_commands_setup_mqtt_and_publish_info_status(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  NativeFrontend frontend(&bindings, &mqtt);
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  config.device_label = "Living Room";
  std::vector<EspectreDeviceConfig> persisted_configs;
  frontend.set_device_config(config);
  frontend.set_device_config_change_callback(
      [&persisted_configs](const EspectreDeviceConfig &config, bool clear, std::string *message) {
        TEST_ASSERT_FALSE(clear);
        persisted_configs.push_back(config);
        if (message != nullptr) {
          *message = "saved";
        }
        return true;
      });
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_EQUAL(0, mqtt_transport_mock::state.setup_calls);

  bindings.emit_connection(true);
  bindings.emit_control("SET_DEVICE_CONFIG:device_label=Kitchen Sensor");
  bindings.emit_control("SET_MQTT_CONFIG:host=127.0.0.1&port=1883&username=mqtt&password=secret&topic_prefix=espectre%2Fv1%2Fdevices");

  TEST_ASSERT_EQUAL(0x0000111122223333ULL, frontend.device_config().device_id);
  TEST_ASSERT_EQUAL_STRING("Kitchen Sensor", frontend.device_config().device_label.c_str());
  TEST_ASSERT_EQUAL(2, static_cast<int>(persisted_configs.size()));
  TEST_ASSERT_EQUAL(0x0000111122223333ULL, persisted_configs.back().device_id);
  TEST_ASSERT_EQUAL_STRING("Kitchen Sensor", persisted_configs.back().device_label.c_str());
  TEST_ASSERT_EQUAL_STRING("127.0.0.1", persisted_configs.back().mqtt_host.c_str());
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.device_names.empty());
  TEST_ASSERT_TRUE(ble_bindings_mock::state.device_names.back().rfind("ESPectre ", 0) == 0);
  TEST_ASSERT_TRUE(ble_bindings_mock::state.device_names.back().find("223333") != std::string::npos);
  TEST_ASSERT_EQUAL(1, mqtt_transport_mock::state.setup_calls);
  ble_bindings_mock::state.sysinfo_lines.clear();
  mqtt.emit_connection(true);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.size() >= 2);
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000111122223333/info";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000111122223333/status";
                               }));
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "mqtt_connected=true") != ble_bindings_mock::state.sysinfo_lines.end());

  ble_bindings_mock::state.sysinfo_lines.clear();
  mqtt.emit_connection(false);
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "mqtt_connected=false") != ble_bindings_mock::state.sysinfo_lines.end());
}

void test_native_frontend_mqtt_connect_publishes_ha_discovery_and_subscribes_birth_topics(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  config.device_label = "Kitchen Sensor";
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
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
                                        "espectre/v1/devices/0x0000111122223333/ha/detector/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0x0000111122223333_motion/config" &&
                                        publish.retain;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0x0000111122223333_motion/config" &&
                                        publish.payload.find(
                                            "\"availability_topic\":\"espectre/v1/devices/0x0000111122223333/status\"") !=
                                            std::string::npos &&
                                        publish.payload.find("\"availability_template\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/motion/state" &&
                                        publish.payload == "ON";
                               }));
}

void test_native_frontend_ha_birth_message_republishes_discovery_and_state(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
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
                                            "homeassistant/binary_sensor/native_0x0000111122223333_motion/config" &&
                                        publish.retain;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/movement/state";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000111122223333/status" &&
                                        publish.payload.find("\"online\":true") != std::string::npos;
                               }));
}

void test_native_frontend_clear_device_config_forwards_to_callback_and_stops_mqtt(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  bool clear_called = false;
  frontend.set_device_config(config);
  frontend.set_device_config_change_callback([&clear_called](const EspectreDeviceConfig &config,
                                                            bool clear,
                                                            std::string *message) {
    (void) config;
    clear_called = clear;
    if (message != nullptr) {
      *message = "cleared";
    }
    return true;
  });
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_EQUAL(1, mqtt_transport_mock::state.setup_calls);

  bindings.emit_connection(true);
  bindings.emit_control("CLEAR_DEVICE_CONFIG");

  TEST_ASSERT_TRUE(clear_called);
  TEST_ASSERT_EQUAL(0x0000abcdeffedcbaULL, frontend.device_config().device_id);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.shutdown_called);
}

void test_native_frontend_clear_mqtt_config_preserves_device_identity(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.device_label = "Lab 01";
  config.mqtt_host = "localhost";
  config.mqtt_username = "mqtt";

  NativeFrontend frontend(&bindings, &mqtt);
  std::vector<EspectreDeviceConfig> persisted_configs;
  frontend.set_device_config(config);
  frontend.set_device_config_change_callback(
      [&persisted_configs](const EspectreDeviceConfig &updated, bool clear, std::string *message) {
        TEST_ASSERT_FALSE(clear);
        persisted_configs.push_back(updated);
        if (message != nullptr) {
          *message = "mqtt cleared";
        }
        return true;
      });
  TEST_ASSERT_TRUE(frontend.setup());

  bindings.emit_connection(true);
  bindings.emit_control("CLEAR_MQTT_CONFIG");

  TEST_ASSERT_EQUAL(1, static_cast<int>(persisted_configs.size()));
  TEST_ASSERT_EQUAL(0x0000abcdeffedcbaULL, frontend.device_config().device_id);
  TEST_ASSERT_EQUAL_STRING("Lab 01", frontend.device_config().device_label.c_str());
  TEST_ASSERT_EQUAL_STRING("", frontend.device_config().mqtt_host.c_str());
}

void test_native_frontend_set_mqtt_config_batch_command_updates_runtime_config(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;

  NativeFrontend frontend(&bindings, &mqtt);
  std::vector<EspectreDeviceConfig> persisted_configs;
  frontend.set_device_config(config);
  frontend.set_device_config_change_callback(
      [&persisted_configs](const EspectreDeviceConfig &updated, bool clear, std::string *message) {
        TEST_ASSERT_FALSE(clear);
        persisted_configs.push_back(updated);
        if (message != nullptr) {
          *message = "mqtt saved";
        }
        return true;
      });
  TEST_ASSERT_TRUE(frontend.setup());

  bindings.emit_connection(true);
  bindings.emit_control(
      "SET_MQTT_CONFIG:host=broker.local&port=2883&username=mqtt%20user&password=sec%40ret&topic_prefix=lab%2Fdevices");

  TEST_ASSERT_EQUAL(1, static_cast<int>(persisted_configs.size()));
  TEST_ASSERT_EQUAL_STRING("broker.local", frontend.device_config().mqtt_host.c_str());
  TEST_ASSERT_EQUAL(2883, frontend.device_config().mqtt_port);
  TEST_ASSERT_EQUAL_STRING("mqtt user", frontend.device_config().mqtt_username.c_str());
  TEST_ASSERT_EQUAL_STRING("sec@ret", frontend.device_config().mqtt_password.c_str());
  TEST_ASSERT_EQUAL_STRING("lab/devices", frontend.device_config().topic_prefix.c_str());
  TEST_ASSERT_EQUAL(1, mqtt_transport_mock::state.setup_calls);
}

void test_native_frontend_periodic_update_publishes_mqtt_telemetry(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  frontend.on_periodic_update(snapshot, 10);

  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000abcdeffedcba/telemetry" &&
                                        publish.payload.find("\"motion_state\":\"motion\"") != std::string::npos &&
                                        publish.payload.find("\"movement_score\":2.75") != std::string::npos;
                               }));
}

void test_native_frontend_motion_edge_publishes_ready_mqtt_telemetry(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.ready_to_publish = false;
  frontend.on_motion_state_changed(snapshot);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.empty());

  snapshot.ready_to_publish = true;
  frontend.on_motion_state_changed(snapshot);
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000abcdeffedcba/telemetry" &&
                                        publish.payload.find("\"motion_state\":\"motion\"") != std::string::npos;
                               }));
}

void test_native_frontend_mqtt_set_threshold_command_publishes_result(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command("{\"command_id\":\"cmd-1\",\"command\":\"set_threshold\",\"threshold\":0.45}");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.45f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_ble_and_mqtt_detector_commands_update_runtime(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  TEST_ASSERT_TRUE(frontend.handle_control_command_("SET_DETECTOR:ml"));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_detector_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_detector == DetectionAlgorithm::ML);
  TEST_ASSERT_FALSE(frontend.handle_control_command_("SET_DETECTOR:pca"));

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"command_id\":\"det-1\",\"command\":\"set_detector\",\"detector\":\"classic\"}");
  TEST_ASSERT_EQUAL(2, frontend_runtime_shim::state.set_detector_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_detector == DetectionAlgorithm::CLASSIC);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":true") !=
                   std::string::npos);
}

void test_native_frontend_ble_and_mqtt_motion_hits_commands_update_runtime(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  TEST_ASSERT_TRUE(frontend.handle_control_command_("SET_MOTION_HITS:on=6&off=4"));
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL_UINT8(6U, frontend_runtime_shim::state.last_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(4U, frontend_runtime_shim::state.last_motion_off_hits);
  TEST_ASSERT_FALSE(frontend.handle_control_command_("SET_MOTION_HITS:on=0&off=4"));

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command(
      "{\"command_id\":\"motion-1\",\"command\":\"set_motion_hits\",\"motion_on_hits\":5,\"motion_off_hits\":3}");
  TEST_ASSERT_EQUAL(2, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL_UINT8(5U, frontend_runtime_shim::state.last_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(3U, frontend_runtime_shim::state.last_motion_off_hits);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":true") !=
                   std::string::npos);
}

void test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";
  frontend_runtime_shim::state.diagnostics.traffic_packets_total = 100U;
  frontend_runtime_shim::state.diagnostics.csi_callbacks_total = 100U;
  frontend_runtime_shim::state.diagnostics.csi_accepted_total = 90U;
  frontend_runtime_shim::state.diagnostics.csi_filtered_total = 10U;

  NativeFrontend frontend(&bindings, &mqtt);
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
  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"command_id\":\"cmd-info\",\"command\":\"info\"}");
  mqtt.emit_command("{\"command_id\":\"cmd-stats\",\"command\":\"stats\"}");

  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.size() >= 4);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/info",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_runtime_motion_hits\":true") !=
                   std::string::npos);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted",
                           mqtt_transport_mock::state.publishes[1].topic.c_str());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/stats",
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
}

void test_native_frontend_mqtt_ota_commands_use_ota_service_and_publish_state(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  MockOtaService ota;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt, &ota);
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
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted",
                           mqtt_transport_mock::state.publishes.back().topic.c_str());

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
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/ota/state",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"state\":\"update_available\"") !=
                   std::string::npos);
}

void test_native_frontend_ble_ota_commands_use_ota_service_and_refresh_sysinfo(void) {
  MockBleBindings bindings;
  MockOtaService ota;
  NativeFrontend frontend(&bindings, nullptr, &ota);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.firmware_version = "1.0.0";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  drain_pending_sysinfo(frontend);

  TEST_ASSERT_TRUE(frontend.handle_control_command_("OTA_CHECK"));
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_check_calls);
  TEST_ASSERT_EQUAL_STRING("1.0.0", ota_service_mock::state.last_current_version.c_str());

  TEST_ASSERT_TRUE(frontend.handle_control_command_("OTA_START"));
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_update_calls);

  ble_bindings_mock::state.sysinfo_lines.clear();
  EspectreOtaStatus ota_status;
  ota_status.state = EspectreOtaState::UPDATE_AVAILABLE;
  ota_status.current_version = "1.0.0";
  ota_status.target_version = "1.1.0";
  ota_status.message = "update available";
  ota_status.update_available = true;
  ota.emit_status(ota_status);
  drain_pending_sysinfo(frontend);

  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "ota_state=update_available") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "ota_target_version=1.1.0") != ble_bindings_mock::state.sysinfo_lines.end());
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
  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":\"bad\"}", &command, &error));
}

void test_native_frontend_control_commands_validate_and_update_runtime(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  ble_bindings_mock::state.sysinfo_lines.clear();

  bindings.emit_control("REQ_SYSINFO");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());

  ble_bindings_mock::state.sysinfo_lines.clear();
  bindings.emit_control("SET_THRESHOLD:invalid");
  bindings.emit_control("SET_THRESHOLD:42");
  bindings.emit_control("SET_MOTION_HITS:on=6");
  bindings.emit_control("SET_MOTION_HITS:on=0&off=4");
  bindings.emit_control("UNKNOWN");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.sysinfo_lines.size()));

  bindings.emit_control("SET_THRESHOLD:0.425");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.425f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_EQUAL_FLOAT(0.425f, frontend.runtime_.config().segmentation_threshold);

  bindings.emit_control("SET_MOTION_HITS:on=7&off=5");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL_UINT8(7U, frontend_runtime_shim::state.last_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(5U, frontend_runtime_shim::state.last_motion_off_hits);
  TEST_ASSERT_EQUAL_UINT8(7U, frontend.runtime_.config().motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(5U, frontend.runtime_.config().motion_off_hits);
}

void test_native_frontend_wifi_provisioning_commands_forward_to_callback(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  std::vector<std::string> received;
  frontend.set_provisioning_command_callback([&received](const std::string &command, std::string *message) {
    received.push_back(command);
    if (message != nullptr) {
      *message = "ok";
    }
    return true;
  });
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  ble_bindings_mock::state.sysinfo_lines.clear();

  bindings.emit_control("SET_WIFI_CONFIG:ssid=Lab%20Network&password=secret&channel=6");

  TEST_ASSERT_EQUAL(1, static_cast<int>(received.size()));
  TEST_ASSERT_EQUAL_STRING("SET_WIFI_CONFIG:ssid=Lab%20Network&password=secret&channel=6", received[0].c_str());
  bindings.emit_control("REQ_SYSINFO");
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());
}

void test_native_frontend_wifi_provisioning_rejects_without_callback(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend.handle_control_command_("SET_WIFI_CONFIG:ssid=Lab&password=secret&channel=6"));
}

void test_native_frontend_live_telemetry_is_encoded_with_optional_motion_state(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());

  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  bindings.emit_connection(true);
  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(1, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));
  const auto &initial_payload = ble_bindings_mock::state.telemetry_events[0].payload;
  TEST_ASSERT_EQUAL(sizeof(float) * 2 + 1, static_cast<int>(initial_payload.size()));
  TEST_ASSERT_EQUAL_FLOAT(2.5f, read_float_at(initial_payload, 0));
  TEST_ASSERT_EQUAL_FLOAT(1.5f, read_float_at(initial_payload, sizeof(float)));
  TEST_ASSERT_EQUAL_UINT8(0, initial_payload[sizeof(float) * 2]);

  bindings.emit_telemetry_subscription(true);
  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(2, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  RuntimeSnapshot motion_snapshot = make_ready_snapshot();
  motion_snapshot.motion_state = MotionState::MOTION;
  frontend.on_motion_state_changed(motion_snapshot);
  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(3, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));
  const auto &first_payload = ble_bindings_mock::state.telemetry_events[2].payload;
  TEST_ASSERT_EQUAL(sizeof(float) * 2 + 1, static_cast<int>(first_payload.size()));
  TEST_ASSERT_EQUAL_FLOAT(2.5f, read_float_at(first_payload, 0));
  TEST_ASSERT_EQUAL_FLOAT(1.5f, read_float_at(first_payload, sizeof(float)));
  TEST_ASSERT_EQUAL_UINT8(1, first_payload[sizeof(float) * 2]);

  bindings.emit_telemetry_subscription(false);
  frontend.on_live_telemetry(3.0f, 2.0f);
  TEST_ASSERT_EQUAL(4, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  bindings.emit_telemetry_subscription(true);
  frontend.on_live_telemetry(3.0f, 2.0f);
  TEST_ASSERT_EQUAL(5, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  RuntimeSnapshot idle_snapshot = make_ready_snapshot();
  idle_snapshot.motion_state = MotionState::IDLE;
  frontend.on_motion_state_changed(idle_snapshot);
  frontend.on_live_telemetry(4.0f, 2.5f);
  TEST_ASSERT_EQUAL(6, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));
  const auto &second_payload = ble_bindings_mock::state.telemetry_events[5].payload;
  TEST_ASSERT_EQUAL_UINT8(0, second_payload[sizeof(float) * 2]);
}

void test_native_frontend_live_telemetry_subscription_toggles_runtime_callback(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());

  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);

  bindings.emit_connection(true);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);

  bindings.emit_telemetry_subscription(true);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  bindings.emit_telemetry_subscription(false);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);

  bindings.emit_connection(false);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);
}

void test_native_frontend_threshold_and_calibration_callbacks_publish_sysinfo(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  ble_bindings_mock::state.sysinfo_lines.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.threshold = 4.5f;
  frontend.on_threshold_changed(snapshot);
  TEST_ASSERT_EQUAL_FLOAT(4.5f, frontend.runtime_.config().segmentation_threshold);
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());

  ble_bindings_mock::state.sysinfo_lines.clear();
  frontend.on_calibration_started(snapshot);
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());

  ble_bindings_mock::state.sysinfo_lines.clear();
  frontend.on_calibration_finished(snapshot, false);
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());
}

void test_native_frontend_motion_state_changes_do_not_publish_sysinfo(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  drain_pending_sysinfo(frontend);
  ble_bindings_mock::state.sysinfo_lines.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.motion_state = MotionState::MOTION;
  frontend.on_motion_state_changed(snapshot);
  drain_pending_sysinfo(frontend);

  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.sysinfo_lines.size()));
}

void test_native_frontend_runtime_fault_is_reported_to_bindings(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());

  frontend.on_runtime_fault("wifi disconnected");
  TEST_ASSERT_EQUAL(1, static_cast<int>(ble_bindings_mock::state.faults.size()));
  TEST_ASSERT_EQUAL_STRING("wifi disconnected", ble_bindings_mock::state.faults[0].c_str());
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_native_frontend_setup_registers_runtime_listener_and_bindings_callbacks);
  RUN_TEST(test_native_frontend_setup_fails_without_bindings_or_when_transport_fails);
  RUN_TEST(test_native_frontend_loop_and_shutdown_forward_to_runtime);
  RUN_TEST(test_native_frontend_connection_and_sysinfo_paths);
  RUN_TEST(test_native_frontend_device_config_commands_setup_mqtt_and_publish_info_status);
  RUN_TEST(test_native_frontend_mqtt_connect_publishes_ha_discovery_and_subscribes_birth_topics);
  RUN_TEST(test_native_frontend_ha_birth_message_republishes_discovery_and_state);
  RUN_TEST(test_native_frontend_clear_device_config_forwards_to_callback_and_stops_mqtt);
  RUN_TEST(test_native_frontend_clear_mqtt_config_preserves_device_identity);
  RUN_TEST(test_native_frontend_set_mqtt_config_batch_command_updates_runtime_config);
  RUN_TEST(test_native_frontend_periodic_update_publishes_mqtt_telemetry);
  RUN_TEST(test_native_frontend_motion_edge_publishes_ready_mqtt_telemetry);
  RUN_TEST(test_native_frontend_mqtt_set_threshold_command_publishes_result);
  RUN_TEST(test_native_frontend_ble_and_mqtt_detector_commands_update_runtime);
  RUN_TEST(test_native_frontend_ble_and_mqtt_motion_hits_commands_update_runtime);
  RUN_TEST(test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads);
  RUN_TEST(test_native_frontend_mqtt_ota_commands_use_ota_service_and_publish_state);
  RUN_TEST(test_native_frontend_ble_ota_commands_use_ota_service_and_refresh_sysinfo);
  RUN_TEST(test_espectre_protocol_parses_config_and_rejects_bad_commands);
  RUN_TEST(test_native_frontend_control_commands_validate_and_update_runtime);
  RUN_TEST(test_native_frontend_wifi_provisioning_commands_forward_to_callback);
  RUN_TEST(test_native_frontend_wifi_provisioning_rejects_without_callback);
  RUN_TEST(test_native_frontend_live_telemetry_is_encoded_with_optional_motion_state);
  RUN_TEST(test_native_frontend_live_telemetry_subscription_toggles_runtime_callback);
  RUN_TEST(test_native_frontend_threshold_and_calibration_callbacks_publish_sysinfo);
  RUN_TEST(test_native_frontend_motion_state_changes_do_not_publish_sysinfo);
  RUN_TEST(test_native_frontend_runtime_fault_is_reported_to_bindings);
  return UNITY_END();
}
