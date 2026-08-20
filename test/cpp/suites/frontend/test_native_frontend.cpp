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
#include "ble_recovery_button_service.h"
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
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.setup_calls);
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
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
                             "supports_runtime_threshold=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "supports_runtime_motion_hits=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "supports_runtime_detector=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "supports_manual_recalibration=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "supports_traffic_control=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "supports_live_telemetry=false") != ble_bindings_mock::state.sysinfo_lines.end());
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
                                 return publish.topic == "espectre/v1/devices/0x0000111122223333/info" &&
                                        publish.retain;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000111122223333/status" &&
                                        publish.payload.find("\"online\":true") != std::string::npos && publish.retain;
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
                                        "espectre/v1/devices/0x0000111122223333/ha/threshold/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0x0000111122223333/ha/motion_on_hits/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0x0000111122223333/ha/motion_off_hits/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0x0000111122223333/ha/calibrate/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0x0000111122223333/ha/detector/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0x0000111122223333/ha/csi_traffic_mode/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0x0000111122223333/ha/traffic_generator_mode/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.subscriptions.begin(),
                               mqtt_transport_mock::state.subscriptions.end(),
                               [](const mqtt_transport_mock::Subscription &subscription) {
                                 return subscription.topic ==
                                        "espectre/v1/devices/0x0000111122223333/ha/diagnostics/set";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0x0000111122223333_motion_detected/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Motion Detected\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/sensor/native_0x0000111122223333_movement_score/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Movement Score\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/sensor/native_0x0000111122223333_intensity/config" &&
                                        publish.retain && publish.payload.empty();
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0x0000111122223333_motion/config" &&
                                        publish.retain && publish.payload.empty();
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/switch/native_0x0000111122223333_calibrate/config" &&
                                        publish.retain && publish.payload.empty();
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/sensor/native_0x0000111122223333_traffic_tx_rate/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Traffic TX Rate\"") != std::string::npos &&
                                        publish.payload.find("\"entity_category\":\"diagnostic\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/button/native_0x0000111122223333_refresh_diagnostics/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Refresh Diagnostics\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/number/native_0x0000111122223333_threshold/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Threshold\"") != std::string::npos &&
                                        publish.payload.find("\"command_topic\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/number/native_0x0000111122223333_motion_on_hits/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Motion On Hits\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/number/native_0x0000111122223333_motion_off_hits/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Motion Off Hits\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/switch/native_0x0000111122223333_trigger_calibration/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"Trigger Calibration\"") != std::string::npos &&
                                        publish.payload.find("\"command_topic\"") != std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/select/native_0x0000111122223333_detection_profile/config" &&
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
                                            "homeassistant/select/native_0x0000111122223333_csi_traffic_ownership/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"CSI Traffic Ownership\"") !=
                                            std::string::npos;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/select/native_0x0000111122223333_csi_traffic_source/config" &&
                                        publish.retain &&
                                        publish.payload.find("\"name\":\"CSI Traffic Source\"") !=
                                            std::string::npos;
                               }));
  const int csi_traffic_discovery = mqtt_publish_index(
      "homeassistant/select/native_0x0000111122223333_csi_traffic_ownership/config");
  const int traffic_generator_discovery = mqtt_publish_index(
      "homeassistant/select/native_0x0000111122223333_csi_traffic_source/config");
  const int calibrate_discovery =
      mqtt_publish_index("homeassistant/switch/native_0x0000111122223333_trigger_calibration/config");
  TEST_ASSERT_TRUE(csi_traffic_discovery >= 0);
  TEST_ASSERT_TRUE(csi_traffic_discovery < traffic_generator_discovery);
  TEST_ASSERT_TRUE(traffic_generator_discovery < calibrate_discovery);
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "homeassistant/binary_sensor/native_0x0000111122223333_motion_detected/config" &&
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
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000111122223333/ha/motion_on_hits/state", "4"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000111122223333/ha/motion_off_hits/state", "3"));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/movement/state" &&
                                        publish.payload == "2.7500";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/threshold/state" &&
                                        publish.payload == "1.5000";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/calibrate/state" &&
                                        publish.payload == "OFF";
                               }));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000111122223333/ha/csi_traffic_mode/state", "internal"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000111122223333/ha/traffic_generator_mode/state", "ping"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000111122223333/ha/traffic_tx_rate/state"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.shutdown();
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000111122223333/status" &&
                                        publish.payload.find("\"online\":false") != std::string::npos && publish.retain;
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
                                            "homeassistant/binary_sensor/native_0x0000111122223333_motion_detected/config" &&
                                        publish.retain;
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/movement/state";
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
                                            "homeassistant/sensor/native_0x0000111122223333_traffic_tx_rate/config" &&
                                        publish.retain;
                               }));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000111122223333/ha/traffic_tx_rate/state"));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/threshold/state";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic ==
                                            "espectre/v1/devices/0x0000111122223333/ha/calibrate/state";
                               }));
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000111122223333/status" &&
                                        publish.payload.find("\"online\":true") != std::string::npos && publish.retain;
                               }));
}

void test_native_frontend_ha_entities_follow_esphome_cadences(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  mqtt_transport_mock::state.publishes.clear();
  RuntimeSnapshot snapshot = make_ready_snapshot();
  frontend.on_live_telemetry(snapshot.movement_metric, snapshot.threshold);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/movement/state", "2.7500"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/telemetry"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/motion/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/threshold/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/intensity/state"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.on_periodic_update(snapshot, 10);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/movement/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/telemetry"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/intensity/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/motion/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/threshold/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.on_motion_state_changed(snapshot);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/motion/state", "ON"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/telemetry"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/movement/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/intensity/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/threshold/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state"));

  mqtt_transport_mock::state.publishes.clear();
  snapshot.threshold = 0.45f;
  frontend.on_threshold_changed(snapshot);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/threshold/state", "0.4500"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/intensity/state"));
}

void test_native_frontend_mqtt_connect_enables_live_telemetry_without_ble(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);

  mqtt.emit_connection(true);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  mqtt.emit_connection(false);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.live_telemetry_enabled);
}

void test_native_frontend_ha_threshold_command_updates_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/threshold/set", "0.45");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.45f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/threshold/state", "0.4500"));
}

void test_native_frontend_ha_motion_hits_commands_update_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/motion_on_hits/set", "6");
  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/motion_off_hits/set", "4");

  TEST_ASSERT_EQUAL(2, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL_UINT8(6U, frontend_runtime_shim::state.last_motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(4U, frontend_runtime_shim::state.last_motion_off_hits);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/motion_on_hits/state", "6"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/motion_off_hits/state", "4"));
}

void test_native_frontend_ha_calibrate_command_triggers_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/set", "ON");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state", "ON"));

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/set", "OFF");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state", "ON"));

  mqtt_transport_mock::state.publishes.clear();
  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.calibrating = true;
  frontend.on_calibration_started(snapshot);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state", "ON"));

  mqtt_transport_mock::state.publishes.clear();
  snapshot.calibrating = false;
  snapshot.threshold = 0.42f;
  frontend.on_calibration_finished(snapshot, true);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state", "OFF"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/threshold/state", "0.4200"));
}

void test_native_frontend_ha_calibrate_command_respects_manual_recalibration_capability(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  frontend_runtime_shim::state.capabilities.supports_manual_recalibration = false;
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/set", "ON");

  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/calibrate/state", "OFF"));
}

void test_native_frontend_ha_traffic_control_commands_update_runtime(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_traffic_mode/set", "external");
  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/traffic_generator_mode/set", "dns");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_csi_traffic_mode == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_traffic_generator_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_traffic_generator_mode == RuntimeTrafficMode::DNS);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_traffic_mode/state", "external"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/traffic_generator_mode/state", "dns"));

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_traffic_mode/set", "pacing");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_csi_traffic_mode == CsiTrafficMode::EXTERNAL);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_traffic_mode/state", "pacing"));
}

void test_native_frontend_ha_diagnostics_button_publishes_cached_sample(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
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

  mqtt.emit_message("espectre/v1/devices/0x0000abcdeffedcba/ha/diagnostics/set", "PRESS");

  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/traffic_tx_rate/state", "100.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_callback_rate/state", "96.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_accepted_rate/state", "90.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_admitted_rate/state", "84.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_filtered_rate/state", "6.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_missing_rate/state", "10.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_excess_rate/state", "6.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_stale_rate/state", "0.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_out_of_order_rate/state", "0.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/csi_occupancy/state", "84.0"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/wifi_channel/state", "10"));
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/wifi_rssi/state", "-55"));
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

void test_native_frontend_live_telemetry_publishes_mqtt_telemetry(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  frontend.on_periodic_update(snapshot, 10);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/telemetry"));

  mqtt_transport_mock::state.publishes.clear();
  frontend.on_live_telemetry(snapshot.movement_metric, snapshot.threshold);
  TEST_ASSERT_TRUE(std::any_of(mqtt_transport_mock::state.publishes.begin(),
                               mqtt_transport_mock::state.publishes.end(),
                               [](const mqtt_transport_mock::Publish &publish) {
                                 return publish.topic == "espectre/v1/devices/0x0000abcdeffedcba/telemetry" &&
                                        publish.payload.find("\"motion_state\":\"motion\"") != std::string::npos &&
                                        publish.payload.find("\"movement_score\":2.75") != std::string::npos;
                               }));
}

void test_native_frontend_motion_edge_publishes_ready_ha_motion(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  snapshot.ready_to_publish = false;
  frontend.on_motion_state_changed(snapshot);
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/motion/state"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/telemetry"));

  snapshot.ready_to_publish = true;
  frontend.on_motion_state_changed(snapshot);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/ha/motion/state", "ON"));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0x0000abcdeffedcba/telemetry"));
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

void test_native_frontend_mqtt_recalibrate_command_publishes_result(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command("{\"command_id\":\"cmd-recal\",\"command\":\"recalibrate\"}");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_detector_command_updates_runtime(void) {
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

  TEST_ASSERT_FALSE(frontend.handle_control_command_("SET_DETECTOR:high_accuracy"));
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_detector_calls);

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

  TEST_ASSERT_FALSE(frontend.handle_control_command_("SET_MOTION_HITS:on=6&off=4"));
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_motion_hits_calls);

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

  TEST_ASSERT_FALSE(frontend.handle_control_command_("SET_CSI_TRAFFIC_MODE:external"));
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_csi_traffic_mode_calls);

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
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].retain);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_runtime_motion_hits\":true") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_manual_recalibration\":true") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_traffic_control\":true") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"supports_ble\":true") !=
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

  mqtt.emit_command("{\"command_id\":\"cmd-commands\",\"command\":\"commands\"}");
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.size() >= 6);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/catalog",
                           mqtt_transport_mock::state.publishes[4].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[4].payload.find("\"commands\":[") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[4].payload.find("\"set_ble\"") != std::string::npos);
}

void test_native_frontend_mqtt_connect_publishes_current_ota_state(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  MockOtaService ota;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";

  NativeFrontend frontend(&bindings, &mqtt, &ota);
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
                                 return publish.topic == "espectre/v1/devices/0x0000abcdeffedcba/ota/state" &&
                                        publish.payload.find("\"state\":\"idle\"") != std::string::npos &&
                                        publish.payload.find("\"current_version\":\"1.2.3\"") != std::string::npos &&
                                        !publish.retain;
                               }));
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
  TEST_ASSERT_TRUE(ota_service_mock::state.last_channel.empty());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted",
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
  TEST_ASSERT_TRUE(ota_service_mock::state.last_channel.empty());

  TEST_ASSERT_TRUE(frontend.handle_control_command_("OTA_START:channel=develop"));
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_update_calls);
  TEST_ASSERT_EQUAL_STRING("develop", ota_service_mock::state.last_channel.c_str());

  TEST_ASSERT_FALSE(frontend.handle_control_command_("OTA_CHECK:channel=latest"));

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
  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"external\"}", &command, &error));
  TEST_ASSERT_TRUE(command.has_csi_traffic_mode);
  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command\":\"set_traffic_generator_mode\",\"traffic_generator_mode\":\"dns\"}", &command, &error));
  TEST_ASSERT_TRUE(command.has_traffic_generator_mode);
  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":\"bad\"}", &command, &error));
}

void test_native_frontend_ble_sensing_commands_are_rejected(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);
  ble_bindings_mock::state.sysinfo_lines.clear();

  bindings.emit_control("REQ_SYSINFO");
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());

  ble_bindings_mock::state.sysinfo_lines.clear();
  bindings.emit_control("SET_THRESHOLD:0.425");
  bindings.emit_control("SET_MOTION_HITS:on=7&off=5");
  bindings.emit_control("SET_DETECTOR:high_accuracy");
  bindings.emit_control("SET_CSI_TRAFFIC_MODE:external");
  bindings.emit_control("SET_TRAFFIC_GENERATOR_MODE:dns");
  bindings.emit_control("RECALIBRATE");
  bindings.emit_control("UNKNOWN");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_motion_hits_calls);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_detector_calls);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_traffic_generator_mode_calls);
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.sysinfo_lines.size()));
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

void test_native_frontend_does_not_publish_ble_live_telemetry(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());

  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));

  bindings.emit_connection(true);
  bindings.emit_telemetry_subscription(true);
  frontend.on_live_telemetry(2.5f, 1.5f);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.telemetry_events.size()));
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

void test_native_frontend_skips_ble_when_wifi_and_mqtt_are_configured(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);

  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_EQUAL(0, ble_bindings_mock::state.setup_calls);
  TEST_ASSERT_FALSE(frontend.ble_active());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
}

void test_native_frontend_skips_ble_when_kconfig_wifi_and_mqtt_are_present(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  NativeFrontend frontend(&bindings, &mqtt);
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = false;
  frontend.set_wifi_provisioning_info(wifi);

  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_EQUAL(0, ble_bindings_mock::state.setup_calls);
  TEST_ASSERT_FALSE(frontend.ble_active());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);

  mqtt.emit_command("{\"command_id\":\"ble-kconfig-1\",\"command\":\"set_ble\",\"ble\":\"on\"}");
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.setup_calls);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);

  mqtt.emit_command("{\"command_id\":\"ble-kconfig-2\",\"command\":\"set_ble\",\"ble\":\"off\"}");
  frontend.loop();
  TEST_ASSERT_FALSE(frontend.ble_active());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
}

void test_native_frontend_keeps_ble_when_wifi_is_saved_but_mqtt_is_missing(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);

  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_FALSE(frontend.handle_control_command_("STOP_BLE"));
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
}

void test_native_frontend_keeps_ble_when_mqtt_is_saved_but_wifi_is_missing(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  frontend.set_device_config(config);

  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend.handle_control_command_("STOP_BLE"));
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
}

void test_native_frontend_stop_ble_is_rejected_until_wifi_is_configured(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend.handle_control_command_("STOP_BLE"));
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_EQUAL(0, ble_bindings_mock::state.shutdown_calls);
}

void test_native_frontend_keeps_ble_advertising_after_disconnect(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  bindings.emit_connection(true);

  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(0, ble_bindings_mock::state.shutdown_calls);

  bindings.emit_connection(false);
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend.client_connected());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(0, ble_bindings_mock::state.shutdown_calls);

  frontend.set_wifi_provisioning_info(wifi);
  frontend.set_device_config(config);
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(0, ble_bindings_mock::state.shutdown_calls);

  TEST_ASSERT_TRUE(frontend.handle_control_command_("STOP_BLE"));
  frontend.loop();
  TEST_ASSERT_FALSE(frontend.ble_active());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.shutdown_calls);
}

void test_native_frontend_mqtt_set_ble_starts_and_stop_ble_stops(void) {
  MockBleBindings bindings;
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_host = "localhost";
  NativeFrontend frontend(&bindings, &mqtt);
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend.ble_active());

  mqtt.emit_command("{\"command_id\":\"ble-1\",\"command\":\"set_ble\",\"ble\":\"on\"}");
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.setup_calls);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":true") != std::string::npos);

  TEST_ASSERT_TRUE(frontend.handle_control_command_("STOP_BLE"));
  frontend.loop();
  TEST_ASSERT_FALSE(frontend.ble_active());
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.shutdown_calls);

  mqtt.emit_command("{\"command_id\":\"ble-2\",\"command\":\"set_ble\",\"ble\":\"off\"}");
  frontend.loop();
  TEST_ASSERT_FALSE(frontend.ble_active());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"command_id\":\"ble-3\",\"command\":\"set_ble\"}");
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const std::string &reject = mqtt_transport_mock::state.publishes.back().payload;
  TEST_ASSERT_TRUE(reject.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_TRUE(reject.find("\"command_id\":\"ble-3\"") != std::string::npos);
  TEST_ASSERT_TRUE(reject.find("\"command\":\"set_ble\"") != std::string::npos);
  TEST_ASSERT_TRUE(reject.find("invalid ble mode") != std::string::npos);
}

void test_native_frontend_clearing_wifi_starts_ble(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend.ble_active());

  wifi.ssid.clear();
  wifi.has_saved_config = false;
  frontend.set_wifi_provisioning_info(wifi);
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.setup_calls);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
}

void test_native_frontend_clearing_mqtt_starts_ble(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend.ble_active());

  config.mqtt_host.clear();
  frontend.set_device_config(config);
  frontend.loop();
  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.setup_calls);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
}

void test_native_ble_recovery_button_requires_one_complete_long_press(void) {
  unsigned callbacks = 0U;
  BleRecoveryButtonService button(3000U, [&callbacks]() { ++callbacks; });

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

void test_native_frontend_physical_recovery_starts_ble_and_pauses_sensing(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend.ble_active());

  frontend.request_ble_recovery();
  frontend.loop();

  TEST_ASSERT_TRUE(frontend.ble_active());
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);
  TEST_ASSERT_EQUAL(1, ble_bindings_mock::state.setup_calls);
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
  RUN_TEST(test_native_frontend_ha_entities_follow_esphome_cadences);
  RUN_TEST(test_native_frontend_ha_threshold_command_updates_runtime);
  RUN_TEST(test_native_frontend_ha_motion_hits_commands_update_runtime);
  RUN_TEST(test_native_frontend_ha_calibrate_command_triggers_runtime);
  RUN_TEST(test_native_frontend_ha_calibrate_command_respects_manual_recalibration_capability);
  RUN_TEST(test_native_frontend_ha_traffic_control_commands_update_runtime);
  RUN_TEST(test_native_frontend_ha_diagnostics_button_publishes_cached_sample);
  RUN_TEST(test_native_frontend_mqtt_connect_enables_live_telemetry_without_ble);
  RUN_TEST(test_native_frontend_clear_device_config_forwards_to_callback_and_stops_mqtt);
  RUN_TEST(test_native_frontend_clear_mqtt_config_preserves_device_identity);
  RUN_TEST(test_native_frontend_set_mqtt_config_batch_command_updates_runtime_config);
  RUN_TEST(test_native_frontend_live_telemetry_publishes_mqtt_telemetry);
  RUN_TEST(test_native_frontend_motion_edge_publishes_ready_ha_motion);
  RUN_TEST(test_native_frontend_mqtt_set_threshold_command_publishes_result);
  RUN_TEST(test_native_frontend_mqtt_recalibrate_command_publishes_result);
  RUN_TEST(test_native_frontend_mqtt_detector_command_updates_runtime);
  RUN_TEST(test_native_frontend_mqtt_motion_hits_command_updates_runtime);
  RUN_TEST(test_native_frontend_mqtt_traffic_commands_update_runtime);
  RUN_TEST(test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads);
  RUN_TEST(test_native_frontend_mqtt_connect_publishes_current_ota_state);
  RUN_TEST(test_native_frontend_mqtt_ota_commands_use_ota_service_and_publish_state);
  RUN_TEST(test_native_frontend_ble_ota_commands_use_ota_service_and_refresh_sysinfo);
  RUN_TEST(test_espectre_protocol_parses_config_and_rejects_bad_commands);
  RUN_TEST(test_native_frontend_ble_sensing_commands_are_rejected);
  RUN_TEST(test_native_frontend_wifi_provisioning_commands_forward_to_callback);
  RUN_TEST(test_native_frontend_wifi_provisioning_rejects_without_callback);
  RUN_TEST(test_native_frontend_does_not_publish_ble_live_telemetry);
  RUN_TEST(test_native_frontend_threshold_and_calibration_callbacks_publish_sysinfo);
  RUN_TEST(test_native_frontend_motion_state_changes_do_not_publish_sysinfo);
  RUN_TEST(test_native_frontend_runtime_fault_is_reported_to_bindings);
  RUN_TEST(test_native_frontend_skips_ble_when_wifi_and_mqtt_are_configured);
  RUN_TEST(test_native_frontend_skips_ble_when_kconfig_wifi_and_mqtt_are_present);
  RUN_TEST(test_native_frontend_keeps_ble_when_wifi_is_saved_but_mqtt_is_missing);
  RUN_TEST(test_native_frontend_keeps_ble_when_mqtt_is_saved_but_wifi_is_missing);
  RUN_TEST(test_native_frontend_stop_ble_is_rejected_until_wifi_is_configured);
  RUN_TEST(test_native_frontend_keeps_ble_advertising_after_disconnect);
  RUN_TEST(test_native_frontend_mqtt_set_ble_starts_and_stop_ble_stops);
  RUN_TEST(test_native_frontend_clearing_wifi_starts_ble);
  RUN_TEST(test_native_frontend_clearing_mqtt_starts_ble);
  RUN_TEST(test_native_ble_recovery_button_requires_one_complete_long_press);
  RUN_TEST(test_native_frontend_physical_recovery_starts_ble_and_pauses_sensing);
  return UNITY_END();
}
