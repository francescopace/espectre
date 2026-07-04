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

using namespace esphome::espectre;
using esphome::espectre::ble_bindings_mock::MockBleBindings;
using esphome::espectre::mqtt_transport_mock::MockMqttTransport;
using esphome::espectre::ota_service_mock::MockOtaService;

namespace {

RuntimeSnapshot make_ready_snapshot() {
  RuntimeSnapshot snapshot{};
  snapshot.ready_to_publish = true;
  snapshot.motion_state = MotionState::MOTION;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.best_pxx = 0.42f;
  snapshot.detector_name = "mvs";
  return snapshot;
}

float read_float_at(const std::vector<uint8_t> &payload, size_t offset) {
  float value = 0.0f;
  std::memcpy(&value, payload.data() + offset, sizeof(float));
  return value;
}

void drain_pending_sysinfo(NativeFrontend &frontend) {
  for (int i = 0; i < 64 && (!frontend.pending_sysinfo_lines_.empty() || frontend.next_sysinfo_line_index_ != 0); ++i) {
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
                             "wifi_connected=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "mqtt_connected=false") != ble_bindings_mock::state.sysinfo_lines.end());
  TEST_ASSERT_EQUAL_STRING("END", ble_bindings_mock::state.sysinfo_lines.back().c_str());

  EspectreDeviceInfo info;
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
  bindings.emit_control("SET_DEVICE_CONFIG:mqtt_host=127.0.0.1");

  TEST_ASSERT_EQUAL(0x0000111122223333ULL, frontend.device_config().device_id);
  TEST_ASSERT_EQUAL_STRING("Kitchen Sensor", frontend.device_config().device_label.c_str());
  TEST_ASSERT_EQUAL(2, static_cast<int>(persisted_configs.size()));
  TEST_ASSERT_EQUAL(0x0000111122223333ULL, persisted_configs.back().device_id);
  TEST_ASSERT_EQUAL_STRING("Kitchen Sensor", persisted_configs.back().device_label.c_str());
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.device_names.empty());
  TEST_ASSERT_TRUE(ble_bindings_mock::state.device_names.back().rfind("ESPectre ", 0) == 0);
  TEST_ASSERT_TRUE(ble_bindings_mock::state.device_names.back().find("223333") != std::string::npos);
  TEST_ASSERT_EQUAL(1, mqtt_transport_mock::state.setup_calls);
  mqtt.emit_connection(true);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.size() >= 2);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000111122223333/info",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000111122223333/status",
                           mqtt_transport_mock::state.publishes[1].topic.c_str());
  ble_bindings_mock::state.sysinfo_lines.clear();
  bindings.emit_control("REQ_SYSINFO");
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(std::find(ble_bindings_mock::state.sysinfo_lines.begin(),
                             ble_bindings_mock::state.sysinfo_lines.end(),
                             "mqtt_connected=true") != ble_bindings_mock::state.sysinfo_lines.end());
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

  TEST_ASSERT_EQUAL(1, static_cast<int>(mqtt_transport_mock::state.publishes.size()));
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/telemetry",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"motion_state\":\"motion\"") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"movement_score\":2.75") !=
                   std::string::npos);
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

  mqtt.emit_command("{\"command_id\":\"cmd-1\",\"command\":\"set_threshold\",\"threshold\":4.5}");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(4.5f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads(void) {
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
  frontend.on_motion_state_changed(snapshot);
  mqtt.emit_command("{\"command_id\":\"cmd-info\",\"command\":\"info\"}");
  mqtt.emit_command("{\"command_id\":\"cmd-stats\",\"command\":\"stats\"}");

  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.size() >= 4);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/info",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted",
                           mqtt_transport_mock::state.publishes[1].topic.c_str());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/stats",
                           mqtt_transport_mock::state.publishes[2].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"uptime\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"free_memory_kb\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"loop_time_ms\":") != std::string::npos);
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

  mqtt.emit_command("{\"command_id\":\"cmd-ota-check\",\"command\":\"ota_check\",\"manifest_url\":\"https://fw.example/manifest.json\"}");
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_check_calls);
  TEST_ASSERT_EQUAL_STRING("https://fw.example/manifest.json", ota_service_mock::state.last_manifest_url.c_str());
  TEST_ASSERT_EQUAL_STRING("1.0.0", ota_service_mock::state.last_current_version.c_str());
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0x0000abcdeffedcba/commands/accepted",
                           mqtt_transport_mock::state.publishes.back().topic.c_str());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"command_id\":\"cmd-ota-start\",\"command\":\"ota_start\",\"image_url\":\"https://fw.example/native.bin\",\"version\":\"1.1.0\"}");
  TEST_ASSERT_EQUAL(1, ota_service_mock::state.start_update_calls);
  TEST_ASSERT_EQUAL_STRING("https://fw.example/native.bin", ota_service_mock::state.last_image_url.c_str());
  TEST_ASSERT_EQUAL_STRING("1.1.0", ota_service_mock::state.last_target_version.c_str());

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

void test_espectre_protocol_parses_config_and_rejects_bad_commands(void) {
  EspectreDeviceConfig config;
  std::string error;
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_label=Living Room", &config, &error));
  TEST_ASSERT_EQUAL_STRING("Living Room", config.device_label.c_str());
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_port=1884", &config, &error));
  TEST_ASSERT_EQUAL(1884, config.mqtt_port);
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_username=mqtt", &config, &error));
  TEST_ASSERT_EQUAL_STRING("mqtt", config.mqtt_username.c_str());
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_password=secret", &config, &error));
  TEST_ASSERT_EQUAL_STRING("secret", config.mqtt_password.c_str());
  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_id=manual", &config, &error));
  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_port=0", &config, &error));

  EspectreCommand command;
  TEST_ASSERT_TRUE(parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":3.25}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("set_threshold", command.command.c_str());
  TEST_ASSERT_TRUE(command.has_threshold);
  TEST_ASSERT_EQUAL_FLOAT(3.25f, command.threshold);
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
  bindings.emit_control("UNKNOWN");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL(0, static_cast<int>(ble_bindings_mock::state.sysinfo_lines.size()));

  bindings.emit_control("SET_THRESHOLD:4.25");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(4.25f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(frontend.runtime_.config().threshold_mode == ThresholdMode::MANUAL);
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

  bindings.emit_control("SET_WIFI_SSID:Lab Network");
  bindings.emit_control("SET_WIFI_PASSWORD:secret");
  bindings.emit_control("SET_WIFI_CHANNEL:6");
  bindings.emit_control("APPLY_WIFI");

  TEST_ASSERT_EQUAL(4, static_cast<int>(received.size()));
  TEST_ASSERT_EQUAL_STRING("SET_WIFI_SSID:Lab Network", received[0].c_str());
  TEST_ASSERT_EQUAL_STRING("SET_WIFI_PASSWORD:secret", received[1].c_str());
  TEST_ASSERT_EQUAL_STRING("SET_WIFI_CHANNEL:6", received[2].c_str());
  TEST_ASSERT_EQUAL_STRING("APPLY_WIFI", received[3].c_str());
  drain_pending_sysinfo(frontend);
  TEST_ASSERT_TRUE(!ble_bindings_mock::state.sysinfo_lines.empty());
}

void test_native_frontend_wifi_provisioning_rejects_without_callback(void) {
  MockBleBindings bindings;
  NativeFrontend frontend(&bindings);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_FALSE(frontend.handle_control_command_("SET_WIFI_SSID:Lab"));
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
  RUN_TEST(test_native_frontend_clear_device_config_forwards_to_callback_and_stops_mqtt);
  RUN_TEST(test_native_frontend_clear_mqtt_config_preserves_device_identity);
  RUN_TEST(test_native_frontend_periodic_update_publishes_mqtt_telemetry);
  RUN_TEST(test_native_frontend_mqtt_set_threshold_command_publishes_result);
  RUN_TEST(test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads);
  RUN_TEST(test_native_frontend_mqtt_ota_commands_use_ota_service_and_publish_state);
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
