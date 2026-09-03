/*
 * ESPectre - Native MQTT Frontend Tests
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "native_frontend_test_support.h"

void test_native_frontend_mqtt_connect_enables_live_telemetry(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

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
  MockDirectHttpService direct;
  EspectreDeviceConfig config;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt, nullptr, &direct);
  frontend.set_device_config(config);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);

  mqtt.emit_connection(true);
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_http_service_mock::state.published_events.size()));
  TEST_ASSERT_EQUAL_STRING(
      "status", direct_http_service_mock::state.published_events[0].event_name.c_str());
  TEST_ASSERT_TRUE(
      direct_http_service_mock::state.published_events[0].data_json.find("\"mqtt_connected\":true") !=
      std::string::npos);

  direct_http_service_mock::state.published_events.clear();
  mqtt.emit_connection(false);
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_http_service_mock::state.published_events.size()));
  TEST_ASSERT_EQUAL_STRING(
      "status", direct_http_service_mock::state.published_events[0].event_name.c_str());
  TEST_ASSERT_TRUE(
      direct_http_service_mock::state.published_events[0].data_json.find("\"mqtt_connected\":false") !=
      std::string::npos);
}

void test_native_frontend_direct_clear_mqtt_disconnects_and_reports_status(void) {
  MockMqttTransport mqtt;
  MockDirectHttpService direct;
  EspectreDeviceConfig config;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt, nullptr, &direct);
  frontend.set_device_config(config);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);
  mqtt.emit_connection(true);
  direct_http_service_mock::state.published_events.clear();

  const std::string response = direct.emit_request(
      DirectRequest{"clear-mqtt", "clear_mqtt_config", "{}"});

  TEST_ASSERT_TRUE(response.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.shutdown_called);
  TEST_ASSERT_EQUAL(2, static_cast<int>(direct_http_service_mock::state.published_events.size()));
  TEST_ASSERT_EQUAL_STRING(
      "status", direct_http_service_mock::state.published_events[0].event_name.c_str());
  TEST_ASSERT_TRUE(
      direct_http_service_mock::state.published_events[0].data_json.find("\"mqtt_connected\":false") !=
      std::string::npos);
  TEST_ASSERT_TRUE(
      direct_http_service_mock::state.published_events[0].data_json.find("\"mqtt_configured\":false") !=
      std::string::npos);

  const std::string status =
      direct.emit_request(DirectRequest{"read-status", "status", "{}"});
  TEST_ASSERT_TRUE(status.find("\"mqtt_connected\":false") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"mqtt_configured\":false") != std::string::npos);
}

void test_native_frontend_live_telemetry_publishes_mqtt_telemetry(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

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

void test_native_frontend_mqtt_set_threshold_command_publishes_result(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command("{\"protocol_version\":\"1.0\",\"command_id\":\"cmd-1\",\"command\":\"set_threshold\",\"threshold\":0.45}");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_threshold_calls);
  TEST_ASSERT_EQUAL_FLOAT(0.45f, frontend_runtime_shim::state.last_threshold);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/result", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_rejects_unsupported_protocol_version(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command(
      "{\"protocol_version\":\"2.0\",\"command_id\":\"cmd-version\",\"command\":\"info\"}");

  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/result", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"command_id\":\"cmd-version\"") != std::string::npos);
  TEST_ASSERT_TRUE(publish.payload.find("\"command\":\"info\"") != std::string::npos);
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_TRUE(publish.payload.find("\"code\":\"unsupported_version\"") != std::string::npos);
}

void test_native_frontend_mqtt_set_device_label_persists_and_republishes_info(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.device_label = "Living Room";
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;
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
      "{\"protocol_version\":\"1.0\",\"command_id\":\"label-1\",\"command\":\"set_device_label\",\"device_label\":\"Kitchen Sensor\"}");

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
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/result", result.topic.c_str());
  TEST_ASSERT_TRUE(result.payload.find("\"command\":\"set_device_label\"") != std::string::npos);
  TEST_ASSERT_TRUE(result.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_recalibrate_command_publishes_result(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command("{\"protocol_version\":\"1.0\",\"command_id\":\"cmd-recal\",\"command\":\"recalibrate\"}");

  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.trigger_recalibration_calls);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  const auto &publish = mqtt_transport_mock::state.publishes.back();
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/result", publish.topic.c_str());
  TEST_ASSERT_TRUE(publish.payload.find("\"accepted\":true") != std::string::npos);
}

void test_native_frontend_mqtt_detector_command_updates_runtime(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command(
      "{\"protocol_version\":\"1.0\",\"command_id\":\"det-1\",\"command\":\"set_detector\",\"detector\":\"lightweight\"}");
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
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command(
      "{\"protocol_version\":\"1.0\",\"command_id\":\"motion-1\",\"command\":\"set_motion_hits\",\"motion_on_hits\":5,\"motion_off_hits\":3}");
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
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  RuntimeConfig runtime_config;
  frontend.set_runtime_config(runtime_config);
  TEST_ASSERT_TRUE(frontend.setup());

  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command(
      "{\"protocol_version\":\"1.0\",\"command_id\":\"traffic-1\",\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"pacing\"}");
  TEST_ASSERT_EQUAL(0, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":false") !=
                   std::string::npos);

  mqtt.emit_command(
      "{\"protocol_version\":\"1.0\",\"command_id\":\"traffic-1b\",\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"external\"}");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_csi_traffic_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_csi_traffic_mode == CsiTrafficMode::EXTERNAL);

  mqtt.emit_command(
      "{\"protocol_version\":\"1.0\",\"command_id\":\"traffic-2\",\"command\":\"set_traffic_generator_mode\",\"traffic_generator_mode\":\"ping\"}");
  TEST_ASSERT_EQUAL(1, frontend_runtime_shim::state.set_traffic_generator_mode_calls);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.last_traffic_generator_mode == RuntimeTrafficMode::PING);
  TEST_ASSERT_TRUE(!mqtt_transport_mock::state.publishes.empty());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.back().payload.find("\"accepted\":true") !=
                   std::string::npos);
}

void test_native_frontend_mqtt_rejects_direct_local_commands_with_forbidden(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;
  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt_transport_mock::state.publishes.clear();

  mqtt.emit_command(
      "{\"protocol_version\":\"1.0\",\"command_id\":\"wifi-local\",\"command\":\"set_wifi_bssid\",\"bssid\":\"E6:FA:C4:20:19:DE\"}");

  TEST_ASSERT_EQUAL(1, static_cast<int>(mqtt_transport_mock::state.publishes.size()));
  const auto &result = mqtt_transport_mock::state.publishes[0];
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/result", result.topic.c_str());
  TEST_ASSERT_TRUE(result.payload.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_TRUE(result.payload.find("\"code\":\"forbidden\"") != std::string::npos);
}

void test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads(void) {
  MockMqttTransport mqtt;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;
  frontend_runtime_shim::state.diagnostics.traffic_packets_total = 100U;
  frontend_runtime_shim::state.diagnostics.csi_callbacks_total = 100U;
  frontend_runtime_shim::state.diagnostics.csi_accepted_total = 90U;
  frontend_runtime_shim::state.diagnostics.csi_filtered_total = 10U;

  NativeFrontend frontend(&mqtt);
  frontend.set_device_config(config);
  TEST_ASSERT_TRUE(frontend.setup());
  frontend_runtime_shim::state.diagnostics.traffic_packets_total = 600U;
  frontend_runtime_shim::state.diagnostics.csi_callbacks_total = 580U;
  frontend_runtime_shim::state.diagnostics.csi_accepted_total = 540U;
  frontend_runtime_shim::state.diagnostics.csi_filtered_total = 40U;
  frontend_runtime_shim::state.diagnostics.wifi_channel = 10U;
  frontend_runtime_shim::state.diagnostics.wifi_rssi_dbm = -55;
  frontend_runtime_shim::state.diagnostics_sample.traffic_tx_pps = 100.0f;
  frontend_runtime_shim::state.diagnostics_sample.csi_callback_pps = 96.0f;
  frontend_runtime_shim::state.diagnostics_sample.csi_accepted_pps = 90.0f;
  frontend_runtime_shim::state.diagnostics_sample.csi_filtered_pps = 6.0f;
  frontend_runtime_shim::state.diagnostics_sample.wifi_channel = 10U;
  frontend_runtime_shim::state.diagnostics_sample.wifi_rssi_dbm = -55;
  mqtt_transport_mock::state.publishes.clear();

  RuntimeSnapshot snapshot = make_ready_snapshot();
  frontend.on_motion_state_changed(snapshot);
  frontend.loop();
  mqtt_transport_mock::state.publishes.clear();
  mqtt.emit_command("{\"protocol_version\":\"1.0\",\"command_id\":\"cmd-info\",\"command\":\"info\"}");
  mqtt.emit_command("{\"protocol_version\":\"1.0\",\"command_id\":\"cmd-diagnostics\",\"command\":\"diagnostics\"}");

  TEST_ASSERT_EQUAL(2, static_cast<int>(mqtt_transport_mock::state.publishes.size()));
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/result",
                           mqtt_transport_mock::state.publishes[0].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"csi_traffic_mode\":\"internal\"") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"traffic_mode\":\"ping\"") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"csi_target_pps\":100") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[0].payload.find("\"evaluation_interval_ms\":250") !=
                   std::string::npos);
  TEST_ASSERT_EQUAL_STRING("espectre/v1/devices/0000abcdeffedcba/commands/result",
                           mqtt_transport_mock::state.publishes[1].topic.c_str());
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"uptime\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"free_memory_kb\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"loop_time_ms\":") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"traffic_tx_pps\":100") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"csi_callback_pps\":96") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"csi_accepted_pps\":90") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"csi_filtered_pps\":6") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"wifi_channel\":10") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"wifi_rssi_dbm\":-55") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[1].payload.find("\"movement\":") == std::string::npos);

  mqtt.emit_command("{\"protocol_version\":\"1.0\",\"command_id\":\"cmd-capabilities\",\"command\":\"capabilities\"}");
  TEST_ASSERT_EQUAL(3, static_cast<int>(mqtt_transport_mock::state.publishes.size()));
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"commands\":[") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes[2].payload.find("\"set_device_label\"") != std::string::npos);
}

void test_native_frontend_serializes_telemetry_once_for_active_transports(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  MockMqttTransport mqtt;
  MockDirectHttpService direct;
  NativeFrontend frontend(&mqtt, nullptr, &direct);
  EspectreDeviceConfig config;
  config.device_id = 0x0000111122223333ULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;
  frontend.set_device_config(config);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  frontend.set_wifi_provisioning_info(wifi);
  EspectreDeviceInfo info;
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  mqtt.emit_connection(true);
  direct.emit_client_count(1U);
  mqtt_transport_mock::state.publishes.clear();

  frontend.on_live_telemetry(2.5f, 1.25f);
  TEST_ASSERT_EQUAL(0, static_cast<int>(direct_http_service_mock::state.published_events.size()));
  frontend.loop();
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_http_service_mock::state.published_events.size()));
  const auto &event = direct_http_service_mock::state.published_events.front();
  TEST_ASSERT_EQUAL_STRING("telemetry", event.event_name.c_str());
  TEST_ASSERT_TRUE(event.replaceable_telemetry);
  TEST_ASSERT_TRUE(event.data_json.find("\"movement_score\":2.5") != std::string::npos);
  const int mqtt_index = mqtt_publish_index("espectre/v1/devices/0000111122223333/telemetry");
  TEST_ASSERT_TRUE(mqtt_index >= 0);
  TEST_ASSERT_EQUAL_STRING(
      event.data_json.c_str(), mqtt_transport_mock::state.publishes[static_cast<size_t>(mqtt_index)].payload.c_str());
}


int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_native_frontend_mqtt_connect_enables_live_telemetry);
  RUN_TEST(test_native_frontend_direct_reports_mqtt_connection_changes);
  RUN_TEST(test_native_frontend_direct_clear_mqtt_disconnects_and_reports_status);
  RUN_TEST(test_native_frontend_live_telemetry_publishes_mqtt_telemetry);
  RUN_TEST(test_native_frontend_mqtt_set_threshold_command_publishes_result);
  RUN_TEST(test_native_frontend_mqtt_rejects_unsupported_protocol_version);
  RUN_TEST(test_native_frontend_mqtt_set_device_label_persists_and_republishes_info);
  RUN_TEST(test_native_frontend_mqtt_recalibrate_command_publishes_result);
  RUN_TEST(test_native_frontend_mqtt_detector_command_updates_runtime);
  RUN_TEST(test_native_frontend_mqtt_motion_hits_command_updates_runtime);
  RUN_TEST(test_native_frontend_mqtt_traffic_commands_update_runtime);
  RUN_TEST(test_native_frontend_mqtt_rejects_direct_local_commands_with_forbidden);
  RUN_TEST(test_native_frontend_mqtt_info_and_stats_commands_publish_protocol_payloads);
  RUN_TEST(test_native_frontend_serializes_telemetry_once_for_active_transports);
  return UNITY_END();
}
