/*
 * ESPectre - Shared Protocol Unit Tests
 *
 * Exercises JSON payload formatting and command parsing helpers used by the
 * runtime protocol surfaces.
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include "direct_http_protocol.h"
#include "espectre_protocol.h"
#include "ota_version.h"
#include "runtime_diagnostics.h"

#include <cmath>
#include <string>

using namespace espectre;

void test_ota_version_ordering_blocks_downgrades_and_divergent_builds(void) {
  TEST_ASSERT_TRUE(compare_ota_versions("2.8.0-280-gac7af68", "2.8.0-279-gc63eaed") ==
                   OtaVersionComparison::NEWER);
  TEST_ASSERT_TRUE(compare_ota_versions("2.8.0-279-gc63eaed", "2.8.0-280-gac7af68") ==
                   OtaVersionComparison::OLDER);
  TEST_ASSERT_TRUE(compare_ota_versions("2.8.0-280-gac7af68", "2.8.0-280-gac7af68-dirty") ==
                   OtaVersionComparison::SAME);
  TEST_ASSERT_TRUE(compare_ota_versions("2.8.0-280-gfffffff", "2.8.0-280-gac7af68") ==
                   OtaVersionComparison::UNORDERED);
  TEST_ASSERT_TRUE(compare_ota_versions("2.8.0-1-g0000001", "2.8.0") ==
                   OtaVersionComparison::NEWER);
  TEST_ASSERT_TRUE(compare_ota_versions("3.0.0", "3.0.0-rc.2") == OtaVersionComparison::NEWER);
  TEST_ASSERT_TRUE(compare_ota_versions("3.0.0-rc.2", "3.0.0-rc.1-5-gabcdef0") ==
                   OtaVersionComparison::NEWER);
  TEST_ASSERT_TRUE(compare_ota_versions("3.0.0", "2.8.0-999-gabcdef0") ==
                   OtaVersionComparison::NEWER);
  TEST_ASSERT_TRUE(compare_ota_versions("3.0.0", "unknown") == OtaVersionComparison::NEWER);
  TEST_ASSERT_TRUE(compare_ota_versions("snapshot", "3.0.0") == OtaVersionComparison::UNORDERED);
}

void test_device_id_helpers_format_and_parse_canonical_hex_consistently(void) {
  TEST_ASSERT_EQUAL_STRING("00007c2c6742bbac", format_espectre_device_id(0x00007C2C6742BBACULL).c_str());

  uint64_t parsed = 0U;
  TEST_ASSERT_TRUE(parse_espectre_device_id("00007c2c6742bbac", &parsed));
  TEST_ASSERT_EQUAL(0x00007C2C6742BBACULL, parsed);
  // Accept the legacy prefix while clients migrate to the canonical form.
  TEST_ASSERT_TRUE(parse_espectre_device_id("0x00007c2c6742bbac", &parsed));
  TEST_ASSERT_EQUAL(0x00007C2C6742BBACULL, parsed);
  TEST_ASSERT_TRUE(parse_espectre_device_id("124", &parsed));
  TEST_ASSERT_EQUAL(0x124ULL, parsed);
  TEST_ASSERT_FALSE(parse_espectre_device_id("bad-id", &parsed));

  const uint8_t mac[6] = {0x7C, 0x2C, 0x67, 0x42, 0xBB, 0xAC};
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  TEST_ASSERT_EQUAL(0x00007C2C6742BBACULL, espectre_device_id_from_mac(mac, sizeof(mac)));
  TEST_ASSERT_EQUAL(ESPECTRE_DEFAULT_DEVICE_ID, espectre_device_id_from_mac(nullptr, 0));
#pragma GCC diagnostic pop
  TEST_ASSERT_EQUAL_STRING("ESPectre C6 42bbac", espectre_device_name(0x00007C2C6742BBACULL, "esp32c6").c_str());
  TEST_ASSERT_EQUAL_STRING("ESPectre UNK 000000", espectre_device_name(ESPECTRE_DEFAULT_DEVICE_ID).c_str());
}

void test_effective_device_helpers_and_topic_generation_use_defaults(void) {
  EspectreDeviceConfig config;
  config.device_id = ESPECTRE_DEFAULT_DEVICE_ID;
  config.device_label.clear();
  config.topic_prefix = "custom/root/";

  TEST_ASSERT_EQUAL_STRING("0000000000000000", espectre_effective_device_id(config).c_str());
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DEFAULT_DEVICE_LABEL, espectre_effective_device_label(config).c_str());
  TEST_ASSERT_EQUAL_STRING("custom/root/0000000000000000/telemetry", espectre_topic(config, "telemetry").c_str());
  TEST_ASSERT_EQUAL_STRING("custom/root/0000000000000000/", espectre_topic(config, nullptr).c_str());
}

void test_clear_mqtt_config_resets_runtime_defaults(void) {
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  config.mqtt_port = 2883;
  config.mqtt_username = "user";
  config.mqtt_password = "secret";
  config.topic_prefix = "custom/root";

  clear_espectre_mqtt_config(&config);

  TEST_ASSERT_TRUE(config.mqtt_host.empty());
  TEST_ASSERT_EQUAL(1883, config.mqtt_port);
  TEST_ASSERT_TRUE(config.mqtt_username.empty());
  TEST_ASSERT_TRUE(config.mqtt_password.empty());
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_TOPIC_PREFIX, config.topic_prefix.c_str());

  clear_espectre_mqtt_config(nullptr);
}

void test_parse_mqtt_batch_config_command_updates_all_fields(void) {
  EspectreDeviceConfig config;
  std::string error;

  TEST_ASSERT_TRUE(parse_espectre_mqtt_config_command(
      "SET_MQTT_CONFIG:host=broker.local&port=2883&username=user%20name&password=s3cr%25t&topic_prefix=lab%2Froot",
      &config,
      &error));
  TEST_ASSERT_EQUAL_STRING("broker.local", config.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(2883, config.mqtt_port);
  TEST_ASSERT_EQUAL_STRING("user name", config.mqtt_username.c_str());
  TEST_ASSERT_EQUAL_STRING("s3cr%t", config.mqtt_password.c_str());
  TEST_ASSERT_EQUAL_STRING("lab/root", config.topic_prefix.c_str());

  TEST_ASSERT_FALSE(parse_espectre_mqtt_config_command("SET_MQTT_CONFIG:host=broker.local", &config, &error));
  TEST_ASSERT_EQUAL_STRING("missing mqtt port", error.c_str());
  TEST_ASSERT_FALSE(parse_espectre_mqtt_config_command("SET_MQTT_CONFIG:host=broker.local&port=0", &config, &error));
  TEST_ASSERT_EQUAL_STRING("mqtt port must be 1..65535", error.c_str());
}

void test_parse_mqtt_batch_config_command_accepts_host_with_scheme(void) {
  EspectreDeviceConfig config;
  std::string error;

  TEST_ASSERT_TRUE(parse_espectre_mqtt_config_command(
      "SET_MQTT_CONFIG:host=mqtts%3A%2F%2Fbroker.example.com&port=8883",
      &config,
      &error));
  TEST_ASSERT_EQUAL_STRING("mqtts://broker.example.com", config.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(8883, config.mqtt_port);
}

void test_status_telemetry_and_diagnostics_payloads_include_expected_fields(void) {
  EspectreDeviceConfig config;
  config.device_id = 0x0000000000000007ULL;

  RuntimeSnapshot snapshot;
  snapshot.motion_state = MotionState::MOTION;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.detector_name = "high_accuracy";

  const std::string status = espectre_status_payload(config, true, 1234);
  const std::string telemetry = espectre_telemetry_payload(config, snapshot, 222, 33, "native");
  const std::string diagnostics = espectre_diagnostics_payload(config, snapshot, 333, 44, 128.5f, 6.25f);

  TEST_ASSERT_TRUE(status.find("\"device_id\":\"0000000000000007\"") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"online\":true") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"frontend\":\"native\"") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"motion_state\":\"motion\"") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"threshold\":1.5") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"detector\":\"high_accuracy\"") != std::string::npos);
  snapshot.movement_metric = NAN;
  snapshot.threshold = NAN;
  const std::string telemetry_nan = espectre_telemetry_payload(config, snapshot, 222, 33, "native");
  TEST_ASSERT_TRUE(telemetry_nan.find("nan") == std::string::npos);
  TEST_ASSERT_TRUE(telemetry_nan.find("\"movement_score\":0") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"uptime\":44") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"free_memory_kb\":128.5") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"loop_time_ms\":6.25") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"traffic_tx_pps\"") == std::string::npos);
}

void test_diagnostics_payload_includes_enabled_runtime_sample(void) {
  EspectreDeviceConfig config;
  RuntimeSnapshot snapshot;
  RuntimeDiagnosticsSample diagnostics;
  diagnostics.traffic_tx_pps = 100.0f;
  diagnostics.csi_callback_pps = 96.0f;
  diagnostics.csi_accepted_pps = 90.0f;
  diagnostics.csi_admitted_pps = 84.0f;
  diagnostics.csi_filtered_pps = 6.0f;
  diagnostics.csi_missing_slots_pps = 16.0f;
  diagnostics.csi_excess_pps = 7.0f;
  diagnostics.csi_stale_pps = 1.0f;
  diagnostics.csi_out_of_order_pps = 2.0f;
  diagnostics.csi_occupancy_ratio = 0.84f;
  diagnostics.wifi_channel = 10U;
  diagnostics.wifi_rssi_dbm = -55;

  const std::string payload =
      espectre_diagnostics_payload(config, snapshot, 333, 44, 128.5f, 6.25f, &diagnostics);

  TEST_ASSERT_TRUE(payload.find("\"traffic_tx_pps\":100") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_callback_pps\":96") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_accepted_pps\":90") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_admitted_pps\":84") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_filtered_pps\":6") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_missing_slots_pps\":16") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_excess_pps\":7") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_stale_pps\":1") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_out_of_order_pps\":2") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_occupancy\":0.84") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"wifi_channel\":10") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"wifi_rssi_dbm\":-55") != std::string::npos);
}

void test_info_payload_uses_defaults_and_optional_sections(void) {
  EspectreDeviceConfig config;
  config.device_id = 0x0000000000000001ULL;
  config.device_label = "Kitchen \"node\"\nA";

  EspectreDeviceInfo info;
  info.frontend = "matter";
  info.firmware_version = "2026.7";
  info.chip = "esp32c6";
  info.detector = "lightweight";
  info.supports_diagnostics = true;
  info.supports_device_config = true;
  info.supports_runtime_threshold = true;
  info.supports_runtime_motion_hits = true;
  info.supports_runtime_detector = true;
  info.supports_manual_recalibration = true;
  info.supports_traffic_control = true;
  info.supports_ota = true;
  info.csi_traffic_mode = "internal";
  info.traffic_mode = "ping";
  info.csi_target_pps = 100U;
  info.evaluation_interval_ms = 250U;
  info.publish_interval_ms = 1000U;
  info.network.ip_address = "192.168.1.10";
  info.network.mac_address = "AA:BB:CC:DD:EE:FF";
  info.network.channel = 6;

  const std::string payload = espectre_info_payload(config, info);

  TEST_ASSERT_TRUE(payload.find("\"device_id\":\"0000000000000001\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"device_name\":\"ESPectre C6 000001\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"device_label\":\"Kitchen \\\"node\\\"\\nA\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"frontend\":\"matter\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"firmware_version\":\"2026.7\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"chip\":\"esp32c6\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"supports_") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"network\":{") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"ip_address\"") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"mac_address\"") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"channel\":{\"primary\":6}") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"detection\":{\"algorithm\":\"lightweight\"}") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_traffic_mode\":\"internal\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"traffic_mode\":\"ping\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_target_pps\":100") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"evaluation_interval_ms\":250") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"publish_interval_ms\":1000") != std::string::npos);

  const std::string catalog =
      espectre_capabilities_payload(config, info, true, true, true, true, true, true, true);
  TEST_ASSERT_TRUE(catalog.find("\"device_id\":\"0000000000000001\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"capabilities\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"traffic_udp_port\":5555") != std::string::npos);
  const std::string marker_property =
      std::string("\"traffic_marker\":\"") + RUNTIME_CSI_TRAFFIC_MARKER_UTF8 + "\"";
  TEST_ASSERT_TRUE(catalog.find(marker_property) != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"diagnostics\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"set_sensing\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"required\":[\"enabled\"]") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"access\":\"network_admin\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"wifi_access_points\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"scan_wifi_access_points\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"set_wifi_bssid\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"clear_wifi_config\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("set_wifi_config") == std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("band_policy") == std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"discover_peers\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"commands\"") == std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"stats\"") == std::string::npos);
  const std::string capability_result =
      "{\"command\":\"capabilities\",\"code\":\"ok\",\"message\":\"capabilities returned\",\"data\":" +
      catalog + "}";
  const std::string capability_response =
      direct_http_success_response("capabilities", capability_result);
  TEST_ASSERT_TRUE(capability_response.size() > ESPECTRE_DIRECT_MAX_REQUEST_SIZE);
  TEST_ASSERT_TRUE(capability_response.size() <= ESPECTRE_DIRECT_MAX_RESPONSE_SIZE);
}

void test_info_payload_omits_optional_sections_when_empty(void) {
  EspectreDeviceConfig config;
  config.device_id = ESPECTRE_DEFAULT_DEVICE_ID;
  config.device_label.clear();

  EspectreDeviceInfo info;
  info.frontend.clear();
  info.firmware_version.clear();
  info.chip.clear();
  info.detector.clear();

  const std::string payload = espectre_info_payload(config, info);

  TEST_ASSERT_TRUE(payload.find("\"device_id\":\"0000000000000000\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"device_name\":\"ESPectre UNK 000000\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"device_label\":\"\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"frontend\":\"native\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"firmware_version\":\"unknown\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"chip\":\"unknown\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"network\":{") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"detection\":{") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_traffic_mode\"") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"traffic_mode\"") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"csi_target_pps\"") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"evaluation_interval_ms\"") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"publish_interval_ms\"") == std::string::npos);

  const std::string catalog = espectre_capabilities_payload(config, info);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"capabilities\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"info\"") != std::string::npos);
  TEST_ASSERT_TRUE(catalog.find("\"name\":\"diagnostics\"") == std::string::npos);
}

void test_command_result_payload_includes_acceptance_and_message(void) {
  EspectreDeviceConfig config;
  config.device_id = 0x0000000000000005ULL;

  EspectreCommand command;
  command.command_id = "abc123";
  command.command = "set_threshold";

  const std::string accepted =
      espectre_command_result_payload(config, command, true, "ok", "applied", "{\"threshold\":0.5}");
  const std::string rejected =
      espectre_command_result_payload(config, command, false, "invalid_params", "");

  TEST_ASSERT_TRUE(accepted.find("\"command_id\":\"abc123\"") != std::string::npos);
  TEST_ASSERT_TRUE(accepted.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(accepted.find("\"code\":\"ok\"") != std::string::npos);
  TEST_ASSERT_TRUE(accepted.find("\"message\":\"applied\"") != std::string::npos);
  TEST_ASSERT_TRUE(accepted.find("\"data\":{\"threshold\":0.5}") != std::string::npos);
  TEST_ASSERT_TRUE(rejected.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_TRUE(rejected.find("\"code\":\"invalid_params\"") != std::string::npos);
  TEST_ASSERT_TRUE(rejected.find("\"message\":\"\"") != std::string::npos);
  TEST_ASSERT_TRUE(rejected.find("\"data\"") == std::string::npos);
}

void test_parse_espectre_command_parses_info_and_threshold_commands(void) {
  EspectreCommand command;
  std::string error;

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command_id\":\"x1\",\"command\":\"info\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("x1", command.command_id.c_str());
  TEST_ASSERT_EQUAL_STRING("info", command.command.c_str());
  TEST_ASSERT_FALSE(command.has_threshold);

  TEST_ASSERT_TRUE(
      parse_espectre_command("{\"command_id\":\"x-capabilities\",\"command\":\"capabilities\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("capabilities", command.command.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x-sensing\",\"command\":\"set_sensing\",\"enabled\":false}", &command, &error));
  TEST_ASSERT_TRUE(command.has_sensing_enabled);
  TEST_ASSERT_FALSE(command.sensing_enabled);

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x-label\",\"command\":\"set_device_label\",\"device_label\":\"Kitchen\"}",
      &command,
      &error));
  TEST_ASSERT_TRUE(command.has_device_label);
  TEST_ASSERT_EQUAL_STRING("Kitchen", command.device_label.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x-label-clear\",\"command\":\"set_device_label\",\"device_label\":\"\"}",
      &command,
      &error));
  TEST_ASSERT_TRUE(command.has_device_label);
  TEST_ASSERT_TRUE(command.device_label.empty());

  TEST_ASSERT_TRUE(
      parse_espectre_command("{\"command_id\":\"x2\",\"command\":\"set_threshold\",\"threshold\":2.5}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("set_threshold", command.command.c_str());
  TEST_ASSERT_TRUE(command.has_threshold);
  TEST_ASSERT_EQUAL_FLOAT(2.5f, command.threshold);

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x-motion\",\"command\":\"set_motion_hits\",\"motion_on_hits\":6,\"motion_off_hits\":4}",
      &command,
      &error));
  TEST_ASSERT_TRUE(command.has_motion_hits);
  TEST_ASSERT_EQUAL_UINT8(6U, command.motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(4U, command.motion_off_hits);

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x-detector\",\"command\":\"set_detector\",\"detector\":\"high_accuracy\"}",
      &command,
      &error));
  TEST_ASSERT_TRUE(command.has_detector);
  TEST_ASSERT_EQUAL_STRING("high_accuracy", command.detector.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command_id\":\"x3\",\"command\":\"ota_check\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("ota_check", command.command.c_str());
  TEST_ASSERT_FALSE(command.has_ota_channel);

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x3b\",\"command\":\"ota_check\",\"channel\":\"preview\"}", &command, &error));
  TEST_ASSERT_TRUE(command.has_ota_channel);
  TEST_ASSERT_EQUAL_STRING("preview", command.ota_channel.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command_id\":\"x4\",\"command\":\"ota_start\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("ota_start", command.command.c_str());
  TEST_ASSERT_FALSE(command.has_ota_channel);

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x4b\",\"command\":\"ota_start\",\"channel\":\"develop\"}", &command, &error));
  TEST_ASSERT_TRUE(command.has_ota_channel);
  TEST_ASSERT_EQUAL_STRING("develop", command.ota_channel.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command_id\":\"x5\",\"command\":\"recalibrate\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("recalibrate", command.command.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x6\",\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"external\"}",
      &command,
      &error));
  TEST_ASSERT_TRUE(command.has_csi_traffic_mode);
  TEST_ASSERT_EQUAL_STRING("external", command.csi_traffic_mode.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command(
      "{\"command_id\":\"x7\",\"command\":\"set_traffic_generator_mode\",\"traffic_generator_mode\":\"dns\"}",
      &command,
      &error));
  TEST_ASSERT_TRUE(command.has_traffic_generator_mode);
  TEST_ASSERT_EQUAL_STRING("dns", command.traffic_generator_mode.c_str());

}

void test_parse_espectre_command_rejects_missing_command_and_invalid_threshold(void) {
  EspectreCommand command;
  std::string error;

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command_id\":\"x3\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("missing command", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"set_device_label\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid device label (accepted: a single-line string)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command(
      "{\"command\":\"set_device_label\",\"device_label\":123}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid device label (accepted: a single-line string)", error.c_str());

  TEST_ASSERT_FALSE(
      parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":\"abc\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid threshold (accepted: 0.0-1.0)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":1e999}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid threshold (accepted: 0.0-1.0)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command(
      "{\"command\":\"set_motion_hits\",\"motion_on_hits\":\"abc\",\"motion_off_hits\":2}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid motion hits (accepted: motion_on_hits and motion_off_hits in 1-20)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command(
      "{\"command\":\"set_detector\",\"detector\":\"pca\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid detector (accepted: lightweight and high_accuracy)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command(
      "{\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"bogus\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid csi traffic mode (accepted: internal and external)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command(
      "{\"command\":\"set_csi_traffic_mode\",\"csi_traffic_mode\":\"pacing\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid csi traffic mode (accepted: internal and external)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command(
      "{\"command\":\"set_traffic_generator_mode\",\"traffic_generator_mode\":\"udp\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid traffic generator mode (accepted: ping and dns)", error.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command\":\"ota_check\"}", &command, &error));

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command\":\"ota_start\"}", &command, &error));

  TEST_ASSERT_FALSE(parse_espectre_command(
      "{\"command\":\"ota_start\",\"image_url\":\"https://fw.example/native.bin\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("ota overrides are not supported (manifest_url, image_url, and version are not accepted)",
                           error.c_str());

  TEST_ASSERT_FALSE(
      parse_espectre_command("{\"command\":\"ota_check\",\"manifest_url\":\"\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("ota overrides are not supported (manifest_url, image_url, and version are not accepted)",
                           error.c_str());

  TEST_ASSERT_FALSE(
      parse_espectre_command("{\"command\":\"ota_check\",\"channel\":\"latest\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid ota channel (accepted: release, preview, and develop)", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"info\"}", nullptr, &error));
}

void test_ota_status_payload_includes_expected_fields(void) {
  EspectreDeviceConfig config;
  config.device_id = 0x000000000000000AULL;

  EspectreOtaStatus status;
  status.state = EspectreOtaState::UPDATE_AVAILABLE;
  status.current_version = "1.0.0";
  status.target_version = "1.1.0";
  status.manifest_url = "https://fw.example/manifest.json";
  status.image_url = "https://fw.example/native.bin";
  status.default_channel = "release";
  status.channel = "preview";
  status.message = "update available";
  status.busy = false;
  status.update_available = true;

  const std::string payload = espectre_ota_status_payload(config, status, 4321);

  TEST_ASSERT_TRUE(payload.find("\"device_id\":\"000000000000000a\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"state\":\"update_available\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"current_version\":\"1.0.0\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"target_version\":\"1.1.0\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"update_available\":true") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"manifest_url\":\"https://fw.example/manifest.json\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"default_channel\":\"release\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"channel\":\"preview\"") != std::string::npos);
}

void test_ota_channel_helpers(void) {
  TEST_ASSERT_TRUE(espectre_ota_channel_accepted("release"));
  TEST_ASSERT_TRUE(espectre_ota_channel_accepted("preview"));
  TEST_ASSERT_TRUE(espectre_ota_channel_accepted("develop"));
  TEST_ASSERT_FALSE(espectre_ota_channel_accepted(""));
  TEST_ASSERT_FALSE(espectre_ota_channel_accepted("latest"));
  TEST_ASSERT_EQUAL_STRING(
      "https://github.com/francescopace/espectre/releases/latest/download/espectre-native-ota-esp32c3.json",
      espectre_ota_manifest_url("native", "esp32c3", "release").c_str());
  const std::string preview_url =
      std::string("https://github.com/francescopace/espectre/releases/download/") +
      ESPECTRE_OTA_RELEASE_TAG_PREVIEW + "/espectre-native-ota-esp32c6.json";
  const std::string develop_url =
      std::string("https://github.com/francescopace/espectre/releases/download/") +
      ESPECTRE_OTA_RELEASE_TAG_DEVELOP + "/espectre-native-ota-esp32s3.json";
  TEST_ASSERT_EQUAL_STRING(preview_url.c_str(),
                           espectre_ota_manifest_url("native", "esp32c6", "preview").c_str());
  TEST_ASSERT_EQUAL_STRING(develop_url.c_str(),
                           espectre_ota_manifest_url("native", "esp32s3", "develop").c_str());
  TEST_ASSERT_TRUE(espectre_ota_manifest_url("native", "esp32c3", "latest").empty());

}

void test_parse_espectre_config_command_updates_supported_fields(void) {
  EspectreDeviceConfig config;
  std::string error;

  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_label=Office", &config, &error));

  TEST_ASSERT_EQUAL_STRING("Office", config.device_label.c_str());
}

void test_parse_espectre_config_command_rejects_invalid_inputs(void) {
  EspectreDeviceConfig config;
  std::string error;

  TEST_ASSERT_FALSE(parse_espectre_config_command("BAD_PREFIX:device_label=Office", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid prefix", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_label", &config, &error));
  TEST_ASSERT_EQUAL_STRING("expected key=value", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_port=2883", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid config field", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:unsupported=value", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid config field", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_label=test", nullptr, &error));
}

void test_direct_http_request_parses_versioned_envelope(void) {
  DirectRequest request;
  std::string error;

  TEST_ASSERT_TRUE(parse_direct_http_request(
      "{\"v\":1,\"type\":\"request\",\"id\":\"req:42\",\"method\":\"set_threshold\","
      "\"params\":{\"threshold\":0.42},\"future\":true}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("req:42", request.id.c_str());
  TEST_ASSERT_EQUAL_STRING("set_threshold", request.method.c_str());
  TEST_ASSERT_EQUAL_STRING("{\"threshold\":0.42}", request.params.c_str());

  TEST_ASSERT_TRUE(parse_direct_http_request(
      "{\"v\":1,\"type\":\"request\",\"id\":\"unicode-\\u0031\",\"method\":\"info\"}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("unicode-1", request.id.c_str());
  TEST_ASSERT_EQUAL_STRING("{}", request.params.c_str());
}

void test_direct_http_request_rejects_invalid_boundaries(void) {
  DirectRequest request;
  std::string error;

  TEST_ASSERT_FALSE(parse_direct_http_request("", &request, &error));
  TEST_ASSERT_EQUAL_STRING("empty Direct request", error.c_str());
  TEST_ASSERT_FALSE(parse_direct_http_request("{", &request, &error));
  TEST_ASSERT_FALSE(parse_direct_http_request(
      "{\"v\":1,\"v\":1,\"type\":\"request\",\"id\":\"x\",\"method\":\"info\"}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("duplicate JSON object field", error.c_str());
  TEST_ASSERT_FALSE(parse_direct_http_request(
      "{\"v\":\"1\",\"type\":\"request\",\"id\":\"x\",\"method\":\"info\"}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("unsupported Direct envelope version", error.c_str());
  TEST_ASSERT_FALSE(parse_direct_http_request(
      "{\"v\":1,\"type\":\"event\",\"id\":\"x\",\"method\":\"info\"}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("Direct client messages must have type request", error.c_str());
  TEST_ASSERT_FALSE(parse_direct_http_request(
      "{\"v\":1,\"type\":\"request\",\"id\":\"bad id\",\"method\":\"info\"}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("invalid Direct request id", error.c_str());
  TEST_ASSERT_FALSE(parse_direct_http_request(
      "{\"v\":1,\"type\":\"request\",\"id\":\"x\",\"method\":\"Info!\"}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("invalid Direct request method", error.c_str());
  TEST_ASSERT_FALSE(parse_direct_http_request(
      "{\"v\":1,\"type\":\"request\",\"id\":\"x\",\"method\":\"info\",\"params\":[]}",
      &request,
      &error));
  TEST_ASSERT_EQUAL_STRING("Direct request params must be an object", error.c_str());

  const std::string oversized(ESPECTRE_DIRECT_MAX_REQUEST_SIZE + 1U, 'x');
  TEST_ASSERT_FALSE(parse_direct_http_request(oversized, &request, &error));
  TEST_ASSERT_EQUAL_STRING("Direct request exceeds the size limit", error.c_str());
  TEST_ASSERT_FALSE(parse_direct_http_request("{}", nullptr, &error));
  TEST_ASSERT_EQUAL_STRING("request output is required", error.c_str());
}

void test_direct_http_builders_emit_valid_correlated_envelopes(void) {
  const std::string success = direct_http_success_response("req-1", "{\"device_id\":\"abc\"}");
  const std::string rejected = direct_http_error_response("req-2", "invalid_request", "bad \"field\"");
  const std::string event = direct_http_event("telemetry", "{\"movement_score\":0.25}");

  TEST_ASSERT_EQUAL_STRING(
      "{\"v\":1,\"type\":\"response\",\"id\":\"req-1\",\"ok\":true,\"result\":{\"device_id\":\"abc\"}}",
      success.c_str());
  TEST_ASSERT_EQUAL_STRING(
      "{\"v\":1,\"type\":\"response\",\"id\":\"req-2\",\"ok\":false,\"error\":{\"code\":\"invalid_request\",\"message\":\"bad \\\"field\\\"\"}}",
      rejected.c_str());
  TEST_ASSERT_EQUAL_STRING(
      "{\"v\":1,\"type\":\"event\",\"event\":\"telemetry\",\"data\":{\"movement_score\":0.25}}",
      event.c_str());
  TEST_ASSERT_TRUE(direct_http_success_response("req-3", "not-json").find("\"result\":{}") !=
                   std::string::npos);
  TEST_ASSERT_TRUE(direct_http_event("status", "[]").find("\"data\":{}") != std::string::npos);
}

void test_direct_http_request_reuses_transport_neutral_command_validation(void) {
  DirectRequest request;
  EspectreCommand command;
  std::string error;

  TEST_ASSERT_TRUE(parse_direct_http_request(
      "{\"v\":1,\"type\":\"request\",\"id\":\"direct-1\",\"method\":\"set_motion_hits\","
      "\"params\":{\"motion_on_hits\":6,\"motion_off_hits\":4}}",
      &request,
      &error));
  TEST_ASSERT_TRUE(direct_http_request_to_command(request, &command, &error));
  TEST_ASSERT_EQUAL_STRING("direct-1", command.command_id.c_str());
  TEST_ASSERT_EQUAL_STRING("set_motion_hits", command.command.c_str());
  TEST_ASSERT_EQUAL_UINT8(6U, command.motion_on_hits);
  TEST_ASSERT_EQUAL_UINT8(4U, command.motion_off_hits);

  request.method = "set_threshold";
  request.params = "{\"threshold\":\"0.5\"}";
  TEST_ASSERT_FALSE(direct_http_request_to_command(request, &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid threshold (accepted: 0.0-1.0)", error.c_str());

  request.method = "unknown_method";
  request.params = "{}";
  TEST_ASSERT_TRUE(direct_http_request_to_command(request, &command, &error));
  TEST_ASSERT_EQUAL_STRING("unknown_method", command.command.c_str());
}

void test_direct_http_configuration_commands_validate_write_only_fields(void) {
  EspectreCommand command;
  std::string error;

  TEST_ASSERT_TRUE(parse_espectre_command_request(
      "wifi-1",
      "set_wifi_bssid",
      "{\"bssid\":\"E6:FA:C4:20:19:DE\"}",
      &command,
      &error));
  TEST_ASSERT_TRUE(command.has_wifi_bssid);
  TEST_ASSERT_EQUAL_STRING("E6:FA:C4:20:19:DE", command.wifi_bssid.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command_request(
      "mqtt-1",
      "set_mqtt_config",
      "{\"host\":\"homeassistant.local\",\"port\":1883,\"username\":\"mqtt\",\"password\":\"secret\"}",
      &command,
      &error));
  TEST_ASSERT_EQUAL_STRING("homeassistant.local", command.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(1883U, command.mqtt_port);
  TEST_ASSERT_EQUAL_STRING("mqtt", command.mqtt_username.c_str());
  TEST_ASSERT_TRUE(command.has_mqtt_password);

  TEST_ASSERT_FALSE(parse_espectre_command_request(
      "wifi-bad", "set_wifi_bssid", "{\"bssid\":\"not-a-bssid\"}", &command, &error));
  TEST_ASSERT_FALSE(parse_espectre_command_request(
      "wifi-credentials", "set_wifi_bssid", "{\"ssid\":\"Lab\",\"password\":\"secret\"}",
      &command, &error));
  TEST_ASSERT_FALSE(parse_espectre_command_request(
      "wifi-band", "set_wifi_bssid", "{\"bssid\":\"\",\"band_policy\":\"auto\"}",
      &command, &error));
  TEST_ASSERT_TRUE(parse_espectre_command_request(
      "clear-bssid", "clear_wifi_bssid", "{}", &command, &error));
  TEST_ASSERT_FALSE(parse_espectre_command_request(
      "clear-bssid-bad", "clear_wifi_bssid", "{\"bssid\":\"E6:FA:C4:20:19:DE\"}",
      &command, &error));
  TEST_ASSERT_TRUE(parse_espectre_command_request(
      "clear-wifi", "clear_wifi_config", "{}", &command, &error));
  TEST_ASSERT_FALSE(parse_espectre_command_request(
      "clear-wifi-bad", "clear_wifi_config", "{\"ssid\":\"Lab\"}", &command, &error));
  TEST_ASSERT_FALSE(parse_espectre_command_request(
      "mqtt-bad", "set_mqtt_config", "{\"host\":\"homeassistant.local\",\"port\":0}", &command, &error));
  TEST_ASSERT_TRUE(parse_espectre_command_request("clear-2", "clear_mqtt_config", "{}", &command, &error));
}

void test_direct_http_read_and_sensing_methods_map_to_shared_commands(void) {
  const char *methods[] = {
      "capabilities", "info", "status", "config", "diagnostics", "ota_status", "wifi_access_points"};
  for (const char *method : methods) {
    EspectreCommand command;
    std::string error;
    TEST_ASSERT_TRUE(parse_espectre_command_request("direct-read", method, "{}", &command, &error));
    TEST_ASSERT_EQUAL_STRING(method, command.command.c_str());
    TEST_ASSERT_EQUAL_STRING("direct-read", command.command_id.c_str());
  }

  EspectreCommand command;
  std::string error;
  TEST_ASSERT_TRUE(
      parse_espectre_command_request("direct-sensing", "set_sensing", "{\"enabled\":true}", &command, &error));
  TEST_ASSERT_TRUE(command.has_sensing_enabled);
  TEST_ASSERT_TRUE(command.sensing_enabled);

  TEST_ASSERT_TRUE(parse_espectre_command_request("legacy", "start_sensing", "{}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("start_sensing", command.command.c_str());
  TEST_ASSERT_FALSE(parse_espectre_command_request(
      "extra", "set_sensing", "{\"enabled\":true,\"unexpected\":1}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("unknown command parameter", error.c_str());
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_ota_version_ordering_blocks_downgrades_and_divergent_builds);
  RUN_TEST(test_device_id_helpers_format_and_parse_canonical_hex_consistently);
  RUN_TEST(test_effective_device_helpers_and_topic_generation_use_defaults);
  RUN_TEST(test_clear_mqtt_config_resets_runtime_defaults);
  RUN_TEST(test_parse_mqtt_batch_config_command_updates_all_fields);
  RUN_TEST(test_status_telemetry_and_diagnostics_payloads_include_expected_fields);
  RUN_TEST(test_diagnostics_payload_includes_enabled_runtime_sample);
  RUN_TEST(test_info_payload_uses_defaults_and_optional_sections);
  RUN_TEST(test_info_payload_omits_optional_sections_when_empty);
  RUN_TEST(test_command_result_payload_includes_acceptance_and_message);
  RUN_TEST(test_parse_espectre_command_parses_info_and_threshold_commands);
  RUN_TEST(test_parse_espectre_command_rejects_missing_command_and_invalid_threshold);
  RUN_TEST(test_ota_status_payload_includes_expected_fields);
  RUN_TEST(test_ota_channel_helpers);
  RUN_TEST(test_parse_espectre_config_command_updates_supported_fields);
  RUN_TEST(test_parse_espectre_config_command_rejects_invalid_inputs);
  RUN_TEST(test_direct_http_request_parses_versioned_envelope);
  RUN_TEST(test_direct_http_request_rejects_invalid_boundaries);
  RUN_TEST(test_direct_http_builders_emit_valid_correlated_envelopes);
  RUN_TEST(test_direct_http_request_reuses_transport_neutral_command_validation);
  RUN_TEST(test_direct_http_configuration_commands_validate_write_only_fields);
  RUN_TEST(test_direct_http_read_and_sensing_methods_map_to_shared_commands);
  return UNITY_END();
}

#if defined(ESP_PLATFORM)
extern "C" void app_main(void) { process(); }
#else
int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  return process();
}
#endif
