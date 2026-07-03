/*
 * ESPectre - Shared Protocol Unit Tests
 *
 * Exercises JSON payload formatting and command parsing helpers used by the
 * runtime protocol surfaces.
 */

#include "test_harness.h"

#include "espectre_protocol.h"

#include <string>

using namespace esphome::espectre;

void test_effective_device_helpers_and_topic_generation_use_defaults(void) {
  EspectreDeviceConfig config;
  config.device_id.clear();
  config.device_name.clear();
  config.topic_prefix = "custom/root/";

  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DEFAULT_DEVICE_ID, espectre_effective_device_id(config).c_str());
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DEFAULT_DEVICE_NAME, espectre_effective_device_name(config).c_str());
  TEST_ASSERT_EQUAL_STRING("custom/root/espectre-node/telemetry", espectre_topic(config, "telemetry").c_str());
  TEST_ASSERT_EQUAL_STRING("custom/root/espectre-node/", espectre_topic(config, nullptr).c_str());
}

void test_clear_mqtt_config_resets_runtime_defaults(void) {
  EspectreDeviceConfig config;
  config.mqtt_host = "broker.local";
  config.mqtt_port = 2883;
  config.mqtt_username = "user";
  config.mqtt_password = "secret";
  config.topic_prefix = "custom/root";
  config.mqtt_enabled = true;

  clear_espectre_mqtt_config(&config);

  TEST_ASSERT_TRUE(config.mqtt_host.empty());
  TEST_ASSERT_EQUAL(1883, config.mqtt_port);
  TEST_ASSERT_TRUE(config.mqtt_username.empty());
  TEST_ASSERT_TRUE(config.mqtt_password.empty());
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_TOPIC_PREFIX, config.topic_prefix.c_str());
  TEST_ASSERT_FALSE(config.mqtt_enabled);

  clear_espectre_mqtt_config(nullptr);
}

void test_status_telemetry_and_stats_payloads_include_expected_fields(void) {
  EspectreDeviceConfig config;
  config.device_id = "node-7";

  RuntimeSnapshot snapshot;
  snapshot.motion_state = MotionState::MOTION;
  snapshot.movement_metric = 2.75f;
  snapshot.threshold = 1.5f;
  snapshot.detector_name = "ml";
  snapshot.gain_locked = true;

  const std::string status = espectre_status_payload(config, true, 1234);
  const std::string telemetry = espectre_telemetry_payload(config, snapshot, 222, 33, "native");
  const std::string stats = espectre_stats_payload(config, snapshot, 333, 44, 128.5f, 6.25f);

  TEST_ASSERT_TRUE(status.find("\"device_id\":\"node-7\"") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"online\":true") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"frontend\":\"native\"") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"motion_state\":\"motion\"") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"threshold\":1.5") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"detector\":\"ml\"") != std::string::npos);
  TEST_ASSERT_TRUE(telemetry.find("\"gain_locked\":true") != std::string::npos);
  TEST_ASSERT_TRUE(stats.find("\"uptime\":44") != std::string::npos);
  TEST_ASSERT_TRUE(stats.find("\"free_memory_kb\":128.5") != std::string::npos);
  TEST_ASSERT_TRUE(stats.find("\"loop_time_ms\":6.25") != std::string::npos);
}

void test_info_payload_uses_defaults_and_optional_sections(void) {
  EspectreDeviceConfig config;
  config.device_id = "node-1";
  config.device_name = "Kitchen \"node\"\nA";

  EspectreDeviceInfo info;
  info.frontend = "streamer";
  info.firmware_version = "2026.7";
  info.chip = "esp32c6";
  info.detector = "mvs";
  info.network.ip_address = "192.168.1.10";
  info.network.mac_address = "AA:BB:CC:DD:EE:FF";
  info.network.channel = 6;

  const std::string payload = espectre_info_payload(config, info);

  TEST_ASSERT_TRUE(payload.find("\"device_id\":\"node-1\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"device_name\":\"Kitchen \\\"node\\\"\\nA\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"frontend\":\"streamer\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"firmware_version\":\"2026.7\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"chip\":\"esp32c6\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"network\":{") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"ip_address\":\"192.168.1.10\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"channel\":{\"primary\":6}") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"detection\":{\"algorithm\":\"mvs\"}") != std::string::npos);
}

void test_info_payload_omits_optional_sections_when_empty(void) {
  EspectreDeviceConfig config;
  config.device_id.clear();
  config.device_name.clear();

  EspectreDeviceInfo info;
  info.frontend.clear();
  info.firmware_version.clear();
  info.chip.clear();
  info.detector.clear();

  const std::string payload = espectre_info_payload(config, info);

  TEST_ASSERT_TRUE(payload.find("\"device_id\":\"espectre-node\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"device_name\":\"ESPectre Node\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"frontend\":\"native\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"firmware_version\":\"unknown\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"chip\":\"unknown\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"network\":{") == std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"detection\":{") == std::string::npos);
}

void test_command_result_payload_includes_acceptance_and_message(void) {
  EspectreDeviceConfig config;
  config.device_id = "node-5";

  EspectreCommand command;
  command.command_id = "abc123";
  command.command = "set_threshold";

  const std::string accepted = espectre_command_result_payload(config, command, true, "applied");
  const std::string rejected = espectre_command_result_payload(config, command, false, nullptr);

  TEST_ASSERT_TRUE(accepted.find("\"command_id\":\"abc123\"") != std::string::npos);
  TEST_ASSERT_TRUE(accepted.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(accepted.find("\"message\":\"applied\"") != std::string::npos);
  TEST_ASSERT_TRUE(rejected.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_TRUE(rejected.find("\"message\":\"\"") != std::string::npos);
}

void test_parse_espectre_command_parses_info_and_threshold_commands(void) {
  EspectreCommand command;
  std::string error;

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command_id\":\"x1\",\"command\":\"info\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("x1", command.command_id.c_str());
  TEST_ASSERT_EQUAL_STRING("info", command.command.c_str());
  TEST_ASSERT_FALSE(command.has_threshold);

  TEST_ASSERT_TRUE(
      parse_espectre_command("{\"command_id\":\"x2\",\"command\":\"set_threshold\",\"threshold\":2.5}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("set_threshold", command.command.c_str());
  TEST_ASSERT_TRUE(command.has_threshold);
  TEST_ASSERT_EQUAL_FLOAT(2.5f, command.threshold);

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command_id\":\"x3\",\"command\":\"ota_check\",\"manifest_url\":\"https://fw.example/manifest.json\"}",
                                          &command,
                                          &error));
  TEST_ASSERT_EQUAL_STRING("ota_check", command.command.c_str());
  TEST_ASSERT_TRUE(command.has_manifest_url);
  TEST_ASSERT_EQUAL_STRING("https://fw.example/manifest.json", command.manifest_url.c_str());

  TEST_ASSERT_TRUE(parse_espectre_command("{\"command_id\":\"x4\",\"command\":\"ota_start\",\"image_url\":\"https://fw.example/native.bin\",\"version\":\"2026.7.3\"}",
                                          &command,
                                          &error));
  TEST_ASSERT_EQUAL_STRING("ota_start", command.command.c_str());
  TEST_ASSERT_TRUE(command.has_image_url);
  TEST_ASSERT_TRUE(command.has_version);
  TEST_ASSERT_EQUAL_STRING("https://fw.example/native.bin", command.image_url.c_str());
  TEST_ASSERT_EQUAL_STRING("2026.7.3", command.version.c_str());
}

void test_parse_espectre_command_rejects_missing_command_and_invalid_threshold(void) {
  EspectreCommand command;
  std::string error;

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command_id\":\"x3\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("missing command", error.c_str());

  TEST_ASSERT_FALSE(
      parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":\"abc\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid threshold", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"set_threshold\",\"threshold\":1e999}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("invalid threshold", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"ota_check\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("missing manifest_url", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"ota_start\"}", &command, &error));
  TEST_ASSERT_EQUAL_STRING("missing manifest_url or image_url", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_command("{\"command\":\"info\"}", nullptr, &error));
}

void test_ota_status_payload_includes_expected_fields(void) {
  EspectreDeviceConfig config;
  config.device_id = "node-ota";

  EspectreOtaStatus status;
  status.state = EspectreOtaState::UPDATE_AVAILABLE;
  status.current_version = "1.0.0";
  status.target_version = "1.1.0";
  status.manifest_url = "https://fw.example/manifest.json";
  status.image_url = "https://fw.example/native.bin";
  status.message = "update available";
  status.busy = false;
  status.update_available = true;

  const std::string payload = espectre_ota_status_payload(config, status, 4321);

  TEST_ASSERT_TRUE(payload.find("\"device_id\":\"node-ota\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"state\":\"update_available\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"current_version\":\"1.0.0\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"target_version\":\"1.1.0\"") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"update_available\":true") != std::string::npos);
  TEST_ASSERT_TRUE(payload.find("\"manifest_url\":\"https://fw.example/manifest.json\"") != std::string::npos);
}

void test_parse_espectre_config_command_updates_supported_fields(void) {
  EspectreDeviceConfig config;
  std::string error;

  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_name=Office", &config, &error));
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_host=broker.local", &config, &error));
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_username=user", &config, &error));
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_password=secret", &config, &error));
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:topic_prefix=", &config, &error));
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_port=2883", &config, &error));
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_enabled=false", &config, &error));
  TEST_ASSERT_TRUE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_enabled=true", &config, &error));

  TEST_ASSERT_EQUAL_STRING("Office", config.device_name.c_str());
  TEST_ASSERT_EQUAL_STRING("broker.local", config.mqtt_host.c_str());
  TEST_ASSERT_EQUAL_STRING("user", config.mqtt_username.c_str());
  TEST_ASSERT_EQUAL_STRING("secret", config.mqtt_password.c_str());
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_TOPIC_PREFIX, config.topic_prefix.c_str());
  TEST_ASSERT_EQUAL(2883, config.mqtt_port);
  TEST_ASSERT_TRUE(config.mqtt_enabled);
}

void test_parse_espectre_config_command_rejects_invalid_inputs(void) {
  EspectreDeviceConfig config;
  std::string error;

  TEST_ASSERT_FALSE(parse_espectre_config_command("BAD_PREFIX:device_name=Office", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid prefix", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_name", &config, &error));
  TEST_ASSERT_EQUAL_STRING("expected key=value", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_port=0", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid config field", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_port=70000", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid config field", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:mqtt_enabled=maybe", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid config field", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:unsupported=value", &config, &error));
  TEST_ASSERT_EQUAL_STRING("invalid config field", error.c_str());

  TEST_ASSERT_FALSE(parse_espectre_config_command("SET_DEVICE_CONFIG:device_name=test", nullptr, &error));
}

int process(void) {
  UNITY_BEGIN();
  RUN_TEST(test_effective_device_helpers_and_topic_generation_use_defaults);
  RUN_TEST(test_clear_mqtt_config_resets_runtime_defaults);
  RUN_TEST(test_status_telemetry_and_stats_payloads_include_expected_fields);
  RUN_TEST(test_info_payload_uses_defaults_and_optional_sections);
  RUN_TEST(test_info_payload_omits_optional_sections_when_empty);
  RUN_TEST(test_command_result_payload_includes_acceptance_and_message);
  RUN_TEST(test_parse_espectre_command_parses_info_and_threshold_commands);
  RUN_TEST(test_parse_espectre_command_rejects_missing_command_and_invalid_threshold);
  RUN_TEST(test_ota_status_payload_includes_expected_fields);
  RUN_TEST(test_parse_espectre_config_command_updates_supported_fields);
  RUN_TEST(test_parse_espectre_config_command_rejects_invalid_inputs);
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
