/*
 * ESPectre - Native Direct Frontend Tests
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "native_frontend_test_support.h"

#include "esp_wifi.h"

void test_native_frontend_direct_service_follows_station_address_lifecycle(void) {
  MockDirectHttpService direct;
  NativeFrontend frontend(nullptr, nullptr, &direct);
  NativeFrontend::WifiProvisioningInfo wifi;
  wifi.ssid = "Lab";
  wifi.has_saved_config = true;
  frontend.set_wifi_provisioning_info(wifi);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_EQUAL(0, direct_http_service_mock::state.setup_calls);

  EspectreDeviceInfo info;
  info.frontend = "native";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_EQUAL(1, direct_http_service_mock::state.setup_calls);
  TEST_ASSERT_TRUE(direct_http_service_mock::state.running);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DIRECT_HTTP_REQUEST_ENDPOINT, "/espectre/v1/request");
  TEST_ASSERT_EQUAL(2, static_cast<int>(direct_http_service_mock::state.last_config.max_event_clients));
  TEST_ASSERT_EQUAL(8, static_cast<int>(direct_http_service_mock::state.last_config.outbound_queue_depth));
  TEST_ASSERT_FALSE(direct_http_service_mock::state.last_config.allow_missing_origin);
  const auto expected_origins = DirectHttpServiceConfig::for_first_party_portals().allowed_origins;
  TEST_ASSERT_EQUAL(expected_origins.size(),
                    direct_http_service_mock::state.last_config.allowed_origins.size());
  for (const auto &origin : expected_origins) {
    TEST_ASSERT_TRUE(std::find(direct_http_service_mock::state.last_config.allowed_origins.begin(),
                               direct_http_service_mock::state.last_config.allowed_origins.end(),
                               origin) != direct_http_service_mock::state.last_config.allowed_origins.end());
  }

  direct.emit_client_count(1U);
  TEST_ASSERT_EQUAL(1, static_cast<int>(frontend.direct_client_count()));
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.live_telemetry_enabled);

  info.network.ip_address.clear();
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(direct_http_service_mock::state.shutdown_called);
  TEST_ASSERT_EQUAL(0, static_cast<int>(frontend.direct_client_count()));
}

void test_native_frontend_direct_requests_share_command_dispatch_and_return_correlated_results(void) {
  MockDirectHttpService direct;
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

  const std::string info_response = direct.emit_request(DirectRequest{"req-info", "info", "{}"});
  TEST_ASSERT_TRUE(info_response.find("\"command_id\":\"req-info\"") != std::string::npos);
  TEST_ASSERT_TRUE(info_response.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(info_response.find("3.0.0-test") != std::string::npos);

  const std::string update_response = direct.emit_request(
      DirectRequest{"req-label", "set_device_label", "{\"device_label\":\"Kitchen\"}"});
  TEST_ASSERT_EQUAL_STRING("Kitchen", saved_label.c_str());
  TEST_ASSERT_TRUE(update_response.find("\"command_id\":\"req-label\"") != std::string::npos);
  TEST_ASSERT_TRUE(update_response.find("\"accepted\":true") != std::string::npos);

  const std::string invalid_response =
      direct.emit_request(DirectRequest{"req-bad", "not_supported", "{}"});
  TEST_ASSERT_TRUE(invalid_response.find("\"command_id\":\"req-bad\"") != std::string::npos);
  TEST_ASSERT_TRUE(invalid_response.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_TRUE(invalid_response.find("\"code\":\"unsupported\"") != std::string::npos);

  const std::string version_response =
      direct.emit_request(DirectRequest{"req-version", "info", "{}", "", "2.0"});
  TEST_ASSERT_TRUE(version_response.find("\"command_id\":\"req-version\"") != std::string::npos);
  TEST_ASSERT_TRUE(version_response.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_TRUE(version_response.find("\"code\":\"unsupported_version\"") != std::string::npos);
}

void test_native_frontend_peer_discovery_is_capability_gated_correlated_and_bounded(void) {
  MockDirectHttpService direct;
  MockPeerDiscoveryService peers;
  NativeFrontend frontend(nullptr, nullptr, &direct);
  frontend.set_peer_discovery_service(&peers);
  EspectreDeviceConfig config;
  config.device_id = 0x0123456789abcdefULL;
  frontend.set_device_config(config);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.chip = "esp32c3";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_EQUAL_STRING("ESPectre C3 abcdef", peers.local_candidate.name.c_str());
  TEST_ASSERT_EQUAL_STRING("ESPectre 0123456789abcdef", peers.local_candidate.instance.c_str());
  direct.emit_client_count(1U);

  auto capabilities = direct.emit_deferred_request(
      77U, DirectRequest{"caps", "capabilities", "{}"});
  TEST_ASSERT_FALSE(capabilities.deferred);
  TEST_ASSERT_TRUE(capabilities.response.find("\"discover_peers\"") != std::string::npos);

  auto request = direct.emit_deferred_request(
      77U, DirectRequest{"peers-1", "discover_peers", "{}"});
  TEST_ASSERT_TRUE(request.deferred);
  TEST_ASSERT_TRUE(peers.active());
  auto invalid = direct.emit_deferred_request(
      77U, DirectRequest{"peers-invalid", "discover_peers", "{\"unexpected\":true}"});
  TEST_ASSERT_FALSE(invalid.deferred);
  TEST_ASSERT_TRUE(invalid.response.find("\"code\":\"invalid_params\"") != std::string::npos);
  auto conflict = direct.emit_deferred_request(
      88U, DirectRequest{"peers-2", "discover_peers", "{}"});
  TEST_ASSERT_FALSE(conflict.deferred);
  TEST_ASSERT_TRUE(conflict.response.find("\"code\":\"conflict\"") != std::string::npos);

  PeerDiscoverySnapshot snapshot;
  snapshot.elapsed_ms = 42U;
  snapshot.timed_out = true;
  peers.finish(snapshot);
  TEST_ASSERT_EQUAL(77U, direct_http_service_mock::state.last_completed_token);
  TEST_ASSERT_TRUE(direct_http_service_mock::state.last_deferred_response.find(
                       "\"command_id\":\"peers-1\"") != std::string::npos);
  TEST_ASSERT_TRUE(direct_http_service_mock::state.last_deferred_response.find(
                       "\"elapsed_ms\":42") != std::string::npos);
  TEST_ASSERT_TRUE(direct_http_service_mock::state.last_deferred_response.find(
                       "\"status\":\"timeout\"") != std::string::npos);

  peers.start_result = false;
  auto unavailable = direct.emit_deferred_request(
      77U, DirectRequest{"peers-unavailable", "discover_peers", "{}"});
  TEST_ASSERT_FALSE(unavailable.deferred);
  TEST_ASSERT_TRUE(unavailable.response.find("\"code\":\"unavailable\"") != std::string::npos);
}

void test_native_frontend_peer_discovery_drops_completion_after_wifi_loss_and_shutdown(void) {
  MockDirectHttpService direct;
  MockPeerDiscoveryService peers;
  NativeFrontend frontend(nullptr, nullptr, &direct);
  frontend.set_peer_discovery_service(&peers);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);

  auto request = direct.emit_deferred_request(
      99U, DirectRequest{"peers-wifi-loss", "discover_peers", "{}"});
  TEST_ASSERT_TRUE(request.deferred);
  TEST_ASSERT_TRUE(peers.active());
  info.network.ip_address.clear();
  frontend.set_device_info(info);
  TEST_ASSERT_FALSE(direct_http_service_mock::state.running);
  TEST_ASSERT_FALSE(peers.wifi_ready);
  peers.finish(PeerDiscoverySnapshot{});
  TEST_ASSERT_EQUAL(0U, direct_http_service_mock::state.last_completed_token);

  frontend.shutdown();
  TEST_ASSERT_EQUAL(1U, peers.shutdown_calls);
}

void test_native_frontend_direct_updates_bssid_and_mqtt_without_returning_secrets(void) {
  MockMqttTransport mqtt;
  MockDirectHttpService direct;
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
  int scan_calls = 0;
  frontend.set_wifi_scan_callback([&scan_calls](std::string *message) {
    ++scan_calls;
    if (message != nullptr) *message = "scan started";
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

  g_esp_wifi_mock.current_ap_info.bssid[0] = 0xE6;
  g_esp_wifi_mock.current_ap_info.bssid[1] = 0xFA;
  g_esp_wifi_mock.current_ap_info.bssid[2] = 0xC4;
  g_esp_wifi_mock.current_ap_info.bssid[3] = 0x20;
  g_esp_wifi_mock.current_ap_info.bssid[4] = 0x19;
  g_esp_wifi_mock.current_ap_info.bssid[5] = 0xDE;

  auto wifi_request = direct.emit_deferred_request(
      77U, DirectRequest{"wifi-1", "set_wifi_bssid",
                         "{\"bssid\":\"E6:FA:C4:20:19:DE\",\"force\":true}"});
  TEST_ASSERT_TRUE(provisioning_command.empty());
  TEST_ASSERT_TRUE(wifi_request.response.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(
      wifi_request.response.find("\"current_bssid\":\"E6:FA:C4:20:19:DE\"") !=
      std::string::npos);
  TEST_ASSERT_TRUE(static_cast<bool>(wifi_request.response_sent_callback));
  wifi_request.response_sent_callback(true);
  TEST_ASSERT_EQUAL_STRING(
      "SET_WIFI_BSSID:bssid=E6%3AFA%3AC4%3A20%3A19%3ADE&force=true",
      provisioning_command.c_str());
  TEST_ASSERT_TRUE(wifi_request.response.find("password") == std::string::npos);

  const std::string scan_response = direct.emit_request(
      DirectRequest{"wifi-scan", "scan_wifi_access_points", "{}"});
  TEST_ASSERT_EQUAL(1, scan_calls);
  TEST_ASSERT_TRUE(scan_response.find("\"accepted\":true") != std::string::npos);

  const std::string clear_bssid_response = direct.emit_request(
      DirectRequest{"wifi-bssid-clear", "clear_wifi_bssid", "{}"});
  TEST_ASSERT_EQUAL_STRING("SET_WIFI_BSSID:bssid=", provisioning_command.c_str());
  TEST_ASSERT_TRUE(clear_bssid_response.find("\"accepted\":true") != std::string::npos);

  const std::string removed_command = direct.emit_request(
      DirectRequest{"wifi-removed", "clear_wifi_config", "{}"});
  TEST_ASSERT_EQUAL_STRING("CLEAR_WIFI", provisioning_command.c_str());
  TEST_ASSERT_TRUE(removed_command.find("\"accepted\":true") != std::string::npos);

  const std::string mqtt_response = direct.emit_request(DirectRequest{
      "mqtt-1",
      "set_mqtt_config",
      "{\"scheme\":\"mqtt\",\"host\":\"homeassistant.local\",\"port\":1883,\"username\":\"mqtt\",\"password\":\"secret\"}"});
  TEST_ASSERT_EQUAL_STRING("mqtt", persisted.mqtt_scheme.c_str());
  TEST_ASSERT_EQUAL_STRING("homeassistant.local", persisted.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(1883U, persisted.mqtt_port);
  TEST_ASSERT_EQUAL_STRING("mqtt", persisted.mqtt_username.c_str());
  TEST_ASSERT_EQUAL_STRING("secret", persisted.mqtt_password.c_str());
  TEST_ASSERT_EQUAL(1, mqtt_transport_mock::state.setup_calls);
  TEST_ASSERT_TRUE(mqtt_response.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_response.find("secret") == std::string::npos);

  const std::string invalid_mqtt_response = direct.emit_request(DirectRequest{
      "mqtt-invalid", "set_mqtt_config",
      "{\"scheme\":\"mqtts\",\"host\":\"mqtts://broker.example.com\",\"port\":8883}"});
  TEST_ASSERT_TRUE(invalid_mqtt_response.find("\"accepted\":false") != std::string::npos);
  TEST_ASSERT_EQUAL_STRING("mqtt", persisted.mqtt_scheme.c_str());
  TEST_ASSERT_EQUAL_STRING("homeassistant.local", persisted.mqtt_host.c_str());
  TEST_ASSERT_EQUAL(1, mqtt_transport_mock::state.setup_calls);
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
  MockDirectHttpService direct;
  direct_http_service_mock::state.diagnostics.accepted_connections = 4U;
  direct_http_service_mock::state.diagnostics.event_client_limit = 2U;
  direct_http_service_mock::state.diagnostics.queue_capacity = 8U;
  direct_http_service_mock::state.diagnostics.dropped_telemetry_events = 3U;
  mqtt_transport_mock::state.diagnostics.queued_publishes = 5U;
  mqtt_transport_mock::state.diagnostics.queue_capacity = 16U;
  mqtt_transport_mock::state.diagnostics.outbox_capacity_bytes = 8192U;
  mqtt_transport_mock::state.diagnostics.dropped_publishes = 6U;
  mqtt_transport_mock::state.diagnostics.publish_failures = 7U;
  mqtt_transport_mock::state.diagnostics.reconnects = 8U;
  NativeFrontend frontend(&mqtt, nullptr, &direct);

  EspectreDeviceConfig config;
  config.device_label = "Kitchen";
  config.mqtt_scheme = "mqtt";
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
  wifi.scan_message = "Wi-Fi access point scan complete";
  wifi.access_points.push_back(
      NativeFrontend::WifiProvisioningInfo::AccessPoint{"AA:BB:CC:DD:EE:FF", -43, 6U});
  frontend.set_wifi_provisioning_info(wifi);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.network.ip_address = "192.168.1.42";
  info.network.channel = 36U;
  frontend.set_device_info(info);
  g_esp_wifi_mock.get_ap_info_result = ESP_ERR_WIFI_NOT_CONNECT;
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);

  const std::string capabilities =
      direct.emit_request(DirectRequest{"read-cap", "capabilities", "{}"});
  TEST_ASSERT_TRUE(capabilities.find("\"name\":\"set_sensing\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("\"raw_csi\":false") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("\"name\":\"wifi_access_points\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("\"name\":\"scan_wifi_access_points\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("\"name\":\"set_wifi_bssid\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("\"name\":\"clear_wifi_bssid\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("\"name\":\"clear_wifi_config\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.find("set_wifi_config") == std::string::npos);

  const std::string status = direct.emit_request(DirectRequest{"read-status", "status", "{}"});
  TEST_ASSERT_TRUE(status.find("\"wifi_connected\":true") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"mqtt_configured\":true") != std::string::npos);
  TEST_ASSERT_TRUE(status.find("\"sensing_enabled\":true") != std::string::npos);

  const std::string visible_config =
      direct.emit_request(DirectRequest{"read-config", "config", "{}"});
  TEST_ASSERT_TRUE(visible_config.find("AA:BB:CC:DD:EE:FF") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("broker.local") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"scheme\":\"mqtt\"") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"apply_state\":\"rolled_back\"") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("last-known-good configuration restored") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"username_configured\":true") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("private-user") == std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("private-password") == std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"password\"") == std::string::npos);

  TEST_ASSERT_TRUE(visible_config.find("\"connected\":true") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"ssid\":\"Lab\"") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"band\":\"5g\"") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"channel\":36") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"rssi_dbm\":null") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"band_policy\"") == std::string::npos);

  const std::string access_points =
      direct.emit_request(DirectRequest{"read-wifi-aps", "wifi_access_points", "{}"});
  TEST_ASSERT_TRUE(access_points.find("AA:BB:CC:DD:EE:FF") != std::string::npos);
  TEST_ASSERT_TRUE(access_points.find("\"rssi_dbm\":-43") != std::string::npos);
  TEST_ASSERT_TRUE(access_points.find("\"channel\":6") != std::string::npos);

  const std::string diagnostics =
      direct.emit_request(DirectRequest{"read-diag", "diagnostics", "{}"});
  TEST_ASSERT_TRUE(diagnostics.find("\"mqtt\":{") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("broker.local") == std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"scheme\"") == std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"event_clients\":1") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"minimum_free_memory_kb\":") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"largest_free_memory_kb\":1") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"cpu_frequency_mhz\":160") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"runtime_load_percent\":7.5") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"loop_avg_us\":75") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"detection_samples\":40") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"detection_max_us\":140") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"task_stack_high_water_bytes\":") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"event_client_limit\":2") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"queue_capacity\":8") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"accepted_connections\":4") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"dropped_telemetry_events\":3") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"queued_publishes\":5") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"queue_capacity\":16") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"outbox_capacity_bytes\":8192") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"dropped_publishes\":6") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"publish_failures\":7") != std::string::npos);
  TEST_ASSERT_TRUE(diagnostics.find("\"reconnects\":8") != std::string::npos);
}

void test_native_frontend_direct_set_sensing_is_correlated(void) {
  MockDirectHttpService direct;
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
      direct.emit_request(DirectRequest{"sense-stop", "set_sensing", "{\"enabled\":false}"});
  TEST_ASSERT_TRUE(stopped.find("\"command_id\":\"sense-stop\"") != std::string::npos);
  TEST_ASSERT_TRUE(stopped.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_FALSE(frontend_runtime_shim::state.services_armed);

  const std::string started =
      direct.emit_request(DirectRequest{"sense-start", "set_sensing", "{\"enabled\":true}"});
  TEST_ASSERT_TRUE(started.find("\"command_id\":\"sense-start\"") != std::string::npos);
  TEST_ASSERT_TRUE(started.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(frontend_runtime_shim::state.services_armed);
}

void test_native_frontend_direct_raw_session_enforces_owner_and_keeps_mqtt_quiet(void) {
  frontend_runtime_shim::state.snapshot = make_ready_snapshot();
  frontend_runtime_shim::state.capabilities.supports_raw_csi = true;
  MockMqttTransport mqtt;
  MockDirectHttpService direct;
  EspectreDeviceConfig config;
  config.device_id = 0x112233445566ULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.firmware_version = "test";
  info.chip = "esp32c3";
  info.network.ip_address = "192.168.1.23";

  NativeFrontend frontend(&mqtt, nullptr, &direct);
  frontend.set_device_config(config);
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(frontend.capabilities().supports_raw_csi);
  direct.emit_client_count(2U);
  mqtt.emit_connection(true);
  mqtt_transport_mock::state.publishes.clear();

  const auto capabilities = direct.emit_deferred_request(
      77U, DirectRequest{"raw-cap", "capabilities", "{}"});
  TEST_ASSERT_TRUE(capabilities.response.find("\"raw_csi\":true") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.response.find("\"protocol_version\":1") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.response.find("\"record_version\":8") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.response.find("\"transport\":\"http\"") != std::string::npos);
  TEST_ASSERT_TRUE(capabilities.response.find("subprotocol") == std::string::npos);

  const auto started = direct.emit_deferred_request(
      77U,
      DirectRequest{
          "raw-start", "start_raw_stream", "{}"});
  TEST_ASSERT_TRUE(started.response.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(started.response.find("\"session_id\"") != std::string::npos);
  TEST_ASSERT_TRUE(direct_http_service_mock::state.raw_session_active);
  TEST_ASSERT_EQUAL(RuntimeOperationState::RAW_COLLECTION, frontend.runtime_.operation_state());

  static constexpr char kHex[] = "0123456789abcdef";
  std::string bearer(ESPECTRE_RAW_CSI_SESSION_ID_BYTES * 2U, '0');
  for (size_t index = 0U; index < ESPECTRE_RAW_CSI_SESSION_ID_BYTES; ++index) {
    const uint8_t value = direct_http_service_mock::state.raw_config.session_id[index];
    bearer[index * 2U] = kHex[(value >> 4U) & 0x0fU];
    bearer[index * 2U + 1U] = kHex[value & 0x0fU];
  }

  frontend.on_live_telemetry(9.0f, 1.0f);
  frontend.loop();
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000112233445566/telemetry"));

  const auto busy = direct.emit_deferred_request(
      88U, DirectRequest{"raw-busy", "set_sensing", "{\"enabled\":false}"});
  TEST_ASSERT_TRUE(busy.response.find("\"code\":\"busy_raw_collection\"") !=
                   std::string::npos);
  const auto non_owner = direct.emit_deferred_request(
      88U,
      DirectRequest{"raw-stop-other",
                    "stop_raw_stream",
                    "{}",
                    "ffffffffffffffffffffffffffffffff"});
  TEST_ASSERT_TRUE(non_owner.response.find("\"code\":\"not_raw_session_owner\"") !=
                   std::string::npos);

  const auto stopped = direct.emit_deferred_request(
      77U, DirectRequest{"raw-stop", "stop_raw_stream", "{}", bearer});
  TEST_ASSERT_TRUE(stopped.response.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_FALSE(direct_http_service_mock::state.raw_session_active);
  TEST_ASSERT_EQUAL(RuntimeOperationState::SENSING, frontend.runtime_.operation_state());
}

void test_native_frontend_queries_stay_on_requesting_transport_and_mutations_fan_out(void) {
  MockMqttTransport mqtt;
  MockDirectHttpService direct;
  EspectreDeviceConfig config;
  config.device_id = 0x0000abcdeffedcbaULL;
  config.mqtt_scheme = "mqtt";
  config.mqtt_host = "localhost";
  config.mqtt_port = 1883U;
  NativeFrontend frontend(&mqtt, nullptr, &direct);
  frontend.set_device_config(config);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);
  TEST_ASSERT_TRUE(frontend.setup());
  direct.emit_client_count(1U);
  mqtt_transport_mock::state.publishes.clear();
  direct_http_service_mock::state.published_events.clear();

  const std::string query = direct.emit_request(DirectRequest{"status-only", "status", "{}"});
  TEST_ASSERT_TRUE(query.find("\"command_id\":\"status-only\"") != std::string::npos);
  TEST_ASSERT_TRUE(query.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(mqtt_transport_mock::state.publishes.empty());

  const std::string mutation = direct.emit_request(
      DirectRequest{"threshold-fanout", "set_threshold", "{\"threshold\":0.4}"});
  TEST_ASSERT_TRUE(mutation.find("\"accepted\":true") != std::string::npos);
  TEST_ASSERT_TRUE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/config"));
  TEST_ASSERT_TRUE(std::any_of(direct_http_service_mock::state.published_events.begin(),
                               direct_http_service_mock::state.published_events.end(),
                               [](const direct_http_service_mock::PublishedEvent &event) {
                                 return event.event_name == "config" &&
                                        event.data_json.find("\"runtime\"") != std::string::npos;
                               }));
  TEST_ASSERT_FALSE(has_mqtt_publish("espectre/v1/devices/0000abcdeffedcba/commands/result"));

  mqtt_transport_mock::state.publishes.clear();
  direct_http_service_mock::state.published_events.clear();
  const std::string sensing = direct.emit_request(
      DirectRequest{"sensing-fanout", "set_sensing", "{\"enabled\":false}"});
  TEST_ASSERT_TRUE(sensing.find("\"accepted\":true") != std::string::npos);
  const int status_index =
      mqtt_publish_index("espectre/v1/devices/0000abcdeffedcba/status");
  TEST_ASSERT_TRUE(status_index >= 0);
  TEST_ASSERT_EQUAL(1, static_cast<int>(direct_http_service_mock::state.published_events.size()));
  TEST_ASSERT_EQUAL_STRING(
      "status", direct_http_service_mock::state.published_events[0].event_name.c_str());
  TEST_ASSERT_EQUAL_STRING(
      mqtt_transport_mock::state.publishes[static_cast<size_t>(status_index)].payload.c_str(),
      direct_http_service_mock::state.published_events[0].data_json.c_str());
}

void test_native_frontend_keeps_direct_available_for_a_legacy_mqtt_endpoint(void) {
  MockMqttTransport mqtt;
  MockDirectHttpService direct;
  NativeFrontend frontend(&mqtt, nullptr, &direct);
  EspectreDeviceConfig legacy;
  legacy.mqtt_host = "broker.local";
  legacy.mqtt_port = 1883U;
  frontend.set_device_config(legacy);
  EspectreDeviceInfo info;
  info.frontend = "native";
  info.network.ip_address = "192.168.1.42";
  frontend.set_device_info(info);

  TEST_ASSERT_TRUE(frontend.setup());
  TEST_ASSERT_TRUE(direct_http_service_mock::state.running);
  TEST_ASSERT_EQUAL(0, mqtt_transport_mock::state.setup_calls);
  const std::string status = direct.emit_request(DirectRequest{"legacy-status", "status", "{}"});
  TEST_ASSERT_TRUE(status.find("\"mqtt_configured\":false") != std::string::npos);
  const std::string visible_config = direct.emit_request(DirectRequest{"legacy-config", "config", "{}"});
  TEST_ASSERT_TRUE(visible_config.find("\"scheme\":\"\"") != std::string::npos);
  TEST_ASSERT_TRUE(visible_config.find("\"host\":\"broker.local\"") != std::string::npos);
}

int main(int argc, char **argv) {
  (void) argc;
  (void) argv;
  UNITY_BEGIN();
  RUN_TEST(test_native_frontend_direct_service_follows_station_address_lifecycle);
  RUN_TEST(test_native_frontend_direct_requests_share_command_dispatch_and_return_correlated_results);
  RUN_TEST(test_native_frontend_peer_discovery_is_capability_gated_correlated_and_bounded);
  RUN_TEST(test_native_frontend_peer_discovery_drops_completion_after_wifi_loss_and_shutdown);
  RUN_TEST(test_native_frontend_direct_updates_bssid_and_mqtt_without_returning_secrets);
  RUN_TEST(test_native_frontend_direct_exposes_portal_reads_without_secrets);
  RUN_TEST(test_native_frontend_keeps_direct_available_for_a_legacy_mqtt_endpoint);
  RUN_TEST(test_native_frontend_direct_set_sensing_is_correlated);
  RUN_TEST(test_native_frontend_direct_raw_session_enforces_owner_and_keeps_mqtt_quiet);
  RUN_TEST(test_native_frontend_queries_stay_on_requesting_transport_and_mutations_fan_out);
  return UNITY_END();
}
