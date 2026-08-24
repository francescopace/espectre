/*
 * ESPectre - ESP-IDF Direct WebSocket Service Tests
 *
 * Author: Francesco Pace <francesco.pace@gmail.com>
 * SPDX-License-Identifier: GPL-3.0-only
 * Commercial licensing available under separate agreement; see LICENSING.md.
 */
#include "test_harness.h"

#include <string>

#include "direct_websocket_service_esp_idf.h"
#include "esp_http_server.h"
#include "esp_timer.h"

using namespace espectre;

namespace {

DirectWebSocketServiceConfig config() {
  return {{"https://espectre.dev"}, 80U, 2U, 2U, 2U, false, true};
}

httpd_req_t request_for(EspIdfDirectWebSocketService &service, int fd = 7) {
  (void) service;
  return httpd_req_t{g_httpd_mock.registered_uri.user_ctx, fd};
}

void allow_portal_handshake() {
  httpd_mock_set_header("Origin", "https://espectre.dev");
  httpd_mock_set_header("Sec-WebSocket-Protocol", "chat, espectre.v1");
}

void test_setup_validates_configuration_and_registers_versioned_endpoint() {
  httpd_mock_reset();
  EspIdfDirectWebSocketService service;
  DirectWebSocketServiceConfig invalid_config;
  invalid_config.max_clients = 0U;
  TEST_ASSERT_FALSE(service.setup(invalid_config, [](const auto &) { return std::string{}; }, {}));
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  TEST_ASSERT_TRUE(service.running());
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DIRECT_WEBSOCKET_ENDPOINT, g_httpd_mock.registered_uri.uri);
  TEST_ASSERT_EQUAL_STRING(ESPECTRE_DIRECT_WEBSOCKET_SUBPROTOCOL,
                           g_httpd_mock.registered_uri.supported_subprotocol);
  TEST_ASSERT_EQUAL(5U, g_httpd_mock.last_config.max_open_sockets);
  service.shutdown();
  TEST_ASSERT_FALSE(service.running());
  TEST_ASSERT_EQUAL(1, g_httpd_mock.stop_calls);
}

void test_handshake_enforces_subprotocol_origin_and_client_limit() {
  httpd_mock_reset();
  EspIdfDirectWebSocketService service;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  httpd_req_t request = request_for(service);

  httpd_mock_set_header("Origin", "https://evil.example");
  httpd_mock_set_header("Sec-WebSocket-Protocol", "espectre.v1");
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uri.ws_pre_handshake_cb(&request));
  TEST_ASSERT_EQUAL_STRING("403 Forbidden", g_httpd_mock.response_status);

  allow_portal_handshake();
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.ws_pre_handshake_cb(&request));

  const char *allowed_loopback_origins[] = {
      "http://localhost", "http://localhost:8080", "http://localhost:8090",
      "http://127.0.0.1:49152", "http://[::1]:3000", "HTTP://LOCALHOST:8080"};
  for (const char *origin : allowed_loopback_origins) {
    httpd_mock_set_header("Origin", origin);
    TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.ws_pre_handshake_cb(&request));
  }

  const char *rejected_loopback_lookalikes[] = {
      "https://localhost:8080", "http://localhost.evil:8080", "http://localhost@evil.example:8080",
      "http://localhost:0", "http://localhost:65536", "http://localhost:8080/path"};
  for (const char *origin : rejected_loopback_lookalikes) {
    httpd_mock_set_header("Origin", origin);
    TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uri.ws_pre_handshake_cb(&request));
  }

  allow_portal_handshake();

  const int clients[] = {7, 8};
  httpd_mock_set_clients(clients, 2U);
  httpd_req_t extra = request_for(service, 9);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uri.ws_pre_handshake_cb(&extra));
  service.loop();
  TEST_ASSERT_EQUAL(2U, service.client_count());
}

void test_dispatches_valid_request_and_sends_correlated_response_asynchronously() {
  httpd_mock_reset();
  EspIdfDirectWebSocketService service;
  std::string dispatched_method;
  TEST_ASSERT_TRUE(service.setup(
      config(),
      [&dispatched_method](const DirectWebSocketRequest &request) {
        dispatched_method = request.method;
        return direct_websocket_success_response(request.id, "{\"methods\":[]}");
      },
      {}));
  allow_portal_handshake();
  httpd_req_t request = request_for(service);
  const int clients[] = {7};
  httpd_mock_set_clients(clients, 1U);
  service.loop();
  httpd_mock_set_incoming(
      "{\"v\":1,\"type\":\"request\",\"id\":\"req-1\",\"method\":\"capabilities\",\"params\":{}}",
      HTTPD_WS_TYPE_TEXT,
      true,
      false);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
  service.loop();
  TEST_ASSERT_EQUAL_STRING("capabilities", dispatched_method.c_str());
  TEST_ASSERT_EQUAL(1, g_httpd_mock.send_calls);
  TEST_ASSERT_TRUE(std::string(g_httpd_mock.sent_payloads[0]).find("\"id\":\"req-1\"") != std::string::npos);
}

void test_first_request_notifies_client_count_before_loop_sync() {
  httpd_mock_reset();
  EspIdfDirectWebSocketService service;
  size_t reported_clients = 0U;
  TEST_ASSERT_TRUE(service.setup(
      config(),
      [](const auto &request) { return direct_websocket_success_response(request.id, "{}"); },
      [&reported_clients](size_t count) { reported_clients = count; }));
  httpd_req_t request = request_for(service);
  httpd_mock_set_incoming(
      "{\"v\":1,\"type\":\"request\",\"id\":\"cap\",\"method\":\"capabilities\",\"params\":{}}",
      HTTPD_WS_TYPE_TEXT,
      true,
      false);

  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
  TEST_ASSERT_EQUAL(1U, service.client_count());
  TEST_ASSERT_EQUAL(1U, reported_clients);
}

void test_rejects_bad_frames_and_rate_limits_mutations() {
  httpd_mock_reset();
  esp_timer_mock::reset(0, 1000);
  EspIdfDirectWebSocketService service;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &request) {
    return direct_websocket_success_response(request.id, "{}");
  }, {}));
  const int clients[] = {7};
  httpd_mock_set_clients(clients, 1U);
  service.loop();
  httpd_req_t request = request_for(service);

  std::string oversized(ESPECTRE_DIRECT_MAX_FRAME_SIZE + 1U, 'x');
  httpd_mock_set_incoming(oversized.c_str(), HTTPD_WS_TYPE_TEXT, true, false);
  TEST_ASSERT_EQUAL(ESP_FAIL, g_httpd_mock.registered_uri.handler(&request));
  TEST_ASSERT_EQUAL(1U, service.diagnostics().oversized_frames);

  const char *mutation =
      "{\"v\":1,\"type\":\"request\",\"id\":\"m1\",\"method\":\"start_sensing\",\"params\":{}}";
  httpd_mock_set_incoming(mutation, HTTPD_WS_TYPE_TEXT, true, false);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
  httpd_mock_set_incoming(
      "{\"v\":1,\"type\":\"request\",\"id\":\"m2\",\"method\":\"stop_sensing\",\"params\":{}}",
      HTTPD_WS_TYPE_TEXT,
      true,
      false);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
  httpd_mock_set_incoming(
      "{\"v\":1,\"type\":\"request\",\"id\":\"m3\",\"method\":\"recalibrate\",\"params\":{}}",
      HTTPD_WS_TYPE_TEXT,
      true,
      false);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
  TEST_ASSERT_EQUAL(1U, service.diagnostics().rate_limited_requests);
  service.loop();
  TEST_ASSERT_TRUE(std::string(g_httpd_mock.sent_payloads[0]).find("\"id\":\"m3\"") != std::string::npos);
  TEST_ASSERT_TRUE(std::string(g_httpd_mock.sent_payloads[0]).find("\"code\":\"rate_limited\"") !=
                   std::string::npos);
}

void test_read_requests_do_not_consume_the_mutation_budget() {
  httpd_mock_reset();
  esp_timer_mock::reset(0, 1000);
  EspIdfDirectWebSocketService service;
  DirectWebSocketServiceConfig limited = config();
  limited.max_mutations_per_minute = 1U;
  limited.outbound_queue_depth = 16U;
  TEST_ASSERT_TRUE(service.setup(limited, [](const auto &request) {
    return direct_websocket_success_response(request.id, "{}");
  }, {}));
  const int clients[] = {7};
  httpd_mock_set_clients(clients, 1U);
  service.loop();
  httpd_req_t request = request_for(service);

  const char *read_methods[] = {
      "capabilities", "info", "commands", "status", "config", "diagnostics", "stats", "ota_status"};
  for (size_t index = 0U; index < sizeof(read_methods) / sizeof(read_methods[0]); ++index) {
    const std::string frame = "{\"v\":1,\"type\":\"request\",\"id\":\"r" + std::to_string(index) +
                              "\",\"method\":\"" + read_methods[index] + "\",\"params\":{}}";
    httpd_mock_set_incoming(frame.c_str(), HTTPD_WS_TYPE_TEXT, true, false);
    TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
    service.loop();
  }

  httpd_mock_set_incoming(
      "{\"v\":1,\"type\":\"request\",\"id\":\"m1\",\"method\":\"start_sensing\",\"params\":{}}",
      HTTPD_WS_TYPE_TEXT,
      true,
      false);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
  service.loop();
  httpd_mock_set_incoming(
      "{\"v\":1,\"type\":\"request\",\"id\":\"m2\",\"method\":\"stop_sensing\",\"params\":{}}",
      HTTPD_WS_TYPE_TEXT,
      true,
      false);
  TEST_ASSERT_EQUAL(ESP_OK, g_httpd_mock.registered_uri.handler(&request));
  service.loop();

  TEST_ASSERT_EQUAL(1U, service.diagnostics().rate_limited_requests);
  TEST_ASSERT_TRUE(std::string(g_httpd_mock.sent_payloads[9]).find("\"id\":\"m2\"") != std::string::npos);
  TEST_ASSERT_TRUE(std::string(g_httpd_mock.sent_payloads[9]).find("\"code\":\"rate_limited\"") !=
                   std::string::npos);
}

void test_coalesces_telemetry_and_cleans_up_disconnected_clients() {
  httpd_mock_reset();
  EspIdfDirectWebSocketService service;
  size_t reported_clients = 99U;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; },
                                 [&reported_clients](size_t count) { reported_clients = count; }));
  const int clients[] = {7};
  httpd_mock_set_clients(clients, 1U);
  service.loop();
  TEST_ASSERT_EQUAL(1U, reported_clients);
  TEST_ASSERT_TRUE(service.publish_event("telemetry", "{\"movement\":0.1}", true));
  TEST_ASSERT_TRUE(service.publish_event("telemetry", "{\"movement\":0.9}", true));
  service.loop();
  TEST_ASSERT_EQUAL(1, g_httpd_mock.send_calls);
  TEST_ASSERT_TRUE(std::string(g_httpd_mock.sent_payloads[0]).find("0.9") != std::string::npos);

  httpd_mock_set_clients(nullptr, 0U);
  service.loop();
  TEST_ASSERT_EQUAL(0U, service.client_count());
  TEST_ASSERT_EQUAL(0U, reported_clients);
  service.shutdown();
  TEST_ASSERT_EQUAL(0U, service.diagnostics().queued_messages);
}

void test_disconnects_a_client_after_repeated_async_send_failures() {
  httpd_mock_reset();
  EspIdfDirectWebSocketService service;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  TEST_ASSERT_EQUAL(2U, service.diagnostics().client_limit);
  TEST_ASSERT_EQUAL(2U, service.diagnostics().queue_capacity);
  const int clients[] = {7};
  httpd_mock_set_clients(clients, 1U);
  service.loop();
  TEST_ASSERT_TRUE(service.publish_event("status", "{\"online\":true}", false));
  g_httpd_mock.send_completion_result = ESP_FAIL;

  service.loop();
  service.loop();
  TEST_ASSERT_EQUAL(0, g_httpd_mock.trigger_close_calls);
  service.loop();

  const auto diagnostics = service.diagnostics();
  TEST_ASSERT_EQUAL(3U, diagnostics.send_failures);
  TEST_ASSERT_EQUAL(1U, diagnostics.slow_client_disconnects);
  TEST_ASSERT_EQUAL(1, g_httpd_mock.trigger_close_calls);
  TEST_ASSERT_EQUAL(7, g_httpd_mock.last_closed_fd);
}

void test_allows_only_one_async_send_in_flight_per_client() {
  httpd_mock_reset();
  EspIdfDirectWebSocketService service;
  TEST_ASSERT_TRUE(service.setup(config(), [](const auto &) { return std::string{"{}"}; }, {}));
  const int clients[] = {7};
  httpd_mock_set_clients(clients, 1U);
  service.loop();
  TEST_ASSERT_TRUE(service.publish_event("status", "{\"sequence\":1}", false));
  TEST_ASSERT_TRUE(service.publish_event("state", "{\"sequence\":2}", false));
  g_httpd_mock.defer_send_completions = true;

  service.loop();
  TEST_ASSERT_EQUAL(1, g_httpd_mock.send_calls);
  TEST_ASSERT_EQUAL(1U, g_httpd_mock.pending_send_completions);
  service.loop();
  service.loop();
  TEST_ASSERT_EQUAL(1, g_httpd_mock.send_calls);
  TEST_ASSERT_EQUAL(1U, g_httpd_mock.pending_send_completions);

  httpd_mock_complete_next_send(ESP_OK);
  service.loop();
  TEST_ASSERT_EQUAL(2, g_httpd_mock.send_calls);
  TEST_ASSERT_EQUAL(1U, g_httpd_mock.pending_send_completions);
  httpd_mock_complete_next_send(ESP_OK);
}

}  // namespace

int main() {
  espectre::test::begin_suite();
  RUN_TEST(test_setup_validates_configuration_and_registers_versioned_endpoint);
  RUN_TEST(test_handshake_enforces_subprotocol_origin_and_client_limit);
  RUN_TEST(test_dispatches_valid_request_and_sends_correlated_response_asynchronously);
  RUN_TEST(test_first_request_notifies_client_count_before_loop_sync);
  RUN_TEST(test_rejects_bad_frames_and_rate_limits_mutations);
  RUN_TEST(test_read_requests_do_not_consume_the_mutation_budget);
  RUN_TEST(test_coalesces_telemetry_and_cleans_up_disconnected_clients);
  RUN_TEST(test_disconnects_a_client_after_repeated_async_send_failures);
  RUN_TEST(test_allows_only_one_async_send_in_flight_per_client);
  return espectre::test::end_suite();
}
